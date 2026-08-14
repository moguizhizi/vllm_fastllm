#!/usr/bin/env python3
"""比较FastLLM与vLLM的SM120动态FP8 W8A8线性算子性能。"""

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys


REPO_DIR = Path(__file__).resolve().parents[2]

# 前七项覆盖FastLLM SM120的M分支边界；其余形状来自vLLM
# tests/kernels/quantization/test_cutlass_scaled_mm.py::MNK_FACTORS。
CASES = [
    ("branch_m1", 1, 4096, 4096),
    ("branch_m16", 16, 4096, 4096),
    ("branch_m17", 17, 4096, 4096),
    ("branch_m32", 32, 4096, 4096),
    ("branch_m33", 33, 4096, 4096),
    ("branch_m256", 256, 4096, 4096),
    ("branch_m257", 257, 4096, 4096),
    ("vllm_m1_n256_k128", 1, 256, 128),
    ("vllm_m1_n16384_k1024", 1, 16384, 1024),
    ("vllm_m1_n24576_k496", 1, 24576, 496),
    ("vllm_m16_n256_k496", 16, 256, 496),
    ("vllm_m16_n16384_k128", 16, 16384, 128),
    ("vllm_m16_n24576_k4096", 16, 24576, 4096),
    ("vllm_m32_n8192_k4096", 32, 8192, 4096),
    ("vllm_m32_n16384_k4096", 32, 16384, 4096),
    ("vllm_m33_n1024_k1024", 33, 1024, 1024),
    ("vllm_m33_n8192_k128", 33, 8192, 128),
    ("vllm_m64_n2048_k496", 64, 2048, 496),
    ("vllm_m64_n16384_k1024", 64, 16384, 1024),
    ("vllm_m100_n8192_k496", 100, 8192, 496),
    ("vllm_m128_n32768_k4096", 128, 32768, 4096),
    ("vllm_m256_n4096_k4096", 256, 4096, 4096),
    ("vllm_m512_n256_k1024", 512, 256, 1024),
    ("vllm_m512_n8192_k4096", 512, 8192, 4096),
    ("vllm_m512_n16384_k128", 512, 16384, 128),
    ("vllm_m512_n24576_k128", 512, 24576, 128),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="SM120 W8A8 FastLLM/vLLM算子性能对比")
    parser.add_argument("--optest", default=str(REPO_DIR / "optest"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--outer-repeats", type=int, default=3)
    return parser.parse_args()


def run_fastllm(args, log_dir):
    pattern = re.compile(r"latency:\s+avg_ms=([0-9.eE+-]+)")
    rows = []
    env = os.environ.copy()
    env.update({
        "FASTLLM_CUDA_W8A8": "1",
        "FASTLLM_CUDA_W8A8_STRICT": "1",
        "FASTLLM_CUDA_W8A8_TRACE": "0",
    })
    for name, m, n, k in CASES:
        samples = []
        for repeat in range(args.outer_repeats):
            command = [
                args.optest, "--op", "linear_fp8_block128", "--device", "cuda:0",
                "--param", f"batch={m}", "--param", f"in={k}",
                "--param", f"out={n}", "--param", "weight_layout=perchannel",
                "--param", "input_type=bf16", "--param", "has_bias=0",
                "--warmup", str(args.warmup), "--iters", str(args.iters),
            ]
            result = subprocess.run(
                command, cwd=REPO_DIR, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
            (log_dir / f"fastllm-{name}-r{repeat}.log").write_text(
                result.stdout, encoding="utf-8")
            matches = pattern.findall(result.stdout)
            if not matches:
                raise RuntimeError(f"FastLLM未输出{name}的latency")
            samples.append(float(matches[-1]))
        rows.append({
            "case": name, "m": m, "n": n, "k": k,
            "fastllm_latency_ms": statistics.median(samples),
        })
        print(json.dumps(rows[-1]), flush=True)
    return rows


def run_vllm(args):
    import torch
    from vllm import _custom_ops as ops

    rows = []
    fp8 = torch.float8_e4m3fn
    for name, m, n, k in CASES:
        samples = []
        for _ in range(args.outer_repeats):
            torch.cuda.empty_cache()
            gc.collect()
            source = torch.empty(
                (m, k), device="cuda", dtype=torch.bfloat16).normal_()
            weight = torch.empty(
                (n, k), device="cuda", dtype=torch.bfloat16).normal_().to(fp8).t()
            weight_scale = torch.empty(
                (1, n), device="cuda", dtype=torch.float32).uniform_(0.001, 0.01)

            def run():
                quantized, token_scale = ops.scaled_fp8_quant(
                    source, scale=None, use_per_token_if_dynamic=True)
                return ops.cutlass_scaled_mm(
                    quantized, weight, token_scale, weight_scale,
                    torch.bfloat16, None)

            for _ in range(args.warmup):
                output = run()
            torch.cuda.synchronize()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(args.iters):
                output = run()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) / args.iters)
            del source, weight, weight_scale, output
        rows.append({
            "case": name, "m": m, "n": n, "k": k,
            "vllm_latency_ms": statistics.median(samples),
        })
        print(json.dumps(rows[-1]), flush=True)
    return rows


def write_reports(output_dir, fastllm_rows, vllm_rows):
    vllm = {row["case"]: row for row in vllm_rows}
    combined = []
    for row in fastllm_rows:
        other = vllm[row["case"]]
        ft_ms = row["fastllm_latency_ms"]
        vl_ms = other["vllm_latency_ms"]
        combined.append({**row, "vllm_latency_ms": vl_ms,
                         "fastllm_speedup": vl_ms / ft_ms})
    (output_dir / "operator-performance-compare.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8")
    with (output_dir / "operator-performance-compare.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    lines = [
        "# SM120 W8A8算子性能对比", "",
        "> 双方都计入BF16激活动态per-token FP8量化、per-channel权重",
        "> CUTLASS GEMM及输出；无bias。每项取外层重复的中位数。", "",
        "| Case | M | N | K | FastLLM(ms) | vLLM(ms) | FastLLM speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in combined:
        lines.append(
            f"| {row['case']} | {row['m']} | {row['n']} | {row['k']} | "
            f"{row['fastllm_latency_ms']:.6f} | {row['vllm_latency_ms']:.6f} | "
            f"{row['fastllm_speedup']:.4f}x |")
    report = "\n".join(lines) + "\n"
    (output_dir / "operator-performance-compare.md").write_text(
        report, encoding="utf-8")
    print(report)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fastllm_rows = run_fastllm(args, output_dir)
    # FastLLM均在子进程中运行并退出，父进程此时才加载vLLM，避免显存共存。
    vllm_rows = run_vllm(args)
    write_reports(output_dir, fastllm_rows, vllm_rows)


if __name__ == "__main__":
    main()
