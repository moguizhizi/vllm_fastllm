#!/usr/bin/env python3
"""分时运行FastLLM和vLLM，并生成相同负载下的整体性能对比表。"""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import requests


REPO_DIR = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREFILL = load_module("fastllm_benchmark_prefill", REPO_DIR / "test/benchmark/prefill.py")
DECODE = load_module("fastllm_benchmark_decode", REPO_DIR / "test/benchmark/decode.py")


def parse_args():
    parser = argparse.ArgumentParser(description="NVFP4 vLLM/FastLLM整体性能对比")
    parser.add_argument("--model", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--vllm-python", default=os.environ.get("NVFP4_VLLM_PYTHON", sys.executable))
    parser.add_argument("--fastllm-python", default=sys.executable)
    parser.add_argument("--flm-dtype", default="auto")
    parser.add_argument("--flm-atype", default="bfloat16")
    parser.add_argument("--flm-device", default="cuda")
    parser.add_argument("--prefill-repeat", type=int, default=256)
    parser.add_argument("--prefill-max-tokens", type=int, default=16)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--decode-prefill-length", type=int, default=512)
    parser.add_argument("--decode-max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    return parser.parse_args()


def run_logged(command, log_path, env=None):
    print(f"SUBPROCESS COMMAND: {shlex.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"COMMAND: {shlex.join(command)}\n\n")
        process = subprocess.Popen(
            command, cwd=REPO_DIR, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        status = process.wait()
    if status != 0:
        raise subprocess.CalledProcessError(status, command)


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def wait_server(base_url, process, timeout, server_log):
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            tail = server_log.read_text(encoding="utf-8", errors="replace")[-8000:]
            raise RuntimeError(f"vLLM服务提前退出，code={process.returncode}\n{tail}")
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=2)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as error:
            last_error = str(error)
        time.sleep(1)
    raise TimeoutError(f"等待vLLM启动超时: {last_error}")


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run_fastllm(args, result_dir):
    env = os.environ.copy()
    env.pop("FASTLLM_CUDA_NVFP4_TRACE", None)
    env["FASTLLM_CUDA_NVFP4_W4A4"] = "1"
    env["FASTLLM_CUDA_NVFP4_W4A4_STRICT"] = "1"
    env["FASTLLM_CUDA_MOE_NVFP4_W4A4"] = "1"
    env["FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT"] = "1"
    prefill_json = result_dir / "fastllm-prefill.json"
    decode_json = result_dir / "fastllm-decode.json"
    common = [args.model, "--dtype", args.flm_dtype, "--atype", args.flm_atype,
              "--device", args.flm_device, "--moe_device", args.flm_device]
    prefill_command = [
        args.fastllm_python, "test/benchmark/prefill.py", *common,
        "--prompt-repeat", str(args.prefill_repeat),
        "--max-tokens", str(args.prefill_max_tokens),
        "--result-json", str(prefill_json),
    ]
    decode_command = [
        args.fastllm_python, "test/benchmark/decode.py", *common,
        "--batch-size", str(args.decode_batch_size),
        "--max_batch", str(args.decode_batch_size),
        "--prefill-length", str(args.decode_prefill_length),
        "--max-tokens", str(args.decode_max_tokens),
        "--result-json", str(decode_json),
    ]
    run_logged(prefill_command, result_dir / "fastllm-prefill.log", env)
    run_logged(decode_command, result_dir / "fastllm-decode.log", env)
    return read_json(prefill_json), read_json(decode_json)


def run_vllm(args, result_dir):
    base_url = f"http://127.0.0.1:{args.port}"
    served_name = "nvfp4-performance"
    command = [
        args.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model, "--served-model-name", served_name,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--dtype", "auto", "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.decode_batch_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code", "--enforce-eager",
        "--linear-backend", "cutlass",
        "--enable-force-include-usage",
    ]
    env = os.environ.copy()
    # Linear固定使用vLLM原生CUTLASS；sampling单独禁用FlashInfer，避免
    # CUDA 12.8环境在SM120上触发FlashInfer运行时JIT。
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_USE_FLASHINFER_MOE_FP4"] = "0"
    server_log = result_dir / "vllm-server.log"
    print(f"SUBPROCESS COMMAND: {shlex.join(command)}", flush=True)
    log_handle = server_log.open("w", encoding="utf-8")
    log_handle.write(f"COMMAND: {shlex.join(command)}\n\n")
    log_handle.write("ENV: VLLM_USE_FLASHINFER_SAMPLER=0\n")
    log_handle.write("ENV: VLLM_USE_FLASHINFER_MOE_FP4=0\n\n")
    log_handle.flush()
    process = subprocess.Popen(
        command, cwd=REPO_DIR, env=env, stdout=log_handle,
        stderr=subprocess.STDOUT, text=True)
    try:
        wait_server(base_url, process, args.startup_timeout, server_log)
        warmup = PREFILL.warmup(base_url, served_name, "no-key", 3600, 8)
        print("vLLM warmup: " + json.dumps(warmup, ensure_ascii=False), flush=True)

        prefill_prompt = PREFILL.build_long_context(
            "FastLLM prefill benchmark context block. ", args.prefill_repeat,
            "请阅读以上上下文，并只回复“测试完成”。")
        prefill = PREFILL.run_chat_completion(
            base_url, served_name, "no-key", prefill_prompt,
            args.prefill_max_tokens, 3600)
        prefill.update({
            "case_name": "single", "backend": "vllm", "model_path": args.model,
            "prompt_repeat": args.prefill_repeat, "prompt_chars": len(prefill_prompt),
            "max_tokens": args.prefill_max_tokens,
        })

        decode_prompt = DECODE.build_prompt_by_chars(
            "FastLLM decode benchmark context block. ", args.decode_prefill_length,
            "请连续输出数字序列，不要解释。")
        decode = DECODE.run_decode_batch(
            base_url, served_name, "no-key", decode_prompt,
            args.decode_max_tokens, 3600, args.decode_batch_size, 0)
        decode.update({
            "case_name": "single", "backend": "vllm", "model_path": args.model,
            "prefill_length_chars": args.decode_prefill_length,
            "prompt_chars": len(decode_prompt), "batch_size": args.decode_batch_size,
            "server_max_batch": args.decode_batch_size,
            "max_tokens": args.decode_max_tokens,
        })
        (result_dir / "vllm-prefill.json").write_text(
            json.dumps(prefill, ensure_ascii=False, indent=2), encoding="utf-8")
        (result_dir / "vllm-decode.json").write_text(
            json.dumps(decode, ensure_ascii=False, indent=2), encoding="utf-8")
        print("vLLM prefill result: " + json.dumps(prefill, ensure_ascii=False), flush=True)
        print("vLLM decode result: " + json.dumps(decode, ensure_ascii=False), flush=True)
        return prefill, decode
    finally:
        stop_process(process)
        log_handle.close()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def make_report(result_dir, ft_prefill, ft_decode, vllm_prefill, vllm_decode):
    prefill_rows = [
        ("FastLLM", ft_prefill), ("vLLM", vllm_prefill),
    ]
    decode_rows = [
        ("FastLLM", ft_decode), ("vLLM", vllm_decode),
    ]
    lines = [
        "# NVFP4整体性能对比", "",
        "## 测试参数", "",
        "| 场景 | Prompt参数 | Batch | 最大生成tokens |",
        "| --- | ---: | ---: | ---: |",
        f"| Prefill | repeat={ft_prefill.get('prompt_repeat', '-')} | 1 | {ft_prefill['max_tokens']} |",
        f"| Decode | chars={ft_decode.get('prefill_length_chars', '-')} | "
        f"{ft_decode['batch_size']} | {ft_decode['max_tokens']} |",
        "",
        "## Prefill", "",
        "| 后端 | Prompt tokens | Output tokens | TTFT(ms) | Prefill(tok/s) | Decode(tok/s) | Total(s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for backend, item in prefill_rows:
        lines.append(
            f"| {backend} | {item['prompt_tokens']} | {item['completion_tokens']} | "
            f"{item['ttft'] * 1000:.3f} | {item['prefill_speed']:.2f} | "
            f"{item['decode_speed']:.2f} | {item['total_time']:.6f} |")
    lines.extend([
        "", "## Decode", "",
        "| 后端 | Batch | Prompt tokens | Output tokens | 平均TTFT(ms) | Batch decode(tok/s) | E2E(tok/s) | Wall(s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for backend, item in decode_rows:
        lines.append(
            f"| {backend} | {item['batch_size']} | {item['total_prompt_tokens']} | "
            f"{item['total_completion_tokens']} | {item['ttft_avg'] * 1000:.3f} | "
            f"{item['batch_decode_speed']:.2f} | {item['end_to_end_speed']:.2f} | "
            f"{item['batch_wall_time']:.6f} |")
    lines.extend([
        "", "## FastLLM / vLLM", "",
        "| 指标 | 比值 |",
        "| --- | ---: |",
        f"| Prefill吞吐 | {ratio(ft_prefill['prefill_speed'], vllm_prefill['prefill_speed']):.4f}x |",
        f"| Decode batch吞吐 | {ratio(ft_decode['batch_decode_speed'], vllm_decode['batch_decode_speed']):.4f}x |",
        f"| Decode E2E吞吐 | {ratio(ft_decode['end_to_end_speed'], vllm_decode['end_to_end_speed']):.4f}x |",
        f"| TTFT速度（vLLM TTFT / FastLLM TTFT） | {ratio(vllm_prefill['ttft'], ft_prefill['ttft']):.4f}x |",
        "",
    ])
    report = "\n".join(lines)
    (result_dir / "model-performance-compare.md").write_text(report, encoding="utf-8")
    rows = []
    for workload, values in (("prefill", prefill_rows), ("decode", decode_rows)):
        for backend, item in values:
            if workload == "prefill":
                rows.append({
                    "workload": workload, "backend": backend, "batch": 1,
                    "prompt_tokens": item["prompt_tokens"],
                    "output_tokens": item["completion_tokens"],
                    "ttft_ms": item["ttft"] * 1000,
                    "prefill_tok_s": item["prefill_speed"],
                    "decode_tok_s": item["decode_speed"],
                    "batch_decode_tok_s": "", "e2e_tok_s": "",
                    "wall_s": item["total_time"],
                })
            else:
                rows.append({
                    "workload": workload, "backend": backend,
                    "batch": item["batch_size"],
                    "prompt_tokens": item["total_prompt_tokens"],
                    "output_tokens": item["total_completion_tokens"],
                    "ttft_ms": item["ttft_avg"] * 1000,
                    "prefill_tok_s": "", "decode_tok_s": "",
                    "batch_decode_tok_s": item["batch_decode_speed"],
                    "e2e_tok_s": item["end_to_end_speed"],
                    "wall_s": item["batch_wall_time"],
                })
    with (result_dir / "model-performance-compare.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    combined = {"fastllm": {"prefill": ft_prefill, "decode": ft_decode},
                "vllm": {"prefill": vllm_prefill, "decode": vllm_decode}}
    (result_dir / "model-performance-compare.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report)
    print(f"Markdown: {result_dir / 'model-performance-compare.md'}")
    print(f"CSV: {result_dir / 'model-performance-compare.csv'}")


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    ft_prefill, ft_decode = run_fastllm(args, result_dir)
    vllm_prefill, vllm_decode = run_vllm(args, result_dir)
    make_report(result_dir, ft_prefill, ft_decode, vllm_prefill, vllm_decode)


if __name__ == "__main__":
    main()
