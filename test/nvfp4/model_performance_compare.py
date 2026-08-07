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
    parser.add_argument(
        "--decode-batch-sizes", default="",
        help="逗号分隔的decode batch矩阵；为空时使用--decode-batch-size")
    parser.add_argument("--decode-prefill-length", type=int, default=512)
    parser.add_argument("--decode-max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    return parser.parse_args()


def decode_batch_cases(args):
    values = ([args.decode_batch_size] if not args.decode_batch_sizes else
              [int(value.strip()) for value in args.decode_batch_sizes.split(",")
               if value.strip()])
    if not values or any(value <= 0 for value in values):
        raise ValueError("decode batch必须是正整数")
    return list(dict.fromkeys(values))


def render_shared_prompts(args, result_dir):
    """用同一个AutoTokenizer渲染完整prompt，供两个后端原样复用。"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    raw_prefill = PREFILL.build_long_context(
        "FastLLM prefill benchmark context block. ", args.prefill_repeat,
        "请阅读以上上下文，并只回复“测试完成”。")
    raw_decode = DECODE.build_prompt_by_chars(
        "FastLLM decode benchmark context block. ", args.decode_prefill_length,
        "请连续输出数字序列，不要解释。")

    def render(content):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    prompts = {"prefill": render(raw_prefill), "decode": render(raw_decode)}
    prompts["prefill_path"] = str(result_dir / "shared-prefill-prompt.txt")
    prompts["decode_path"] = str(result_dir / "shared-decode-prompt.txt")
    Path(prompts["prefill_path"]).write_text(prompts["prefill"], encoding="utf-8")
    Path(prompts["decode_path"]).write_text(prompts["decode"], encoding="utf-8")
    metadata = {
        "tokenizer": tokenizer.__class__.__name__,
        "prefill_chars": len(prompts["prefill"]),
        "decode_chars": len(prompts["decode"]),
        "prefill_tokens_by_tokenizer": len(tokenizer.encode(
            prompts["prefill"], add_special_tokens=False)),
        "decode_tokens_by_tokenizer": len(tokenizer.encode(
            prompts["decode"], add_special_tokens=False)),
    }
    (result_dir / "shared-prompt-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Shared prompt metadata: " + json.dumps(metadata, ensure_ascii=False), flush=True)
    return prompts


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


def run_fastllm(args, result_dir, prompts):
    env = os.environ.copy()
    env.pop("FASTLLM_CUDA_NVFP4_TRACE", None)
    env["FASTLLM_CUDA_NVFP4_W4A4"] = "1"
    env["FASTLLM_CUDA_NVFP4_W4A4_STRICT"] = "1"
    env["FASTLLM_CUDA_MOE_NVFP4_W4A4"] = "1"
    env["FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT"] = "1"
    prefill_json = result_dir / "fastllm-prefill.json"
    common = [args.model, "--dtype", args.flm_dtype, "--atype", args.flm_atype,
              "--device", args.flm_device, "--moe_device", args.flm_device]
    prefill_command = [
        args.fastllm_python, "test/benchmark/prefill.py", *common,
        "--prompt-file", prompts["prefill_path"], "--completion-api",
        "--prompt-repeat", str(args.prefill_repeat),
        "--max-tokens", str(args.prefill_max_tokens),
        "--result-json", str(prefill_json),
    ]
    run_logged(prefill_command, result_dir / "fastllm-prefill.log", env)
    decode_results = []
    for batch in decode_batch_cases(args):
        decode_json = result_dir / f"fastllm-decode-b{batch}.json"
        decode_command = [
            args.fastllm_python, "test/benchmark/decode.py", *common,
            "--batch-size", str(batch), "--max_batch", str(batch),
            "--prompt-file", prompts["decode_path"], "--completion-api",
            "--prefill-length", str(args.decode_prefill_length),
            "--max-tokens", str(args.decode_max_tokens),
            "--result-json", str(decode_json),
        ]
        run_logged(decode_command, result_dir / f"fastllm-decode-b{batch}.log", env)
        decode_results.append(read_json(decode_json))
    (result_dir / "fastllm-decode.json").write_text(
        json.dumps(decode_results, ensure_ascii=False, indent=2), encoding="utf-8")
    return read_json(prefill_json), decode_results


def run_vllm(args, result_dir, prompts):
    base_url = f"http://127.0.0.1:{args.port}"
    served_name = "nvfp4-performance"
    batches = decode_batch_cases(args)
    command = [
        args.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model, "--served-model-name", served_name,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--dtype", "auto", "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(max(batches)),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code", "--enforce-eager",
        "--linear-backend", "cutlass",
        "--no-enable-prefix-caching",
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
        warmup = PREFILL.warmup(
            base_url, served_name, "no-key", 3600, 8,
            prompt_text=prompts["decode"], completion_api=True)
        print("vLLM warmup: " + json.dumps(warmup, ensure_ascii=False), flush=True)

        prefill = PREFILL.run_chat_completion(
            base_url, served_name, "no-key", prompts["prefill"],
            args.prefill_max_tokens, 3600, completion_api=True)
        prefill.update({
            "case_name": "single", "backend": "vllm", "model_path": args.model,
            "prompt_repeat": args.prefill_repeat,
            "prompt_chars": len(prompts["prefill"]),
            "max_tokens": args.prefill_max_tokens,
        })

        decode_results = []
        for batch in batches:
            decode = DECODE.run_decode_batch(
                base_url, served_name, "no-key", prompts["decode"],
                args.decode_max_tokens, 3600, batch, 0, completion_api=True)
            decode.update({
                "case_name": f"batch-{batch}", "backend": "vllm",
                "model_path": args.model,
                "prefill_length_chars": args.decode_prefill_length,
                "prompt_chars": len(prompts["decode"]), "batch_size": batch,
                "server_max_batch": max(batches),
                "max_tokens": args.decode_max_tokens,
            })
            decode_results.append(decode)
            (result_dir / f"vllm-decode-b{batch}.json").write_text(
                json.dumps(decode, ensure_ascii=False, indent=2), encoding="utf-8")
        (result_dir / "vllm-prefill.json").write_text(
            json.dumps(prefill, ensure_ascii=False, indent=2), encoding="utf-8")
        (result_dir / "vllm-decode.json").write_text(
            json.dumps(decode_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print("vLLM prefill result: " + json.dumps(prefill, ensure_ascii=False), flush=True)
        print("vLLM decode results: " + json.dumps(decode_results, ensure_ascii=False), flush=True)
        return prefill, decode_results
    finally:
        stop_process(process)
        log_handle.close()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def validate_prompt_tokens(ft_prefill, ft_decode, vllm_prefill, vllm_decode):
    """两个后端必须报告完全相同的prompt token数，否则拒绝生成性能结论。"""
    if ft_prefill["prompt_tokens"] != vllm_prefill["prompt_tokens"]:
        raise RuntimeError(
            "Prefill prompt token不一致: "
            f"FastLLM={ft_prefill['prompt_tokens']}, "
            f"vLLM={vllm_prefill['prompt_tokens']}")
    ft_by_batch = {item["batch_size"]: item for item in ft_decode}
    vllm_by_batch = {item["batch_size"]: item for item in vllm_decode}
    if set(ft_by_batch) != set(vllm_by_batch):
        raise RuntimeError("FastLLM与vLLM的decode batch集合不一致")
    for batch in sorted(ft_by_batch):
        ft_tokens = ft_by_batch[batch]["total_prompt_tokens"]
        vllm_tokens = vllm_by_batch[batch]["total_prompt_tokens"]
        if ft_tokens != vllm_tokens:
            raise RuntimeError(
                f"Decode batch={batch} prompt token不一致: "
                f"FastLLM={ft_tokens}, vLLM={vllm_tokens}")
    print("Prompt token check: PASS", flush=True)


def args_value(items, key):
    return items[0].get(key, "-") if items else "-"


def make_report(result_dir, ft_prefill, ft_decode, vllm_prefill, vllm_decode):
    prefill_rows = [
        ("FastLLM", ft_prefill), ("vLLM", vllm_prefill),
    ]
    decode_rows = [("FastLLM", item) for item in ft_decode]
    decode_rows.extend(("vLLM", item) for item in vllm_decode)
    ft_decode_by_batch = {item["batch_size"]: item for item in ft_decode}
    vllm_decode_by_batch = {item["batch_size"]: item for item in vllm_decode}
    lines = [
        "# NVFP4整体性能对比", "",
        "## 测试参数", "",
        "| 场景 | Prompt参数 | Batch | 最大生成tokens |",
        "| --- | ---: | ---: | ---: |",
        f"| Prefill | repeat={ft_prefill.get('prompt_repeat', '-')} | 1 | {ft_prefill['max_tokens']} |",
        f"| Decode | chars={args_value(ft_decode, 'prefill_length_chars')} | "
        f"{','.join(str(item['batch_size']) for item in ft_decode)} | "
        f"{args_value(ft_decode, 'max_tokens')} |",
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
        "| 指标 | Batch | 比值 |",
        "| --- | ---: | ---: |",
        f"| Prefill吞吐 | 1 | {ratio(ft_prefill['prefill_speed'], vllm_prefill['prefill_speed']):.4f}x |",
        f"| TTFT速度（vLLM TTFT / FastLLM TTFT） | 1 | {ratio(vllm_prefill['ttft'], ft_prefill['ttft']):.4f}x |",
    ])
    for batch in sorted(set(ft_decode_by_batch) & set(vllm_decode_by_batch)):
        ft_item = ft_decode_by_batch[batch]
        vllm_item = vllm_decode_by_batch[batch]
        lines.append(
            f"| Decode batch吞吐 | {batch} | "
            f"{ratio(ft_item['batch_decode_speed'], vllm_item['batch_decode_speed']):.4f}x |")
        lines.append(
            f"| Decode E2E吞吐 | {batch} | "
            f"{ratio(ft_item['end_to_end_speed'], vllm_item['end_to_end_speed']):.4f}x |")
    lines.append("")
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
    prompts = render_shared_prompts(args, result_dir)
    ft_prefill, ft_decode = run_fastllm(args, result_dir, prompts)
    vllm_prefill, vllm_decode = run_vllm(args, result_dir, prompts)
    validate_prompt_tokens(ft_prefill, ft_decode, vllm_prefill, vllm_decode)
    make_report(result_dir, ft_prefill, ft_decode, vllm_prefill, vllm_decode)


if __name__ == "__main__":
    main()
