#!/usr/bin/env python3
"""使用同一组token ID分时测试FastLLM和vLLM整体性能。"""

import argparse
import csv
import ctypes
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

import requests


REPO_DIR = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="NVFP4 vLLM/FastLLM严格整体性能对比")
    parser.add_argument("--model", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--vllm-python", default=os.environ.get("NVFP4_VLLM_PYTHON", sys.executable))
    parser.add_argument("--fastllm-python", default=sys.executable)
    parser.add_argument("--flm-dtype", default="auto")
    parser.add_argument("--flm-atype", default="bfloat16")
    parser.add_argument("--flm-device", default="cuda")
    parser.add_argument("--prefill-input-tokens", type=int, default=4096)
    parser.add_argument("--prefill-max-tokens", type=int, default=16)
    parser.add_argument("--decode-input-tokens", type=int, default=512)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument(
        "--decode-batch-sizes", default="",
        help="逗号分隔的decode batch矩阵；为空时使用--decode-batch-size")
    parser.add_argument("--decode-max-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--stage", choices=("orchestrate", "fastllm"),
                        default="orchestrate", help=argparse.SUPPRESS)
    parser.add_argument("--prompt-token-file", default="", help=argparse.SUPPRESS)
    parser.add_argument("--output", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def require_positive(name, value):
    if value <= 0:
        raise ValueError(f"{name}必须大于0")


def decode_batch_cases(args):
    values = ([args.decode_batch_size] if not args.decode_batch_sizes else
              [int(value.strip()) for value in args.decode_batch_sizes.split(",")
               if value.strip()])
    if not values or any(value <= 0 for value in values):
        raise ValueError("decode batch必须是正整数")
    return list(dict.fromkeys(values))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_logged(command, log_path, env=None):
    print(f"SUBPROCESS COMMAND: {shlex.join(command)}", flush=True)
    with Path(log_path).open("w", encoding="utf-8") as log:
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


def render_prompt_tokens(tokenizer, target_tokens, label):
    """生成完整Chat Prompt，并用有效上下文token补到指定长度。"""
    messages = [{
        "role": "user",
        "content": f"这是{label}性能测试。请基于前面的上下文继续回答。",
    }]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        prompt = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    base_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(base_ids) > target_tokens:
        raise ValueError(
            f"{label}目标长度{target_tokens}小于基础Chat Prompt长度{len(base_ids)}")
    filler = tokenizer.encode(
        " FastLLM vLLM shared benchmark context block.",
        add_special_tokens=False)
    if not filler:
        raise RuntimeError("上下文填充文本未产生token")
    missing = target_tokens - len(base_ids)
    fill_ids = (filler * ((missing + len(filler) - 1) // len(filler)))[:missing]
    # 保留Chat Template的结尾，在assistant生成前缀之前插入上下文token。
    token_ids = base_ids[:-1] + fill_ids + base_ids[-1:] if base_ids else fill_ids
    if len(token_ids) != target_tokens:
        raise AssertionError("固定Prompt token长度构造失败")
    return token_ids, prompt


def build_shared_prompts(args, result_dir):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prefill_ids, prefill_prompt = render_prompt_tokens(
        tokenizer, args.prefill_input_tokens, "Prefill")
    decode_ids, decode_prompt = render_prompt_tokens(
        tokenizer, args.decode_input_tokens, "Decode")
    payload = {
        "model": args.model,
        "tokenizer": type(tokenizer).__name__,
        "prefill": {
            "token_ids": prefill_ids,
            "prompt_preview": prefill_prompt[:200],
        },
        "decode": {
            "token_ids": decode_ids,
            "prompt_preview": decode_prompt[:200],
        },
    }
    path = result_dir / "shared-prompt-token-ids.json"
    write_json(path, payload)
    return path


def mean_or_none(values):
    return statistics.mean(values) if values else None


def summarize_requests(backend, workload, batch, input_tokens, target_output_tokens,
                       requests_data, batch_start, batch_end):
    ttfts = []
    tpots = []
    itls = []
    e2els = []
    for item in requests_data:
        token_times = item["token_times"]
        e2els.append(item["end_time"] - item["start_time"])
        if token_times:
            ttfts.append(token_times[0] - item["start_time"])
        if len(token_times) > 1:
            tpots.append((item["end_time"] - token_times[0]) / (len(token_times) - 1))
            itls.extend(
                token_times[index] - token_times[index - 1]
                for index in range(1, len(token_times)))
    total_output_tokens = sum(item["output_tokens"] for item in requests_data)
    wall = batch_end - batch_start
    ttft_avg = mean_or_none(ttfts)
    return {
        "workload": workload,
        "backend": backend,
        "batch": batch,
        "prompt_tokens_per_request": len(input_tokens),
        "total_prompt_tokens": len(input_tokens) * batch,
        "target_output_tokens_per_request": target_output_tokens,
        "total_output_tokens": total_output_tokens,
        "ttft_s": ttft_avg,
        "tpot_s": mean_or_none(tpots),
        "itl_s": mean_or_none(itls),
        "e2el_s": mean_or_none(e2els),
        "wall_s": wall,
        "prefill_tok_s": (
            len(input_tokens) / ttft_avg
            if batch == 1 and ttft_avg and ttft_avg > 0 else None),
        "output_tok_s": total_output_tokens / wall if wall > 0 else 0.0,
        "requests": requests_data,
    }


def launch_fastllm_request(model, input_tokens, output_tokens):
    from ftllm import llm

    token_buffer = (ctypes.c_int * len(input_tokens))(*input_tokens)
    empty_stops = (ctypes.c_int * 0)()
    return llm.fastllm_lib.launch_response_llm_model(
        model.model, len(input_tokens), token_buffer,
        ctypes.c_int(output_tokens), ctypes.c_int(0), ctypes.c_bool(False),
        ctypes.c_float(1.0), ctypes.c_int(1), ctypes.c_float(1.0),
        ctypes.c_float(1.0), ctypes.c_bool(False), ctypes.c_int(0), empty_stops)


def run_fastllm_batch(model, workload, input_tokens, output_tokens, batch):
    from ftllm import llm

    requests_data = []
    batch_start = time.perf_counter()
    for request_id in range(batch):
        start_time = time.perf_counter()
        handle = launch_fastllm_request(model, input_tokens, output_tokens)
        requests_data.append({
            "request_id": request_id,
            "handle": handle,
            "start_time": start_time,
            "end_time": None,
            "token_times": [],
            "output_tokens": 0,
            "finish_code": None,
        })
    pending = set(range(batch))
    while pending:
        progressed = False
        for index in list(pending):
            item = requests_data[index]
            if not llm.fastllm_lib.can_fetch_response_llm_model(model.model, item["handle"]):
                continue
            token = llm.fastllm_lib.fetch_response_llm_model(model.model, item["handle"])
            now = time.perf_counter()
            progressed = True
            if token <= -1:
                item["end_time"] = now
                item["finish_code"] = token
                pending.remove(index)
            else:
                item["token_times"].append(now)
                item["output_tokens"] += 1
        if not progressed:
            time.sleep(0.0005)
    batch_end = max(item["end_time"] for item in requests_data)
    for item in requests_data:
        item.pop("handle")
    return summarize_requests(
        "FastLLM", workload, batch, input_tokens, output_tokens,
        requests_data, batch_start, batch_end)


def run_fastllm_stage(args):
    from ftllm import llm

    prompts = read_json(args.prompt_token_file)
    batches = decode_batch_cases(args)
    llm.set_device_map(args.flm_device)
    llm.set_device_map(args.flm_device, True)
    model = llm.model(args.model, dtype=args.flm_dtype)
    try:
        model.set_atype(args.flm_atype)
        model.set_max_batch(max(batches))
        model.warmup()
        cases = [
            ("prefill", prompts["prefill"]["token_ids"], args.prefill_max_tokens, 1),
            *[("decode", prompts["decode"]["token_ids"],
               args.decode_max_tokens, batch) for batch in batches],
        ]
        results = []
        for workload, token_ids, output_tokens, batch in cases:
            for _ in range(args.warmup):
                run_fastllm_batch(
                    model, "warmup", token_ids, min(output_tokens, 8), batch)
            result = run_fastllm_batch(
                model, workload, token_ids, output_tokens, batch)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            results.append(result)
        write_json(args.output, results)
    finally:
        model.release_memory()


def wait_server(base_url, process, timeout, server_log):
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            tail = Path(server_log).read_text(
                encoding="utf-8", errors="replace")[-8000:]
            raise RuntimeError(
                f"vLLM服务提前退出，code={process.returncode}\n{tail}")
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


def run_vllm_request(base_url, model_name, input_tokens, output_tokens,
                     request_timeout, request_id):
    payload = {
        "model": model_name,
        "prompt": input_tokens,
        "add_special_tokens": False,
        "max_tokens": output_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    start_time = time.perf_counter()
    response = requests.post(
        f"{base_url}/v1/completions", json=payload, stream=True,
        timeout=request_timeout)
    response.raise_for_status()
    token_times = []
    returned_prompt_ids = None
    usage = {}
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data = raw_line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        usage = chunk.get("usage") or usage
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if choice.get("prompt_token_ids") is not None:
            returned_prompt_ids = choice["prompt_token_ids"]
        delta_ids = choice.get("token_ids") or []
        now = time.perf_counter()
        token_times.extend([now] * len(delta_ids))
    end_time = time.perf_counter()
    if returned_prompt_ids != input_tokens:
        raise RuntimeError(
            f"vLLM请求{request_id}的prompt token IDs与共享输入不一致")
    output_count = int(usage.get("completion_tokens") or len(token_times))
    if output_count != len(token_times):
        raise RuntimeError(
            f"vLLM请求{request_id}流式token数{len(token_times)}与usage中的"
            f"{output_count}不一致")
    return {
        "request_id": request_id,
        "start_time": start_time,
        "end_time": end_time,
        "token_times": token_times,
        "output_tokens": output_count,
        "finish_code": -1,
    }


def run_vllm_batch(base_url, model_name, workload, input_tokens, output_tokens,
                   batch, request_timeout):
    batch_start = time.perf_counter()
    requests_data = []
    with ThreadPoolExecutor(max_workers=batch) as executor:
        futures = [
            executor.submit(
                run_vllm_request, base_url, model_name, input_tokens,
                output_tokens, request_timeout, request_id)
            for request_id in range(batch)
        ]
        for future in as_completed(futures):
            requests_data.append(future.result())
    requests_data.sort(key=lambda item: item["request_id"])
    batch_end = max(item["end_time"] for item in requests_data)
    return summarize_requests(
        "vLLM", workload, batch, input_tokens, output_tokens,
        requests_data, batch_start, batch_end)


def run_vllm(args, prompt_path, result_dir):
    prompts = read_json(prompt_path)
    batches = decode_batch_cases(args)
    base_url = f"http://127.0.0.1:{args.port}"
    served_name = "nvfp4-performance"
    command = [
        args.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model, "--served-model-name", served_name,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--dtype", "auto", "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(max(batches)),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code", "--enforce-eager",
        "--linear-backend", "cutlass", "--no-enable-prefix-caching",
        "--enable-force-include-usage",
    ]
    env = os.environ.copy()
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
        cases = [
            ("prefill", prompts["prefill"]["token_ids"], args.prefill_max_tokens, 1),
            *[("decode", prompts["decode"]["token_ids"],
               args.decode_max_tokens, batch) for batch in batches],
        ]
        results = []
        for workload, token_ids, output_tokens, batch in cases:
            for _ in range(args.warmup):
                run_vllm_batch(
                    base_url, served_name, "warmup", token_ids,
                    min(output_tokens, 8), batch, args.request_timeout)
            result = run_vllm_batch(
                base_url, served_name, workload, token_ids,
                output_tokens, batch, args.request_timeout)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            results.append(result)
        write_json(result_dir / "vllm-results.json", results)
        return results
    finally:
        stop_process(process)
        log_handle.close()


def fmt_ms(value):
    return "n/a" if value is None else f"{value * 1000:.3f}"


def fmt_number(value):
    return "n/a" if value is None else f"{value:.2f}"


def ratio(fastllm_value, vllm_value):
    if fastllm_value is None or vllm_value in (None, 0):
        return "n/a"
    return f"{fastllm_value / vllm_value:.4f}x"


def ordered_results(fastllm_results, vllm_results):
    fastllm_map = {
        (item["workload"], item["batch"]): item for item in fastllm_results}
    vllm_map = {(item["workload"], item["batch"]): item for item in vllm_results}
    keys = [("prefill", 1)] + [
        ("decode", item["batch"])
        for item in fastllm_results if item["workload"] == "decode"]
    rows = []
    for key in keys:
        if key not in fastllm_map or key not in vllm_map:
            raise RuntimeError(f"FastLLM/vLLM缺少对应测试项: {key}")
        rows.extend((fastllm_map[key], vllm_map[key]))
    return rows


def make_report(result_dir, fastllm_results, vllm_results):
    rows = ordered_results(fastllm_results, vllm_results)
    lines = [
        "# NVFP4严格整体性能对比", "",
        "> 两个后端直接接收同一组token ID；不经过各自Chat Template。", "",
        "| Workload | 后端 | Batch | Prompt/请求 | Prompt总数 | Output总数 | "
        "TTFT(ms) | TPOT(ms) | ITL(ms) | E2EL(ms) | Prefill(tok/s) | "
        "Output(tok/s) | Wall(s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]
    csv_rows = []
    for item in rows:
        lines.append(
            f"| {item['workload']} | {item['backend']} | {item['batch']} | "
            f"{item['prompt_tokens_per_request']} | {item['total_prompt_tokens']} | "
            f"{item['total_output_tokens']} | {fmt_ms(item['ttft_s'])} | "
            f"{fmt_ms(item['tpot_s'])} | {fmt_ms(item['itl_s'])} | "
            f"{fmt_ms(item['e2el_s'])} | {fmt_number(item['prefill_tok_s'])} | "
            f"{fmt_number(item['output_tok_s'])} | {item['wall_s']:.6f} |")
        csv_rows.append({
            "workload": item["workload"],
            "backend": item["backend"],
            "batch": item["batch"],
            "prompt_tokens_per_request": item["prompt_tokens_per_request"],
            "total_prompt_tokens": item["total_prompt_tokens"],
            "output_tokens": item["total_output_tokens"],
            "ttft_ms": None if item["ttft_s"] is None else item["ttft_s"] * 1000,
            "tpot_ms": None if item["tpot_s"] is None else item["tpot_s"] * 1000,
            "itl_ms": None if item["itl_s"] is None else item["itl_s"] * 1000,
            "e2el_ms": None if item["e2el_s"] is None else item["e2el_s"] * 1000,
            "prefill_tok_s": item["prefill_tok_s"],
            "output_tok_s": item["output_tok_s"],
            "wall_s": item["wall_s"],
        })
    lines.extend(["", "## FastLLM / vLLM", "",
                  "| Workload | Batch | TTFT | TPOT | ITL | E2EL | Output吞吐 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for index in range(0, len(rows), 2):
        fastllm_item, vllm_item = rows[index:index + 2]
        lines.append(
            f"| {fastllm_item['workload']} | {fastllm_item['batch']} | "
            f"{ratio(fastllm_item['ttft_s'], vllm_item['ttft_s'])} | "
            f"{ratio(fastllm_item['tpot_s'], vllm_item['tpot_s'])} | "
            f"{ratio(fastllm_item['itl_s'], vllm_item['itl_s'])} | "
            f"{ratio(fastllm_item['e2el_s'], vllm_item['e2el_s'])} | "
            f"{ratio(fastllm_item['output_tok_s'], vllm_item['output_tok_s'])} |")
    lines.append("")
    report = "\n".join(lines)
    markdown_path = result_dir / "model-performance-compare.md"
    csv_path = result_dir / "model-performance-compare.csv"
    markdown_path.write_text(report, encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_json(result_dir / "model-performance-compare.json", {
        "fastllm": fastllm_results,
        "vllm": vllm_results,
    })
    print(report)
    print(f"Markdown: {markdown_path}")
    print(f"CSV: {csv_path}")


def orchestrate(args):
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = build_shared_prompts(args, result_dir)
    fastllm_output = result_dir / "fastllm-results.json"
    command = [
        args.fastllm_python, str(Path(__file__).resolve()),
        "--stage", "fastllm", "--model", args.model,
        "--result-dir", str(result_dir),
        "--prompt-token-file", str(prompt_path),
        "--output", str(fastllm_output),
        "--flm-dtype", args.flm_dtype, "--flm-atype", args.flm_atype,
        "--flm-device", args.flm_device,
        "--prefill-input-tokens", str(args.prefill_input_tokens),
        "--prefill-max-tokens", str(args.prefill_max_tokens),
        "--decode-input-tokens", str(args.decode_input_tokens),
        "--decode-batch-sizes", ",".join(map(str, decode_batch_cases(args))),
        "--decode-max-tokens", str(args.decode_max_tokens),
        "--warmup", str(args.warmup),
    ]
    env = os.environ.copy()
    env.pop("FASTLLM_CUDA_NVFP4_TRACE", None)
    env["FASTLLM_CUDA_NVFP4_W4A4"] = "1"
    env["FASTLLM_CUDA_NVFP4_W4A4_STRICT"] = "1"
    env["FASTLLM_CUDA_MOE_NVFP4_W4A4"] = "1"
    env["FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT"] = "1"
    run_logged(command, result_dir / "fastllm.log", env)
    fastllm_results = read_json(fastllm_output)
    vllm_results = run_vllm(args, prompt_path, result_dir)
    make_report(result_dir, fastllm_results, vllm_results)


def main():
    args = parse_args()
    require_positive("prefill-input-tokens", args.prefill_input_tokens)
    require_positive("prefill-max-tokens", args.prefill_max_tokens)
    require_positive("decode-input-tokens", args.decode_input_tokens)
    require_positive("decode-max-tokens", args.decode_max_tokens)
    if args.warmup < 0:
        raise ValueError("warmup不能小于0")
    if args.stage == "fastllm":
        if not args.prompt_token_file or not args.output:
            raise ValueError("FastLLM阶段缺少prompt-token-file或output")
        run_fastllm_stage(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
