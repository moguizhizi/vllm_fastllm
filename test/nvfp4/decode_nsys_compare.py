#!/usr/bin/env python3
"""按Decode Batch采集并对比FastLLM与vLLM的Nsight Systems报告。"""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import requests


REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "test" / "nvfp4"))

import model_performance_compare as model_compare  # noqa: E402


BACKENDS = ("fastllm", "vllm")
KERNEL_CATEGORIES = (
    ("GEMM/Linear", ("gemm", "cutlass", "scaled_mm", "marlin", "matmul")),
    ("Attention", ("attention", "flash", "fmha", "fused_attn", "mla")),
    ("Sampling", ("sampling", "sample", "topk", "top_k", "topp", "top_p")),
    ("Quantize", ("quant", "fp8", "nvfp4", "float4")),
    ("Norm", ("rmsnorm", "rms_norm", "layernorm", "layer_norm")),
    ("KV Cache", ("kv_cache", "paged", "reshape_and_cache", "cache_kernel")),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="按Batch对比FastLLM/vLLM稳定Decode阶段的Nsight Systems数据")
    parser.add_argument("--model", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--quantization", choices=("nvfp4", "w8a8"),
                        default="w8a8")
    parser.add_argument("--backends", default="fastllm,vllm",
                        help="逗号分隔：fastllm,vllm")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--vllm-python", default=os.environ.get(
        "NVFP4_VLLM_PYTHON", sys.executable))
    parser.add_argument("--fastllm-python", default=sys.executable)
    parser.add_argument("--flm-dtype", default="auto")
    parser.add_argument("--flm-atype", default="bfloat16")
    parser.add_argument("--flm-device", default="cuda")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--top-kernels", type=int, default=20)
    parser.add_argument(
        "--vllm-extra-arg", action="append", default=[],
        help="追加一个vLLM服务参数；需要多个参数时重复使用")
    return parser.parse_args()


def parse_positive_csv(value, label):
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{label}必须是逗号分隔的正整数")
    return list(dict.fromkeys(values))


def selected_backends(value):
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not values or any(item not in BACKENDS for item in values):
        raise ValueError("backends只能包含fastllm和vllm")
    return list(dict.fromkeys(values))


def run_command(command, *, env=None, check=True, capture=True):
    print(f"COMMAND: {shlex.join([str(item) for item in command])}", flush=True)
    return subprocess.run(
        [str(item) for item in command], cwd=REPO_DIR, env=env, check=check,
        text=True, stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None)


def nsys_control_env():
    # FastLLM Python可能需要预加载系统libstdc++，但该库不能注入Nsight自身的
    # importer，否则部分Nsight版本只能留下qdstrm而无法生成nsys-rep。
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    return env


def target_command(command):
    preload = os.environ.get("LD_PRELOAD", "")
    if not preload:
        return command
    return ["/usr/bin/env", f"LD_PRELOAD={preload}", *command]


def make_prompt_tokens(tokenizer, length, batch):
    return model_compare.render_prompt_tokens(
        tokenizer, length, f"decode nsys batch {batch}")


def wait_server(base_url, launcher, timeout, log_path, backend):
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        if launcher.poll() is not None:
            tail = log_path.read_text(
                encoding="utf-8", errors="replace")[-12000:]
            raise RuntimeError(
                f"{backend} Nsight服务提前退出，code={launcher.returncode}\n{tail}")
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=2)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as error:
            last_error = str(error)
        time.sleep(1)
    raise TimeoutError(f"等待{backend}服务启动超时：{last_error}")


def server_spec(args, backend, max_batch):
    if backend == "fastllm":
        command = [
            args.fastllm_python,
            str(REPO_DIR / "test/nvfp4/fastllm_http_benchmark_server.py"),
            "--model", args.model, "--host", "127.0.0.1",
            "--port", str(args.port), "--dtype", args.flm_dtype,
            "--atype", args.flm_atype, "--device", args.flm_device,
            "--max-batch", str(max_batch),
        ]
        env = os.environ.copy()
        env["FASTLLM_CUDA_GRAPH"] = "1"
        env.pop("FASTLLM_CUDA_GRAPH_TRACE", None)
        env.pop("FASTLLM_CUDA_NVFP4_TRACE", None)
        if args.quantization == "w8a8":
            env["FASTLLM_CUDA_W8A8"] = "1"
            env["FASTLLM_CUDA_W8A8_STRICT"] = "1"
        else:
            env["FASTLLM_CUDA_NVFP4_W4A4"] = "1"
            env["FASTLLM_CUDA_NVFP4_W4A4_STRICT"] = "1"
            env["FASTLLM_CUDA_MOE_NVFP4_W4A4"] = "1"
            env["FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT"] = "1"
        return command, env, "nvfp4-performance"

    command = [
        args.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model, "--served-model-name", "decode-nsys",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--dtype", "auto", "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(max_batch), "--gpu-memory-utilization",
        str(args.gpu_memory_utilization), "--trust-remote-code",
        "--linear-backend", "cutlass", "--enable-prefix-caching",
        "--enable-force-include-usage",
    ]
    if args.quantization == "nvfp4":
        command.extend(("--moe-backend", "cutlass"))
    command.extend(args.vllm_extra_arg)
    env = os.environ.copy()
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_USE_FLASHINFER_MOE_FP4"] = "0"
    return command, env, "decode-nsys"


def launch_profiled_server(args, backend, max_batch, backend_dir):
    command, target_env, served_name = server_spec(args, backend, max_batch)
    session = re.sub(
        r"[^A-Za-z0-9_-]", "_",
        f"decode_{backend}_{os.getpid()}_{int(time.time())}")
    launch = [
        args.nsys, "launch", f"--session-new={session}",
        "--trace=cuda,nvtx", "--cuda-graph-trace=node",
        *target_command(command),
    ]
    log_path = backend_dir / "server.log"
    log_handle = log_path.open("w", encoding="utf-8")
    log_handle.write(f"COMMAND: {shlex.join(launch)}\n\n")
    log_handle.flush()
    control_env = nsys_control_env()
    # 除LD_PRELOAD外，目标服务与Nsight控制进程共享显式测试环境。
    control_env.update({key: value for key, value in target_env.items()
                        if key != "LD_PRELOAD"})
    launcher = subprocess.Popen(
        launch, cwd=REPO_DIR, env=control_env, stdout=log_handle,
        stderr=subprocess.STDOUT, text=True)
    base_url = f"http://127.0.0.1:{args.port}"
    try:
        wait_server(base_url, launcher, args.startup_timeout, log_path, backend)
    except Exception:
        log_handle.close()
        shutdown_session(args, session, launcher)
        raise
    return {
        "backend": backend,
        "session": session,
        "launcher": launcher,
        "log_handle": log_handle,
        "log_path": log_path,
        "base_url": base_url,
        "served_name": served_name,
    }


def shutdown_session(args, session, launcher):
    run_command(
        [args.nsys, "shutdown", f"--session={session}", "--kill=sigterm"],
        env=nsys_control_env(), check=False)
    if launcher.poll() is None:
        try:
            launcher.wait(timeout=60)
        except subprocess.TimeoutExpired:
            launcher.terminate()
            try:
                launcher.wait(timeout=10)
            except subprocess.TimeoutExpired:
                launcher.kill()


def resolve_report(output_base):
    direct = output_base.with_suffix(".nsys-rep")
    if direct.exists():
        return direct
    matches = sorted(output_base.parent.glob(output_base.name + "*.nsys-rep"))
    if matches:
        return matches[-1]
    qdstrm = output_base.with_suffix(".qdstrm")
    detail = f"；仅生成了{qdstrm}" if qdstrm.exists() else ""
    raise RuntimeError(
        "Nsight未生成nsys-rep" + detail +
        "。请确认Nsight importer可运行，并确保LD_PRELOAD未注入nsys进程。")


def start_collection(args, session, output_base):
    result = run_command([
        args.nsys, "start", f"--session={session}", "--sample=none",
        "--cpuctxsw=none", f"--output={output_base}",
        "--force-overwrite=true",
    ], env=nsys_control_env(), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"nsys start失败：\n{result.stdout}")


def stop_collection(args, session, output_base):
    result = run_command(
        [args.nsys, "stop", f"--session={session}"],
        env=nsys_control_env(), check=False)
    if result.stdout:
        (output_base.parent / f"{output_base.name}-stop.log").write_text(
            result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"nsys stop失败：\n{result.stdout}")
    return resolve_report(output_base)


def run_decode_batch(server, prompt, output_tokens, batch, args, workload):
    return model_compare.run_http_batch(
        server["base_url"], server["served_name"], server["backend"],
        "best", "cache_hit", workload, prompt, output_tokens, batch,
        args.request_timeout)


def export_stats(args, report, report_name, output_base):
    command = [
        args.nsys, "stats", "--force-export=true", "--force-overwrite=true",
        "--format=csv", f"--report={report_name}",
        f"--output={output_base}", str(report),
    ]
    result = run_command(command, env=nsys_control_env(), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"生成Nsight统计{report_name}失败：\n{result.stdout}")
    candidates = sorted(output_base.parent.glob(
        output_base.name + "*" + report_name.split(":", 1)[0] + "*.csv"))
    if not candidates:
        candidates = sorted(output_base.parent.glob(output_base.name + "*.csv"))
    if not candidates:
        raise RuntimeError(f"Nsight没有输出{report_name} CSV")
    return candidates[-1]


def numeric(row, names):
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        value = value.replace(",", "").strip()
        try:
            return float(value)
        except ValueError:
            continue
    return 0.0


def text_value(row, names):
    for name in names:
        value = row.get(name)
        if value:
            return value.strip()
    return "unknown"


def read_stats_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_named_rows(rows):
    result = []
    for row in rows:
        total_ns = numeric(row, ("Total Time (ns)", "Total Time"))
        instances = int(numeric(row, ("Instances", "Count", "Calls")))
        average_ns = numeric(row, ("Avg (ns)", "Average (ns)", "Avg"))
        if not average_ns and instances:
            average_ns = total_ns / instances
        result.append({
            "name": text_value(row, ("Name", "Operation", "API Name")),
            "total_ms": total_ns / 1e6,
            "instances": instances,
            "average_us": average_ns / 1e3,
        })
    result.sort(key=lambda item: item["total_ms"], reverse=True)
    return result


def kernel_category(name):
    lowered = name.lower()
    for category, patterns in KERNEL_CATEGORIES:
        if any(pattern in lowered for pattern in patterns):
            return category
    return "Other"


def summarize_report(args, report, output_base):
    kernel_csv = export_stats(
        args, report, "cuda_gpu_kern_sum", output_base.with_name(
            output_base.name + "-kernels"))
    api_csv = export_stats(
        args, report, "cuda_api_sum", output_base.with_name(
            output_base.name + "-cuda-api"))
    memory_csv = export_stats(
        args, report, "cuda_gpu_mem_time_sum", output_base.with_name(
            output_base.name + "-memory"))
    kernels = summarize_named_rows(read_stats_csv(kernel_csv))
    apis = summarize_named_rows(read_stats_csv(api_csv))
    memory = summarize_named_rows(read_stats_csv(memory_csv))
    categories = {}
    for item in kernels:
        category = kernel_category(item["name"])
        entry = categories.setdefault(
            category, {"total_ms": 0.0, "instances": 0})
        entry["total_ms"] += item["total_ms"]
        entry["instances"] += item["instances"]
    return {
        "report": str(report),
        "kernel_csv": str(kernel_csv),
        "cuda_api_csv": str(api_csv),
        "memory_csv": str(memory_csv),
        "kernel_total_ms": sum(item["total_ms"] for item in kernels),
        "kernel_instances": sum(item["instances"] for item in kernels),
        "memory_total_ms": sum(item["total_ms"] for item in memory),
        "categories": categories,
        "top_kernels": kernels[:args.top_kernels],
        "top_cuda_apis": apis[:10],
        "memory_operations": memory,
    }


def profile_backend(args, backend, batches, prompts, result_dir):
    backend_dir = result_dir / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    server = launch_profiled_server(args, backend, max(batches), backend_dir)
    rows = []
    try:
        for batch in batches:
            case_dir = backend_dir / f"batch-{batch}"
            case_dir.mkdir(parents=True, exist_ok=True)
            prompt = prompts[batch]
            # 使用同一批请求完成CUDA Graph/shape warmup并建立Prefix Cache；采集
            # 窗口中的请求因此以稳定Decode为主，而不是模型加载或Cold Prefill。
            run_decode_batch(
                server, prompt, args.warmup_output_tokens, batch, args,
                "warmup")
            output_base = case_dir / "decode"
            start_collection(args, server["session"], output_base)
            request_result = run_decode_batch(
                server, prompt, args.output_tokens, batch, args,
                "decode")
            report = stop_collection(args, server["session"], output_base)
            summary = summarize_report(args, report, output_base)
            row = {
                "backend": backend,
                "batch": batch,
                "prompt_tokens": len(prompt),
                "output_tokens_per_request": args.output_tokens,
                "request_wall_ms": request_result["wall_s"] * 1000,
                "tpot_ms": (None if request_result["tpot_s"] is None else
                            request_result["tpot_s"] * 1000),
                "itl_ms": (None if request_result["itl_s"] is None else
                           request_result["itl_s"] * 1000),
                "cache_hit_ratio": request_result["cache_hit_ratio"],
                "nsys": summary,
            }
            (case_dir / "result.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(row)
            print(json.dumps({
                "backend": backend, "batch": batch,
                "tpot_ms": row["tpot_ms"],
                "kernel_total_ms": summary["kernel_total_ms"],
                "kernel_instances": summary["kernel_instances"],
            }, ensure_ascii=False), flush=True)
    finally:
        shutdown_session(args, server["session"], server["launcher"])
        server["log_handle"].close()
    return rows


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def make_report(result_dir, rows, top_count):
    lines = [
        "# FastLLM / vLLM Decode Nsight Systems对比", "",
        "> 每个Batch先使用同一Prompt完成warmup及Prefix Cache填充，再采集一次",
        "> BEST模式HTTP请求。报告包含请求调度、缓存命中和稳定Decode；Nsight",
        "> 会扰动绝对延迟，正式TPOT仍应以非Profiler性能脚本为准。", "",
        "## 总览", "",
        "| Batch | 后端 | TPOT(ms) | ITL(ms) | 请求Wall(ms) | GPU Kernel(ms) | "
        "Kernel数 | MemOp(ms) | Cache命中 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        nsys = row["nsys"]
        cache = row["cache_hit_ratio"]
        lines.append(
            f"| {row['batch']} | {row['backend']} | {fmt(row['tpot_ms'])} | "
            f"{fmt(row['itl_ms'])} | {fmt(row['request_wall_ms'])} | "
            f"{fmt(nsys['kernel_total_ms'])} | {nsys['kernel_instances']} | "
            f"{fmt(nsys['memory_total_ms'])} | "
            f"{'n/a' if cache is None else f'{cache * 100:.1f}%'} |")

    lines.extend(["", "## 算子类别", "",
                  "| Batch | 后端 | 类别 | GPU时间(ms) | Kernel数 |",
                  "| ---: | --- | --- | ---: | ---: |"])
    for row in rows:
        for category, value in sorted(
                row["nsys"]["categories"].items(),
                key=lambda item: item[1]["total_ms"], reverse=True):
            lines.append(
                f"| {row['batch']} | {row['backend']} | {category} | "
                f"{value['total_ms']:.3f} | {value['instances']} |")

    for row in rows:
        lines.extend([
            "", f"## Batch={row['batch']} / {row['backend']} Top Kernels", "",
            "| Kernel | 调用次数 | 总时间(ms) | 平均(us) |",
            "| --- | ---: | ---: | ---: |",
        ])
        for item in row["nsys"]["top_kernels"][:top_count]:
            name = item["name"].replace("|", "\\|")
            lines.append(
                f"| `{name}` | {item['instances']} | "
                f"{item['total_ms']:.3f} | {item['average_us']:.3f} |")
    report = "\n".join(lines) + "\n"
    path = result_dir / "decode-nsys-compare.md"
    path.write_text(report, encoding="utf-8")
    (result_dir / "decode-nsys-compare.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report)
    print(f"Markdown: {path}")


def main():
    args = parse_args()
    batches = parse_positive_csv(args.batch_sizes, "batch-sizes")
    backends = selected_backends(args.backends)
    for name in ("prompt_tokens", "output_tokens", "warmup_output_tokens",
                 "max_model_len", "startup_timeout", "request_timeout"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')}必须大于0")
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = {
        batch: make_prompt_tokens(tokenizer, args.prompt_tokens, batch)
        for batch in batches
    }
    (result_dir / "prompt-token-ids.json").write_text(json.dumps(
        {str(batch): value for batch, value in prompts.items()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for backend in backends:
        rows.extend(profile_backend(
            args, backend, batches, prompts, result_dir))
    rows.sort(key=lambda item: (item["batch"], BACKENDS.index(item["backend"])))
    make_report(result_dir, rows, args.top_kernels)


if __name__ == "__main__":
    main()
