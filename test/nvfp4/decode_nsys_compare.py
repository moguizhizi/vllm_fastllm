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
    parser.add_argument(
        "--prefix-cache-padding-tokens", type=int, default=128,
        help="Warmup Prompt在正式Prompt之后追加的Token数，用于缓存正式Prompt的最后一页")
    parser.add_argument(
        "--collection-ready-timeout", type=float, default=30.0,
        help="等待Nsight会话进入Collection状态的超时时间（秒）")
    parser.add_argument(
        "--collection-ready-delay", type=float, default=1.0,
        help="Nsight进入Collection状态后、发送正式请求前额外等待的时间（秒）")
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


def wait_collection_ready(args, session):
    """等待Nsight真正进入采集状态，避免遗漏短暂的Prefill开头。"""
    deadline = time.time() + args.collection_ready_timeout
    last_output = ""
    while time.time() < deadline:
        result = subprocess.run(
            [str(args.nsys), "sessions", "list"], cwd=REPO_DIR,
            env=nsys_control_env(), check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        last_output = result.stdout or ""
        if (result.returncode == 0 and session in last_output and
                re.search(r"\bCollection\b", last_output)):
            # Collection状态表示控制面已切换；再留出少量时间，让目标进程的
            # CUDA追踪完成挂接，然后才发送本次需要统计的HTTP请求。
            time.sleep(args.collection_ready_delay)
            return
        time.sleep(0.1)
    raise RuntimeError(
        f"等待Nsight会话{session}进入Collection状态超时：\n{last_output}")


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


def make_cache_warmup_prompt(prompt, padding_tokens):
    """构造以正式Prompt为完整前缀的加长Prompt，使最后一个Prefix页可缓存。"""
    if not prompt:
        raise ValueError("Decode Nsight Prompt不能为空")
    if padding_tokens <= 0:
        return list(prompt)
    # 复用Prompt中已有的有效Token ID，避免依赖不同Tokenizer的特殊Token。
    padding_source = prompt[-min(len(prompt), padding_tokens):]
    repeats = (padding_tokens + len(padding_source) - 1) // len(padding_source)
    return list(prompt) + (padding_source * repeats)[:padding_tokens]


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


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


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
    # 分类必须先识别语义明确的kernel，不能看到字符串"cutlass"就认为是
    # GEMM：FlashAttention模板参数也包含cutlass类型，Triton融合kernel名称
    # 则可能同时出现scaled_mm、RMSNorm、SwiGLU和量化。
    if any(pattern in lowered for pattern in (
            "qkvrmsnormropesplit", "fused_add_rms_norm")):
        return "Fused"
    if any(pattern in lowered for pattern in (
            "reshape_and_cache", "pagedcachecopy", "appendpagedcache",
            "kv_cache", "cache_kernel", "gather_block_tables")):
        return "KV Cache"
    if any(pattern in lowered for pattern in (
            "flash_fwd", "flashinfer::batchprefill", "attention", "fmha",
            "fused_attn", "persistentvariablelengthmergestates",
            "flashmla", "mla::")):
        return "Attention"
    if any(pattern in lowered for pattern in (
            "gumbel", "sampling", "sample_kernel", "topk", "top_k", "topp",
            "top_p", "argmax", "layernormkerneltop1", "resetlogitsofeos")):
        return "Sampling"
    semantic_markers = sum(pattern in lowered for pattern in (
        "scaled_mm", "rms_norm", "rmsnorm", "silu", "swiglu", "quant",
        "embedding", "rope"))
    if "triton" in lowered and semantic_markers >= 2:
        return "Fused"
    if any(pattern in lowered for pattern in (
            "quantize", "scaled_fp8_quant", "nvfp4quant", "float4quant")):
        return "Quantize"
    if any(pattern in lowered for pattern in (
            "rmsnorm", "rms_norm", "layernorm", "layer_norm")):
        return "Norm"
    if any(pattern in lowered for pattern in (
            "swiglu", "silu", "gelu", "activation")):
        return "Activation"
    if any(pattern in lowered for pattern in (
            "gemmkernel", "gemm::kernel", "gemv", "scaled_mm",
            "matmul", "marlin", "kernel2<cutlass")):
        return "GEMM/Linear"
    return "Other"


def summarize_gpu_timeline(rows):
    intervals = []
    for row in rows:
        start = numeric(row, ("Start (ns)", "Start"))
        duration = numeric(row, ("Duration (ns)", "Duration"))
        if duration > 0:
            intervals.append((start, start + duration))
    if not intervals:
        return {
            "event_count": 0, "span_ms": 0.0, "active_ms": 0.0,
            "idle_ms": 0.0, "idle_ratio": None,
        }
    intervals.sort()
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    span_ns = max(end for _, end in intervals) - min(
        start for start, _ in intervals)
    active_ns = sum(end - start for start, end in merged)
    idle_ns = max(0.0, span_ns - active_ns)
    return {
        "event_count": len(intervals),
        "span_ms": span_ns / 1e6,
        "active_ms": active_ns / 1e6,
        "idle_ms": idle_ns / 1e6,
        "idle_ratio": idle_ns / span_ns if span_ns > 0 else 0.0,
    }


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
    timeline_csv = export_stats(
        args, report, "cuda_gpu_trace", output_base.with_name(
            output_base.name + "-gpu-trace"))
    kernels = summarize_named_rows(read_stats_csv(kernel_csv))
    apis = summarize_named_rows(read_stats_csv(api_csv))
    memory = summarize_named_rows(read_stats_csv(memory_csv))
    timeline = summarize_gpu_timeline(read_stats_csv(timeline_csv))
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
        "timeline_csv": str(timeline_csv),
        "kernel_total_ms": sum(item["total_ms"] for item in kernels),
        "kernel_instances": sum(item["instances"] for item in kernels),
        "memory_total_ms": sum(item["total_ms"] for item in memory),
        "timeline": timeline,
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
            cache_warmup_prompt = make_cache_warmup_prompt(
                prompt, args.prefix_cache_padding_tokens)
            if (len(cache_warmup_prompt) + args.warmup_output_tokens >
                    args.max_model_len):
                raise ValueError(
                    "加长后的Prefix Cache Warmup超过max-model-len："
                    f"{len(cache_warmup_prompt)} + "
                    f"{args.warmup_output_tokens} > {args.max_model_len}")
            # Warmup Prompt把正式Prompt作为完整前缀，并在末尾追加一个Cache
            # 页。这样正式Prompt的最后一页也能复用，避免Decode报告混入
            # 一个未命中的Prefill尾块。
            run_decode_batch(
                server, cache_warmup_prompt,
                args.warmup_output_tokens, batch, args,
                "warmup")
            output_base = case_dir / "decode"
            start_collection(args, server["session"], output_base)
            wait_collection_ready(args, server["session"])
            request_result = run_decode_batch(
                server, prompt, args.output_tokens, batch, args,
                "decode")
            report = stop_collection(args, server["session"], output_base)
            cache_ratio = request_result["cache_hit_ratio"]
            if (backend == "fastllm" and
                    (cache_ratio is None or cache_ratio < 0.999)):
                actual = "未报告" if cache_ratio is None else f"{cache_ratio:.1%}"
                raise RuntimeError(
                    "FastLLM Decode Nsight采集要求Prefix Cache完全命中，"
                    f"实际为{actual}；请检查Cache页大小和Warmup Prompt。")
            summary = summarize_report(args, report, output_base)
            # 第一个输出Token属于Prefill结束点；之后每个Token才对应一次稳定
            # Decode迭代。按decode步数归一化后，才能比较不同batch下每步的
            # kernel数量与GPU时间，而不会把请求长度误认为算子变慢。
            decode_steps = max(args.output_tokens - 1, 1)
            request_tpots_ms = []
            request_e2els_ms = []
            for request in request_result["requests"]:
                token_times = request["token_times"]
                request_e2els_ms.append(
                    (request["end_time"] - request["start_time"]) * 1000)
                if len(token_times) > 1:
                    request_tpots_ms.append(
                        (request["end_time"] - token_times[0]) * 1000 /
                        (len(token_times) - 1))
            summary["decode_steps"] = decode_steps
            summary["kernel_ms_per_decode_step"] = (
                summary["kernel_total_ms"] / decode_steps)
            summary["kernel_instances_per_decode_step"] = (
                summary["kernel_instances"] / decode_steps)
            summary["memory_ms_per_decode_step"] = (
                summary["memory_total_ms"] / decode_steps)
            for value in summary["categories"].values():
                value["total_ms_per_decode_step"] = (
                    value["total_ms"] / decode_steps)
                value["instances_per_decode_step"] = (
                    value["instances"] / decode_steps)
            timeline = summary["timeline"]
            timeline["span_ms_per_decode_step"] = (
                timeline["span_ms"] / decode_steps)
            timeline["active_ms_per_decode_step"] = (
                timeline["active_ms"] / decode_steps)
            timeline["idle_ms_per_decode_step"] = (
                timeline["idle_ms"] / decode_steps)
            row = {
                "backend": backend,
                "batch": batch,
                "prompt_tokens": len(prompt),
                "warmup_prompt_tokens": len(cache_warmup_prompt),
                "output_tokens_per_request": args.output_tokens,
                "request_wall_ms": request_result["wall_s"] * 1000,
                "tpot_ms": (None if request_result["tpot_s"] is None else
                            request_result["tpot_s"] * 1000),
                "itl_ms": (None if request_result["itl_s"] is None else
                           request_result["itl_s"] * 1000),
                "request_tpot_p50_ms": percentile(request_tpots_ms, 50),
                "request_tpot_p95_ms": percentile(request_tpots_ms, 95),
                "request_e2el_p50_ms": percentile(request_e2els_ms, 50),
                "request_e2el_p95_ms": percentile(request_e2els_ms, 95),
                "output_tok_s": request_result["output_tok_s"],
                "cache_hit_ratio": cache_ratio,
                "cache_status": ("reported" if cache_ratio is not None else
                                 "unreported"),
                "nsys": summary,
            }
            (case_dir / "result.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(row)
            print(json.dumps({
                "backend": backend, "batch": batch,
                "tpot_ms": row["tpot_ms"],
                "output_tok_s": row["output_tok_s"],
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
        "> 每个Batch先使用加长Prompt完成warmup，使正式Prompt成为完整缓存前缀；",
        "> Nsight进入Collection状态并稳定后才发送BEST模式HTTP请求。FastLLM",
        "> 要求正式请求Prefix Cache命中率为100%，避免报告混入Prefill尾块；Nsight",
        "> 会扰动绝对延迟，正式TPOT仍应以非Profiler性能脚本为准。",
        "> GPU时间线空闲只统计首个与最后一个GPU事件之间的内部空隙；",
        "> 第一个输出Token属于Prefill，逐步指标按`output_tokens - 1`归一化。", "",
        "## 引擎指标", "",
        "| Batch | 后端 | 平均TPOT(ms) | 请求TPOT P50(ms) | 请求TPOT P95(ms) | "
        "ITL(ms) | 请求E2EL P50(ms) | 请求E2EL P95(ms) | Batch Wall(ms) | "
        "Output(tok/s) | Cache命中 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | --- |",
    ]
    for row in rows:
        cache = row["cache_hit_ratio"]
        cache_text = ("未报告" if row["cache_status"] == "unreported" else
                      f"{cache * 100:.1f}%")
        lines.append(
            f"| {row['batch']} | {row['backend']} | {fmt(row['tpot_ms'])} | "
            f"{fmt(row['request_tpot_p50_ms'])} | "
            f"{fmt(row['request_tpot_p95_ms'])} | {fmt(row['itl_ms'])} | "
            f"{fmt(row['request_e2el_p50_ms'])} | "
            f"{fmt(row['request_e2el_p95_ms'])} | "
            f"{fmt(row['request_wall_ms'])} | {fmt(row['output_tok_s'], 2)} | "
            f"{cache_text} |")

    lines.extend([
        "", "## GPU时间线", "",
        "| Batch | 后端 | Decode步数 | Kernel(ms/步) | Kernel数/步 | "
        "MemOp(ms/步) | GPU跨度(ms/步) | GPU活跃(ms/步) | "
        "GPU内部空闲(ms/步) | 空闲占比 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        nsys = row["nsys"]
        timeline = nsys["timeline"]
        idle_ratio = timeline["idle_ratio"]
        lines.append(
            f"| {row['batch']} | {row['backend']} | {nsys['decode_steps']} | "
            f"{fmt(nsys['kernel_ms_per_decode_step'])} | "
            f"{fmt(nsys['kernel_instances_per_decode_step'], 2)} | "
            f"{fmt(nsys['memory_ms_per_decode_step'])} | "
            f"{fmt(timeline['span_ms_per_decode_step'])} | "
            f"{fmt(timeline['active_ms_per_decode_step'])} | "
            f"{fmt(timeline['idle_ms_per_decode_step'])} | "
            f"{'n/a' if idle_ratio is None else f'{idle_ratio * 100:.1f}%'} |")

    by_case = {(row["batch"], row["backend"]): row for row in rows}
    paired_batches = sorted({
        row["batch"] for row in rows
        if (row["batch"], "fastllm") in by_case and
        (row["batch"], "vllm") in by_case
    })
    if paired_batches:
        lines.extend([
            "", "## FastLLM / vLLM比值", "",
            "> 延迟比小于1表示FastLLM更快；吞吐比大于1表示FastLLM更快。", "",
            "| Batch | TPOT比 | Batch Wall比 | Output吞吐比 | "
            "Kernel时间/步比 | GPU内部空闲/步比 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for batch in paired_batches:
            fast = by_case[(batch, "fastllm")]
            vllm = by_case[(batch, "vllm")]

            def ratio(left, right):
                if left is None or right is None or right == 0:
                    return None
                return left / right

            def ratio_text(left, right):
                value = ratio(left, right)
                return "n/a" if value is None else f"{value:.3f}x"

            lines.append(
                f"| {batch} | {ratio_text(fast['tpot_ms'], vllm['tpot_ms'])} | "
                f"{ratio_text(fast['request_wall_ms'], vllm['request_wall_ms'])} | "
                f"{ratio_text(fast['output_tok_s'], vllm['output_tok_s'])} | "
                f"{ratio_text(fast['nsys']['kernel_ms_per_decode_step'], vllm['nsys']['kernel_ms_per_decode_step'])} | "
                f"{ratio_text(fast['nsys']['timeline']['idle_ms_per_decode_step'], vllm['nsys']['timeline']['idle_ms_per_decode_step'])} |")

    lines.extend(["", "## 算子类别", "",
                  "| Batch | 后端 | 类别 | GPU总时间(ms) | GPU时间(ms/步) | "
                  "Kernel总数 | Kernel数/步 |",
                  "| ---: | --- | --- | ---: | ---: | ---: | ---: |"])
    for row in rows:
        for category, value in sorted(
                row["nsys"]["categories"].items(),
                key=lambda item: item[1]["total_ms"], reverse=True):
            lines.append(
                f"| {row['batch']} | {row['backend']} | {category} | "
                f"{value['total_ms']:.3f} | "
                f"{value['total_ms_per_decode_step']:.3f} | "
                f"{value['instances']} | "
                f"{value['instances_per_decode_step']:.2f} |")

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
    if args.prefix_cache_padding_tokens <= 0:
        raise ValueError("prefix-cache-padding-tokens必须大于0")
    if args.collection_ready_timeout <= 0:
        raise ValueError("collection-ready-timeout必须大于0")
    if args.collection_ready_delay < 0:
        raise ValueError("collection-ready-delay不能小于0")
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
