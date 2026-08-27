#!/usr/bin/env python3
"""用Nsight Compute对比FastLLM与vLLM的Attention合并Kernel。"""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time


REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "test" / "nvfp4"))

import decode_nsys_compare as decode_nsys  # noqa: E402


BACKEND_KERNELS = {
    "fastllm": "PersistentVariableLengthMergeStatesKernel",
    "vllm": "flash_fwd_splitkv_combine_kernel",
}

SUMMARY_METRICS = {
    "Duration": ("gpu__time_duration.sum",),
    "DRAM Read Bytes": ("dram__bytes_read.sum",),
    "DRAM Write Bytes": ("dram__bytes_write.sum",),
    "DRAM Throughput": (
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",),
    "Achieved Occupancy": (
        "sm__warps_active.avg.pct_of_peak_sustained_active",),
    "Registers/Thread": ("launch__registers_per_thread",),
    "Shared Memory/Block": (
        "launch__shared_mem_per_block", "launch__shared_mem_per_block_allocated"),
    "Waves/SM": ("launch__waves_per_multiprocessor",),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="对比相同Decode形状下两个Attention合并Kernel的NCU指标")
    parser.add_argument("--model", required=True)
    parser.add_argument("--nsys-result-dir", required=True,
                        help="同版本decode_nsys_compare.py生成的校准结果目录")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--quantization", choices=("nvfp4", "w8a8"),
                        default="w8a8")
    parser.add_argument("--backends", default="fastllm,vllm")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--prefix-cache-padding-tokens", type=int, default=128)
    parser.add_argument("--vllm-python", default=os.environ.get(
        "NVFP4_VLLM_PYTHON", sys.executable))
    parser.add_argument("--fastllm-python", default=sys.executable)
    parser.add_argument("--flm-dtype", default="auto")
    parser.add_argument("--flm-atype", default="bfloat16")
    parser.add_argument("--flm-device", default="cuda")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--profile-timeout", type=int, default=1800)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--set", dest="section_set", default="full")
    parser.add_argument(
        "--profile-cuda-graph", action="store_true",
        help="实验选项：在CUDA Graph内采集；部分NCU版本无法后挂接节点")
    parser.add_argument(
        "--vllm-extra-arg", action="append", default=[],
        help="追加一个vLLM服务参数；需要多个参数时重复使用")
    return parser.parse_args()


def selected_backends(value):
    result = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not result or any(item not in BACKEND_KERNELS for item in result):
        raise ValueError("backends只能包含fastllm和vllm")
    return list(dict.fromkeys(result))


def read_calibration(result_dir, backend, batch, marker):
    """读取Nsight Systems稳定Decode边界并计算NCU应跳过的目标调用数。"""
    result_path = result_dir / backend / f"batch-{batch}" / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"缺少Nsight校准结果：{result_path}")
    row = json.loads(result_path.read_text(encoding="utf-8"))
    timeline_path = Path(row["nsys"]["timeline_csv"])
    if not timeline_path.exists():
        raise FileNotFoundError(f"缺少Nsight逐事件GPU Trace：{timeline_path}")
    boundary = row["nsys"]["stable_decode_filter"]["boundary_ns"]
    before = 0
    stable = 0
    for event in decode_nsys.read_stats_csv(timeline_path):
        name = decode_nsys.trace_name(event)
        if marker.lower() not in name.lower():
            continue
        start = decode_nsys.numeric(event, ("Start (ns)", "Start"))
        if start < boundary:
            before += 1
        else:
            stable += 1
    if stable <= 0:
        raise RuntimeError(
            f"{backend}稳定Decode区间未找到目标Kernel：{marker}")
    return {
        "result": row,
        "timeline_csv": str(timeline_path),
        "launch_skip": before,
        "stable_matches": stable,
        "boundary_ns": boundary,
    }


def wait_pid_files(path, process, timeout, log_path):
    deadline = time.time() + timeout
    while time.time() < deadline:
        pid_files = list(path.glob("*.pid")) if path.exists() else []
        if pid_files:
            return pid_files
        if process.poll() is not None:
            tail = log_path.read_text(
                encoding="utf-8", errors="replace")[-12000:]
            raise RuntimeError(f"NCU目标服务提前退出：\n{tail}")
        time.sleep(0.1)
    raise TimeoutError("等待NCU目标PID超时")


def signal_target_processes(pid_dir, signum):
    roots = []
    for pid_file in sorted(pid_dir.glob("*.pid")):
        try:
            roots.append(int(pid_file.read_text(encoding="utf-8").strip()))
        except ValueError:
            continue
    # multiprocessing使用spawn时子进程会自行登记；使用fork时只继承信号
    # handler而不会重新执行sitecustomize。递归读取/proc补齐后一种子进程。
    candidates = set(roots)
    pending = list(roots)
    while pending:
        parent = pending.pop()
        children_path = Path(
            f"/proc/{parent}/task/{parent}/children")
        try:
            children = [int(value) for value in
                        children_path.read_text().split()]
        except (OSError, ValueError):
            continue
        for child in children:
            if child not in candidates:
                candidates.add(child)
                pending.append(child)

    signaled = []
    for pid in sorted(candidates):
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().lower()
            # 只通知已登记进程或其Python子进程，避免误杀外部辅助程序。
            if pid not in roots and b"python" not in command:
                continue
            os.kill(pid, signum)
            signaled.append(pid)
        except OSError:
            continue
    if not signaled:
        raise RuntimeError("没有找到可控制的NCU Python目标进程")
    return signaled


def find_report(output_base):
    direct = output_base.with_suffix(".ncu-rep")
    if direct.exists():
        return direct
    matches = sorted(output_base.parent.glob(output_base.name + "*.ncu-rep"))
    if matches:
        return matches[-1]
    raise RuntimeError(f"Nsight Compute没有生成报告：{output_base}.ncu-rep")


def stop_process_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_profile(args, backend, prompt, calibration, backend_dir):
    """启动关闭默认采集的NCU服务，只剖析首个稳定Decode合并Kernel。"""
    backend_dir.mkdir(parents=True, exist_ok=True)
    server_command, target_env, served_name = decode_nsys.server_spec(
        args, backend, args.batch)
    if not args.profile_cuda_graph:
        # NCU在CUDA Graph已经录制后才由cudaProfilerStart挂接时，部分版本
        # 看不到已有Graph内的独立节点。微观对比只关闭Graph包装，模型、
        # Prompt、Batch、Kernel和张量形状保持不变；正式TPOT仍以BEST测试为准。
        if backend == "fastllm":
            target_env["FASTLLM_CUDA_GRAPH"] = "0"
        elif "--enforce-eager" not in server_command:
            server_command.append("--enforce-eager")
    pid_dir = backend_dir / "target-pids"
    pid_dir.mkdir(parents=True, exist_ok=True)
    for stale_pid in pid_dir.glob("*.pid"):
        stale_pid.unlink()
    target_env["FASTLLM_NCU_TARGET_PID_DIR"] = str(pid_dir)
    sitecustomize_dir = str(
        REPO_DIR / "test" / "nvfp4" / "ncu_sitecustomize")
    current_pythonpath = target_env.get("PYTHONPATH", "")
    target_env["PYTHONPATH"] = (
        sitecustomize_dir if not current_pythonpath else
        f"{sitecustomize_dir}{os.pathsep}{current_pythonpath}")
    target = decode_nsys.target_command(server_command)
    marker = BACKEND_KERNELS[backend]
    output_base = backend_dir / "attention-merge"
    ncu_command = [
        args.ncu, "--target-processes", "all",
        "--profile-from-start", "off", "--kill", "yes",
        "--replay-mode", "kernel", "--cache-control", "none",
        "--clock-control", "none", "--kernel-name-base", "demangled",
        "--kernel-name", f"regex:.*{re.escape(marker)}.*",
        "--launch-skip", str(calibration["launch_skip"]),
        "--launch-count", "1", "--set", args.section_set,
        "--export", str(output_base), "--force-overwrite",
        *target,
    ]
    log_path = backend_dir / "ncu-run.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"COMMAND: {shlex.join(ncu_command)}\n\n")
        log_handle.flush()
        control_env = decode_nsys.nsys_control_env()
        control_env.update({key: value for key, value in target_env.items()
                            if key != "LD_PRELOAD"})
        process = subprocess.Popen(
            ncu_command, cwd=REPO_DIR, env=control_env, stdout=log_handle,
            stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            wait_pid_files(
                pid_dir, process, args.startup_timeout, log_path)
            server = {
                "base_url": f"http://127.0.0.1:{args.port}",
                "served_name": served_name, "backend": backend,
            }
            decode_nsys.wait_server(
                server["base_url"], process, args.startup_timeout,
                log_path, backend)
            warmup_prompt = decode_nsys.make_cache_warmup_prompt(
                prompt, args.prefix_cache_padding_tokens)
            decode_nsys.run_decode_batch(
                server, warmup_prompt, args.warmup_output_tokens,
                args.batch, args, "ncu-warmup")
            # FastLLM通常只有一个Python进程；vLLM还包含真正持有CUDA
            # Context的EngineCore子进程。只通知API Server会漏采combine。
            profiled_pids = signal_target_processes(pid_dir, signal.SIGUSR1)
            log_handle.write(
                f"PROFILER START PIDS: {','.join(map(str, profiled_pids))}\n")
            log_handle.flush()
            time.sleep(0.2)
            request_completed = False
            try:
                decode_nsys.run_decode_batch(
                    server, prompt, args.output_tokens,
                    args.batch, args, "ncu-profile")
                request_completed = True
            except Exception as error:
                # --kill=yes会在目标Kernel采集完毕后终止服务，因此正式HTTP
                # 请求通常在完整Token返回前断开；报告存在时这是预期行为。
                log_handle.write(f"EXPECTED PROFILE REQUEST END: {error}\n")
                log_handle.flush()
            if request_completed and process.poll() is None:
                # --kill=yes应在目标Kernel采集后终止服务。完整请求已经返回却
                # 仍在运行，说明过滤条件没有命中；立即失败，禁止空等超时。
                signal_target_processes(pid_dir, signal.SIGUSR2)
                raise RuntimeError(
                    f"NCU未命中目标Kernel：{marker}；完整HTTP请求已结束")
            process.wait(timeout=args.profile_timeout)
        except Exception:
            stop_process_group(process)
            raise
        finally:
            stop_process_group(process)
    report = find_report(output_base)
    return report, log_path


def parse_ncu_raw(args, report, output_csv):
    command = [
        args.ncu, "--import", str(report), "--csv", "--page", "raw",
        "--print-metric-name", "name", "--print-units", "base",
    ]
    result = subprocess.run(
        command, cwd=REPO_DIR, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_csv.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"导出NCU raw指标失败：\n{result.stdout[-12000:]}")
    lines = result.stdout.splitlines()
    header = next((index for index, line in enumerate(lines)
                   if re.match(r'^"?ID"?,', line)), None)
    if header is None:
        raise RuntimeError("NCU raw输出中没有找到CSV表头")
    rows = list(csv.DictReader(lines[header:]))
    if not rows:
        raise RuntimeError("NCU raw输出没有指标记录")
    return rows


def metric_number(value):
    if value is None:
        return None
    cleaned = str(value).replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_metrics(rows):
    metrics = []
    for row in rows:
        name = row.get("Metric Name", "").strip()
        if not name:
            continue
        metrics.append({
            "name": name,
            "value": metric_number(row.get("Metric Value")),
            "value_text": row.get("Metric Value", "").strip(),
            "unit": row.get("Metric Unit", "").strip(),
        })
    first = rows[0]
    return metrics, {
        "kernel": first.get("Kernel Name", ""),
        "grid": first.get("Grid Size", first.get("GridXYZ", "")),
        "block": first.get("Block Size", first.get("BlockXYZ", "")),
        "device": first.get("Device Name", first.get("Device", "")),
    }


def select_metric(metrics, candidates):
    by_name = {item["name"]: item for item in metrics}
    for candidate in candidates:
        if candidate in by_name:
            return by_name[candidate]
    return None


def duration_ns(metric):
    if metric is None or metric["value"] is None:
        return None
    unit = metric["unit"].lower()
    factors = {
        "ns": 1.0, "nsecond": 1.0, "nanosecond": 1.0,
        "us": 1e3, "usecond": 1e3, "microsecond": 1e3,
        "ms": 1e6, "msecond": 1e6, "millisecond": 1e6,
        "s": 1e9, "second": 1e9,
    }
    return metric["value"] * factors.get(unit, 1.0)


def summarize_backend(backend, report, raw_csv, calibration, rows,
                      profiling_mode):
    metrics, launch = normalize_metrics(rows)
    selected = {
        label: select_metric(metrics, candidates)
        for label, candidates in SUMMARY_METRICS.items()
    }
    stalls = [item for item in metrics
              if "warp_issue_stalled" in item["name"] and
              item["value"] is not None]
    stalls.sort(key=lambda item: item["value"], reverse=True)
    return {
        "backend": backend,
        "kernel_marker": BACKEND_KERNELS[backend],
        "report": str(report),
        "raw_csv": str(raw_csv),
        "launch_skip": calibration["launch_skip"],
        "stable_matches_in_calibration": calibration["stable_matches"],
        "grid": launch["grid"], "block": launch["block"],
        "device": launch["device"], "kernel": launch["kernel"],
        "selected_metrics": selected,
        "top_warp_stalls": stalls[:10],
        "all_metrics": metrics,
        "split_count": None,
        "split_count_note": "NCU不直接暴露逻辑Split数量，不能仅凭Grid臆测",
        "profiling_mode": profiling_mode,
    }


def metric_text(metric):
    if metric is None:
        return "n/a"
    return f"{metric['value_text']} {metric['unit']}".strip()


def make_reports(result_dir, summaries):
    lines = [
        "# FastLLM / vLLM Attention合并Kernel NCU对比", "",
        "> NCU使用Nsight Systems稳定Decode边界校准`launch-skip`，只采集",
        "> Batch=1正式Decode中的首个目标合并Kernel。NCU会重放Kernel，",
        "> 本报告用于微观归因，不能替代未使用Profiler的TPOT。", "",
        "> 默认关闭双方CUDA Graph，避免NCU后挂接时遗漏已录制Graph节点；",
        "> Kernel与张量形状不变。正式BEST性能仍使用CUDA Graph。", "",
        "| 后端 | Kernel | Grid | Block | Duration | DRAM Read | DRAM Write | "
        "DRAM吞吐 | Occupancy | Registers/Thread | Shared Memory/Block |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        selected = item["selected_metrics"]
        lines.append(
            f"| {item['backend']} | `{item['kernel_marker']}` | "
            f"{item['grid'] or 'n/a'} | {item['block'] or 'n/a'} | "
            f"{metric_text(selected['Duration'])} | "
            f"{metric_text(selected['DRAM Read Bytes'])} | "
            f"{metric_text(selected['DRAM Write Bytes'])} | "
            f"{metric_text(selected['DRAM Throughput'])} | "
            f"{metric_text(selected['Achieved Occupancy'])} | "
            f"{metric_text(selected['Registers/Thread'])} | "
            f"{metric_text(selected['Shared Memory/Block'])} |")

    by_backend = {item["backend"]: item for item in summaries}
    if "fastllm" in by_backend and "vllm" in by_backend:
        fast_duration = duration_ns(
            by_backend["fastllm"]["selected_metrics"]["Duration"])
        vllm_duration = duration_ns(
            by_backend["vllm"]["selected_metrics"]["Duration"])
        ratio = (None if not fast_duration or vllm_duration is None else
                 vllm_duration / fast_duration)
        lines.extend([
            "", "## 时长结论", "",
            f"- vLLM/FastLLM Kernel时长比："
            f"{'n/a' if ratio is None else f'{ratio:.3f}x'}",
            "- Split数量：NCU不直接暴露，不能仅凭Grid判定工作量相同。",
        ])

    for item in summaries:
        lines.extend([
            "", f"## {item['backend']} Warp Stall", "",
            "| Metric | Value | Unit |", "| --- | ---: | --- |",
        ])
        for metric in item["top_warp_stalls"]:
            lines.append(
                f"| `{metric['name']}` | {metric['value_text']} | {metric['unit']} |")

    md_path = result_dir / "attention-ncu-compare.md"
    json_path = result_dir / "attention-ncu-compare.json"
    xlsx_path = result_dir / "attention-ncu-compare.xlsx"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows = []
    all_metric_rows = []
    stall_rows = []
    for item in summaries:
        selected = item["selected_metrics"]
        summary_rows.append([
            item["backend"], item["kernel_marker"], item["grid"], item["block"],
            metric_text(selected["Duration"]),
            metric_text(selected["DRAM Read Bytes"]),
            metric_text(selected["DRAM Write Bytes"]),
            metric_text(selected["DRAM Throughput"]),
            metric_text(selected["Achieved Occupancy"]),
            metric_text(selected["Registers/Thread"]),
            metric_text(selected["Shared Memory/Block"]),
            item["launch_skip"], item["stable_matches_in_calibration"],
        ])
        for metric in item["all_metrics"]:
            all_metric_rows.append([
                item["backend"], metric["name"], metric["value"],
                metric["value_text"], metric["unit"],
            ])
        for metric in item["top_warp_stalls"]:
            stall_rows.append([
                item["backend"], metric["name"], metric["value"],
                metric["value_text"], metric["unit"],
            ])
    decode_nsys.write_xlsx(xlsx_path, [
        ("关键指标", [
            "后端", "Kernel", "Grid", "Block", "Duration", "DRAM Read",
            "DRAM Write", "DRAM吞吐", "Occupancy", "Registers/Thread",
            "Shared Memory/Block", "launch-skip", "校准区间稳定调用数"],
         summary_rows),
        ("Warp Stall", ["后端", "Metric", "数值", "原始值", "单位"],
         stall_rows),
        ("全部NCU指标", ["后端", "Metric", "数值", "原始值", "单位"],
         all_metric_rows),
    ])
    return md_path, json_path, xlsx_path


def main():
    args = parse_args()
    if args.batch <= 0 or args.prompt_tokens <= 0 or args.output_tokens <= 1:
        raise ValueError("batch和prompt-tokens必须大于0，output-tokens必须大于1")
    backends = selected_backends(args.backends)
    result_dir = Path(args.result_dir).resolve()
    calibration_dir = Path(args.nsys_result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = decode_nsys.make_prompt_tokens(
        tokenizer, args.prompt_tokens, args.batch)
    calibration_prompts = calibration_dir / "prompt-token-ids.json"
    if calibration_prompts.exists():
        saved = json.loads(calibration_prompts.read_text(encoding="utf-8"))
        if saved.get(str(args.batch)) != prompt:
            raise RuntimeError("当前Prompt与Nsight Systems校准Prompt不一致")

    summaries = []
    for index, backend in enumerate(backends):
        marker = BACKEND_KERNELS[backend]
        calibration = read_calibration(
            calibration_dir, backend, args.batch, marker)
        backend_dir = result_dir / backend
        original_port = args.port
        args.port = original_port + index
        try:
            report, _ = run_profile(
                args, backend, prompt, calibration, backend_dir)
        finally:
            args.port = original_port
        raw_csv = backend_dir / "attention-merge-raw.csv"
        raw_rows = parse_ncu_raw(args, report, raw_csv)
        summaries.append(summarize_backend(
            backend, report, raw_csv, calibration, raw_rows,
            "cuda_graph" if args.profile_cuda_graph else "eager"))

    md_path, json_path, xlsx_path = make_reports(result_dir, summaries)
    print(f"Markdown: {md_path}")
    print(f"JSON: {json_path}")
    print(f"Excel: {xlsx_path}")


if __name__ == "__main__":
    main()
