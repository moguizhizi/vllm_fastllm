#!/usr/bin/env python3
"""Compare one NVFP4 FastLLM request with vLLM in isolated processes."""

import argparse
import csv
import ctypes
import heapq
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "test" / "basic"))
from config import default_messages_list  # noqa: E402
from xlsx_report import write_xlsx  # noqa: E402


FASTLLM_PRODUCTION_TRACE_BEGIN = "FASTLLM_PRODUCTION_TRACE_BEGIN"
FASTLLM_PRODUCTION_TRACE_END = "FASTLLM_PRODUCTION_TRACE_END"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential vLLM/FastLLM NVFP4 forward check")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--top-logprobs", type=int, default=10)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--flm-dtype", default="auto")
    parser.add_argument("--flm-atype", default="bfloat16")
    parser.add_argument("--flm-device", default="cuda")
    parser.add_argument("--min-topk-overlap", type=float, default=0.8)
    parser.add_argument("--max-first-logprob-diff", type=float, default=0.1)
    parser.add_argument(
        "--strict-alignment", action="store_true",
        help=("require the first-token logprob difference to satisfy "
              "--max-first-logprob-diff; by default it is diagnostic only"))
    parser.add_argument("--label", default="NVFP4", help=argparse.SUPPRESS)
    parser.add_argument("--result-dir", default="")
    parser.add_argument("--target-trace-pattern", default="",
                        help=argparse.SUPPRESS)
    parser.add_argument(
        "--vllm-python", default=os.environ.get("NVFP4_VLLM_PYTHON", sys.executable))
    parser.add_argument("--stage", choices=("orchestrate", "vllm", "fastllm"),
                        default="orchestrate", help=argparse.SUPPRESS)
    parser.add_argument("--output", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def get_messages(case_index):
    if case_index < 0 or case_index >= len(default_messages_list):
        raise ValueError(f"case-index must be in [0, {len(default_messages_list)})")
    return default_messages_list[case_index]


def render_prompt(tokenizer, messages):
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def top_logprobs(logits, count):
    maximum = max(logits)
    log_norm = maximum + math.log(sum(math.exp(value - maximum) for value in logits))
    token_ids = heapq.nlargest(count, range(len(logits)), key=logits.__getitem__)
    return {str(token_id): logits[token_id] - log_norm for token_id in token_ids}


def strip_trailing_eos(tokenizer, token_ids):
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    token_ids = list(token_ids)
    while token_ids and token_ids[-1] in eos_ids:
        token_ids.pop()
    return token_ids


def format_metric(value, digits=6):
    return "n/a" if value is None else f"{value:.{digits}f}"


def run_vllm(args):
    try:
        from transformers import AutoTokenizer
        from vllm.entrypoints.llm import LLM
        from vllm.sampling_params import SamplingParams
    except ImportError as error:
        raise RuntimeError(
            "vLLM mode failed to import its offline LLM API; verify the vLLM "
            "installation or set NVFP4_VLLM_PYTHON=/path/to/python") from error

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = render_prompt(tokenizer, get_messages(args.case_index))
    engine = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        linear_backend="cutlass",
        # linear_backend只控制Dense Linear；MoE必须单独固定为vLLM
        # CUTLASS，否则auto会优先选择FLASHINFER_CUTLASS并触发运行时JIT。
        moe_backend="cutlass",
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.tokens,
        seed=0,
        logprobs=args.top_logprobs,
    )
    completion = engine.generate([prompt], params)[0].outputs[0]
    first_logprobs = {}
    if completion.logprobs:
        first_logprobs = {
            str(token_id): entry.logprob
            for token_id, entry in completion.logprobs[0].items()
        }
    first_argmax = (
        max(first_logprobs, key=first_logprobs.get)
        if first_logprobs else None)
    write_json(args.output, {
        "backend": "vllm",
        "prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=True),
        "generated_token_ids": strip_trailing_eos(tokenizer, completion.token_ids),
        "generated_text": completion.text,
        "first_argmax_token_id": (
            int(first_argmax) if first_argmax is not None else None),
        "first_top_logprobs": first_logprobs,
    })


def generate_fastllm_token_ids(model, tokenizer, prompt, max_tokens):
    """使用正式生产配置执行FastLLM贪心生成。"""
    from ftllm import llm

    input_ids = llm.encode_hf_prompt(tokenizer, prompt)
    input_array = (ctypes.c_int * len(input_ids))(*input_ids)
    stop_count, stop_tokens = model.stop_token_ctypes(None)
    handle = llm.fastllm_lib.launch_response_llm_model(
        model.model, len(input_ids), input_array,
        max_tokens, 0, False, 1.0, 1, 1.0, 1.0, False,
        stop_count, stop_tokens)

    output = []
    while True:
        if not llm.fastllm_lib.can_fetch_response_llm_model(model.model, handle):
            continue
        token_id = llm.fastllm_lib.fetch_response_llm_model(model.model, handle)
        if token_id <= -1:
            break
        output.append(token_id)
    return input_ids, output


def run_fastllm(args):
    from transformers import AutoTokenizer

    # 在加载FastLLM动态库前取得同一份HF tokenizer，避免FastLLM内部
    # tokenizer探测失败后丢失具体异常，导致forward check无法继续。
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    from ftllm import llm

    llm.set_device_map(args.flm_device)
    model = llm.model(args.model, dtype=args.flm_dtype)
    model.set_atype(args.flm_atype)
    if model.hf_tokenizer is not None:
        tokenizer = model.hf_tokenizer
    prompt = render_prompt(tokenizer, get_messages(args.case_index))
    model.direct_query = True

    print(FASTLLM_PRODUCTION_TRACE_BEGIN, flush=True)
    input_ids, generated_ids = generate_fastllm_token_ids(
        model, tokenizer, prompt, args.tokens)
    print(FASTLLM_PRODUCTION_TRACE_END, flush=True)

    # response_logits会令FastLLM进入完整logits返回分支，因此这里只把它
    # 当作独立诊断探针；功能结论始终使用上面的正式生产生成结果。
    probe_logits = model.response_logits(prompt, tokenizer=tokenizer)
    probe_argmax = max(
        range(len(probe_logits)), key=probe_logits.__getitem__)
    write_json(args.output, {
        "backend": "fastllm",
        "prompt_token_ids": input_ids,
        "generated_token_ids": strip_trailing_eos(tokenizer, generated_ids),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "logits_probe_argmax_token_id": probe_argmax,
        "logits_probe_top_logprobs": top_logprobs(
            probe_logits, args.top_logprobs),
    })
    model.release_memory()


def compare_results(args, vllm_result, fastllm_result):
    same_prompt = vllm_result["prompt_token_ids"] == fastllm_result["prompt_token_ids"]
    same_tokens = vllm_result["generated_token_ids"] == fastllm_result["generated_token_ids"]
    v_probs = {int(key): value for key, value in vllm_result["first_top_logprobs"].items()}
    f_probs = {
        int(key): value
        for key, value in fastllm_result["logits_probe_top_logprobs"].items()
    }
    common = sorted(set(v_probs) & set(f_probs))
    denominator = max(1, min(args.top_logprobs, len(v_probs), len(f_probs)))
    overlap = len(common) / denominator
    max_diff = max(
        (abs(v_probs[token] - f_probs[token]) for token in common),
        default=None)
    first_vllm = vllm_result["generated_token_ids"][:1]
    first_fastllm = fastllm_result["generated_token_ids"][:1]
    first_token_match = first_vllm == first_fastllm and bool(first_vllm)
    v_argmax = vllm_result.get("first_argmax_token_id")
    f_probe_argmax = fastllm_result.get("logits_probe_argmax_token_id")
    probe_argmax_match = (
        v_argmax == f_probe_argmax and v_argmax is not None)
    v_generation_matches_argmax = bool(first_vllm) and first_vllm[0] == v_argmax
    f_production_matches_probe = (
        bool(first_fastllm) and first_fastllm[0] == f_probe_argmax)
    target_backend_required = bool(args.target_trace_pattern)
    target_backend_confirmed = fastllm_result.get("target_backend_confirmed")
    target_backend_ok = (
        not target_backend_required or target_backend_confirmed is True)

    v_ranked = sorted(v_probs, key=v_probs.get, reverse=True)
    f_ranked = sorted(f_probs, key=f_probs.get, reverse=True)
    v_token_rank_in_fastllm_probe = (
        f_ranked.index(first_vllm[0]) + 1
        if first_vllm and first_vllm[0] in f_probs else None)
    f_production_rank_in_vllm = (
        v_ranked.index(first_fastllm[0]) + 1
        if first_fastllm and first_fastllm[0] in v_probs else None)

    first_token_logprob_diff = None
    if first_token_match:
        first_token = first_vllm[0]
        if first_token in v_probs and first_token in f_probs:
            first_token_logprob_diff = abs(
                v_probs[first_token] - f_probs[first_token])

    if not same_prompt:
        failure_class = "prompt_token_ids_mismatch"
    elif not target_backend_ok:
        failure_class = "target_backend_not_confirmed"
    elif not v_generation_matches_argmax:
        failure_class = "vllm_generation_differs_from_reported_argmax"
    elif not first_token_match:
        failure_class = "production_first_token_mismatch"
    elif not same_tokens:
        failure_class = "later_decode_token_mismatch"
    else:
        failure_class = "none"

    print(f"\n== {args.label} vLLM forward check ==")
    print(f"prompt_token_ids_match: {same_prompt}")
    print(f"generated_token_ids_match: {same_tokens}")
    print(f"vllm_token_ids: {vllm_result['generated_token_ids']}")
    print(f"fastllm_token_ids: {fastllm_result['generated_token_ids']}")
    print(f"first_token_match: {first_token_match}")
    print(f"vllm_first_argmax_token_id: {v_argmax}")
    print(f"fastllm_logits_probe_argmax_token_id: {f_probe_argmax}")
    print(f"vllm_argmax_matches_fastllm_probe: {probe_argmax_match}")
    print(f"vllm_generation_matches_argmax: {v_generation_matches_argmax}")
    print(f"fastllm_production_matches_probe_argmax: {f_production_matches_probe}")
    print(f"target_backend_required: {target_backend_required}")
    print(f"target_backend_confirmed: {target_backend_confirmed}")
    print("production_trace_boundaries_found: "
          f"{fastllm_result.get('production_trace_boundaries_found', False)}")
    print(f"target_backend_trace_count: "
          f"{fastllm_result.get('target_backend_trace_count', 0)}")
    print("vllm_token_rank_in_fastllm_probe_topk: "
          f"{v_token_rank_in_fastllm_probe}")
    print("fastllm_production_rank_in_vllm_topk: "
          f"{f_production_rank_in_vllm}")
    print("first_token_logprob_diff: " + format_metric(first_token_logprob_diff))
    print(f"first_top{args.top_logprobs}_overlap: {overlap:.4f}")
    print("common_topk_max_logprob_diff: " + format_metric(max_diff))
    print(f"failure_class: {failure_class}")

    # 功能结论只使用双方正式生产生成路径。FastLLM的完整logits探针会
    # 改变其采样实现，因此TopK、探针argmax和logprob差只作为诊断项。
    functional_passed = (
        same_prompt and target_backend_ok and first_token_match and
        v_generation_matches_argmax)
    alignment_passed = (
        first_token_logprob_diff is not None and
        first_token_logprob_diff <= args.max_first_logprob_diff)
    passed = (functional_passed and alignment_passed
              if args.strict_alignment else functional_passed)
    print(f"functional_check: {'PASS' if functional_passed else 'FAIL'}")
    print("strict_alignment: " + (
        "PASS" if alignment_passed else
        ("FAIL" if args.strict_alignment else "WARN")))
    check_mode = "strict-alignment" if args.strict_alignment else "functional"
    print(f"check_mode: {check_mode}")
    result = "PASS" if passed else "FAIL"
    print(f"Summary: {result}")
    return passed, {
        "result": result,
        "label": args.label,
        "check_mode": check_mode,
        "prompt_token_ids_match": same_prompt,
        "generated_token_ids_match": same_tokens,
        "first_token_match": first_token_match,
        "vllm_first_argmax_token_id": v_argmax,
        "fastllm_logits_probe_argmax_token_id": f_probe_argmax,
        "vllm_argmax_matches_fastllm_probe": probe_argmax_match,
        "vllm_generation_matches_argmax": v_generation_matches_argmax,
        "fastllm_production_matches_probe_argmax": f_production_matches_probe,
        "target_backend_required": target_backend_required,
        "target_backend_confirmed": target_backend_confirmed,
        "target_backend_trace_pattern": args.target_trace_pattern,
        "target_backend_trace_count": fastllm_result.get(
            "target_backend_trace_count", 0),
        "production_trace_boundaries_found": fastllm_result.get(
            "production_trace_boundaries_found", False),
        "vllm_token_rank_in_fastllm_probe_topk": (
            v_token_rank_in_fastllm_probe),
        "fastllm_production_rank_in_vllm_topk": (
            f_production_rank_in_vllm),
        "failure_class": failure_class,
        "topk": args.top_logprobs,
        "topk_overlap": overlap,
        "minimum_topk_overlap": args.min_topk_overlap,
        "first_token_logprob_diff": first_token_logprob_diff,
        "maximum_first_logprob_diff": args.max_first_logprob_diff,
        "strict_alignment_result": (
            "PASS" if alignment_passed else
            ("FAIL" if args.strict_alignment else "WARN")),
        "common_topk_max_logprob_diff": max_diff,
        "vllm_first_top_logprobs": vllm_result["first_top_logprobs"],
        "fastllm_logits_probe_top_logprobs": (
            fastllm_result["logits_probe_top_logprobs"]),
        "vllm_generated_token_ids": vllm_result["generated_token_ids"],
        "fastllm_generated_token_ids": fastllm_result["generated_token_ids"],
    }


def write_summary_reports(result_dir, summary):
    """写出Forward Check的机器可读结果和单行Excel汇总。"""
    output = result_dir / "forward-check-summary"
    write_json(output.with_suffix(".json"), summary)
    target_backend = (
        summary["target_backend_confirmed"]
        if summary["target_backend_required"] else "n/a")

    lines = [
        f"# {summary['label']} Forward Check汇总", "",
        "| 状态 | 失败分类 | 目标后端确认 | Prompt一致 | 正式首Token一致 | "
        "FastLLM正式Token/探针argmax一致 | 探针TopK重合率 | "
        "首Token logprob差 | 严格对齐 |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
        f"| {summary['result']} | {summary['failure_class']} | "
        f"{target_backend} | {summary['prompt_token_ids_match']} | "
        f"{summary['first_token_match']} | "
        f"{summary['fastllm_production_matches_probe_argmax']} | "
        f"{summary['topk_overlap']:.3f} | "
        f"{format_metric(summary['first_token_logprob_diff'], 3)} | "
        f"{summary['strict_alignment_result']} |",
    ]
    output.with_suffix(".md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    with output.with_suffix(".csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow({
            key: (json.dumps(value, ensure_ascii=False)
                  if isinstance(value, (dict, list)) else value)
            for key, value in summary.items()
        })

    headers = [
        "状态", "标签", "模式", "失败分类", "Prompt一致", "生成Token一致",
        "正式首Token一致", "目标后端要求", "目标后端确认", "目标Trace次数",
        "生产Trace边界完整", "目标Trace模式", "vLLM首步argmax",
        "FastLLM logits探针argmax",
        "vLLM argmax/FastLLM探针一致", "vLLM生成/argmax一致",
        "FastLLM正式Token/探针argmax一致",
        "vLLM首Token在FastLLM探针TopK排名",
        "FastLLM正式首Token在vLLM TopK排名",
        "探针TopK", "探针TopK重合率", "最低重合率", "首Token logprob差",
        "首Token logprob阈值", "严格对齐结果", "共同TopK最大logprob差",
        "vLLM首步TopK", "FastLLM logits探针TopK", "vLLM生成Token",
        "FastLLM生成Token",
    ]
    row = [
        summary["result"], summary["label"], summary["check_mode"],
        summary["failure_class"],
        summary["prompt_token_ids_match"], summary["generated_token_ids_match"],
        summary["first_token_match"], summary["target_backend_required"],
        summary["target_backend_confirmed"],
        summary["target_backend_trace_count"],
        summary["production_trace_boundaries_found"],
        summary["target_backend_trace_pattern"],
        summary["vllm_first_argmax_token_id"],
        summary["fastllm_logits_probe_argmax_token_id"],
        summary["vllm_argmax_matches_fastllm_probe"],
        summary["vllm_generation_matches_argmax"],
        summary["fastllm_production_matches_probe_argmax"],
        summary["vllm_token_rank_in_fastllm_probe_topk"],
        summary["fastllm_production_rank_in_vllm_topk"],
        summary["topk"], summary["topk_overlap"],
        summary["minimum_topk_overlap"], summary["first_token_logprob_diff"],
        summary["maximum_first_logprob_diff"],
        summary["strict_alignment_result"],
        summary["common_topk_max_logprob_diff"],
        json.dumps(summary["vllm_first_top_logprobs"], ensure_ascii=False),
        json.dumps(
            summary["fastllm_logits_probe_top_logprobs"], ensure_ascii=False),
        json.dumps(summary["vllm_generated_token_ids"], ensure_ascii=False),
        json.dumps(summary["fastllm_generated_token_ids"], ensure_ascii=False),
    ]
    write_xlsx(output.with_suffix(".xlsx"), [("Forward Check", headers, [row])])
    for suffix in ("md", "csv", "json", "xlsx"):
        print(f"{suffix.upper()}: {output.with_suffix('.' + suffix)}")


def child_command(args, stage, output, python):
    return [
        python, str(Path(__file__).resolve()),
        "--stage", stage,
        "--output", str(output),
        "--model", args.model,
        "--tokens", str(args.tokens),
        "--top-logprobs", str(args.top_logprobs),
        "--case-index", str(args.case_index),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--flm-dtype", args.flm_dtype,
        "--flm-atype", args.flm_atype,
        "--flm-device", args.flm_device,
        "--label", args.label,
    ]


def production_trace(output):
    """截取FastLLM正式生成请求期间的原生日志。"""
    begin = output.find(FASTLLM_PRODUCTION_TRACE_BEGIN)
    if begin < 0:
        return "", False
    begin += len(FASTLLM_PRODUCTION_TRACE_BEGIN)
    end = output.find(FASTLLM_PRODUCTION_TRACE_END, begin)
    if end < 0:
        return "", False
    return output[begin:end], True


def orchestrate(args):
    result_dir = Path(args.result_dir or (REPO_DIR / "test" / "nvfp4" / "results"))
    result_dir.mkdir(parents=True, exist_ok=True)
    vllm_output = result_dir / "vllm.json"
    fastllm_output = result_dir / "fastllm.json"
    vllm_command = child_command(
        args, "vllm", vllm_output, args.vllm_python)
    print(f"SUBPROCESS COMMAND: {shlex.join(vllm_command)}", flush=True)
    subprocess.run(vllm_command, cwd=REPO_DIR, check=True)

    fastllm_command = child_command(
        args, "fastllm", fastllm_output, sys.executable)
    print(f"SUBPROCESS COMMAND: {shlex.join(fastllm_command)}", flush=True)
    process = subprocess.Popen(
        fastllm_command, cwd=REPO_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", bufsize=1)
    output_chunks = []
    for line in process.stdout:
        output_chunks.append(line)
        print(line, end="", flush=True)
    returncode = process.wait()
    fastllm_log = "".join(output_chunks)
    (result_dir / "fastllm-stage.log").write_text(
        fastllm_log, encoding="utf-8")
    if returncode != 0:
        raise subprocess.CalledProcessError(
            returncode, fastllm_command, output=fastllm_log)

    with open(vllm_output, encoding="utf-8") as handle:
        vllm_result = json.load(handle)
    with open(fastllm_output, encoding="utf-8") as handle:
        fastllm_result = json.load(handle)

    request_trace, boundaries_found = production_trace(fastllm_log)
    trace_count = (
        request_trace.count(args.target_trace_pattern)
        if args.target_trace_pattern and boundaries_found else 0)
    fastllm_result["target_backend_confirmed"] = (
        trace_count > 0 if args.target_trace_pattern else None)
    fastllm_result["target_backend_trace_count"] = trace_count
    fastllm_result["production_trace_boundaries_found"] = boundaries_found
    write_json(fastllm_output, fastllm_result)

    passed, summary = compare_results(args, vllm_result, fastllm_result)
    write_summary_reports(result_dir, summary)
    return passed


def main():
    args = parse_args()
    if args.stage != "orchestrate" and not args.output:
        raise ValueError("internal stage requires --output")
    if args.stage == "vllm":
        run_vllm(args)
        return 0
    if args.stage == "fastllm":
        run_fastllm(args)
        return 0
    return 0 if orchestrate(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
