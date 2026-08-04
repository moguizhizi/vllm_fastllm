#!/usr/bin/env python3
"""Compare one NVFP4 FastLLM request with vLLM in isolated processes."""

import argparse
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
    parser.add_argument("--min-topk-overlap", type=float, default=0.5)
    parser.add_argument("--max-logprob-diff", type=float, default=1.0)
    parser.add_argument("--result-dir", default="")
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
    write_json(args.output, {
        "backend": "vllm",
        "prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=True),
        "generated_token_ids": strip_trailing_eos(tokenizer, completion.token_ids),
        "generated_text": completion.text,
        "first_top_logprobs": first_logprobs,
    })


def generate_fastllm_token_ids(model, tokenizer, prompt, max_tokens):
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
    from ftllm import llm

    llm.set_device_map(args.flm_device)
    model = llm.model(args.model, dtype=args.flm_dtype)
    model.set_atype(args.flm_atype)
    tokenizer = model.hf_tokenizer
    if tokenizer is None:
        raise RuntimeError("FastLLM did not load the Hugging Face tokenizer")
    prompt = render_prompt(tokenizer, get_messages(args.case_index))
    model.direct_query = True
    logits = model.response_logits(prompt, tokenizer=tokenizer)
    input_ids, generated_ids = generate_fastllm_token_ids(
        model, tokenizer, prompt, args.tokens)
    write_json(args.output, {
        "backend": "fastllm",
        "prompt_token_ids": input_ids,
        "generated_token_ids": strip_trailing_eos(tokenizer, generated_ids),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "first_top_logprobs": top_logprobs(logits, args.top_logprobs),
    })
    model.release_memory()


def compare_results(args, vllm_result, fastllm_result):
    same_prompt = vllm_result["prompt_token_ids"] == fastllm_result["prompt_token_ids"]
    same_tokens = vllm_result["generated_token_ids"] == fastllm_result["generated_token_ids"]
    v_probs = {int(key): value for key, value in vllm_result["first_top_logprobs"].items()}
    f_probs = {int(key): value for key, value in fastllm_result["first_top_logprobs"].items()}
    common = sorted(set(v_probs) & set(f_probs))
    denominator = max(1, min(args.top_logprobs, len(v_probs), len(f_probs)))
    overlap = len(common) / denominator
    max_diff = max((abs(v_probs[token] - f_probs[token]) for token in common), default=float("inf"))
    first_vllm = vllm_result["generated_token_ids"][:1]
    first_fastllm = fastllm_result["generated_token_ids"][:1]

    print("\n== NVFP4 vLLM forward check ==")
    print(f"prompt_token_ids_match: {same_prompt}")
    print(f"generated_token_ids_match: {same_tokens}")
    print(f"vllm_token_ids: {vllm_result['generated_token_ids']}")
    print(f"fastllm_token_ids: {fastllm_result['generated_token_ids']}")
    print(f"first_token_match: {first_vllm == first_fastllm}")
    print(f"first_top{args.top_logprobs}_overlap: {overlap:.4f}")
    print(f"common_topk_max_logprob_diff: {max_diff:.6f}")

    passed = (same_prompt and same_tokens and first_vllm == first_fastllm and
              overlap >= args.min_topk_overlap and max_diff <= args.max_logprob_diff)
    print(f"Summary: {'PASS' if passed else 'FAIL'}")
    return passed


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
    ]


def orchestrate(args):
    result_dir = Path(args.result_dir or (REPO_DIR / "test" / "nvfp4" / "results"))
    result_dir.mkdir(parents=True, exist_ok=True)
    vllm_output = result_dir / "vllm.json"
    fastllm_output = result_dir / "fastllm.json"
    commands = [
        child_command(args, "vllm", vllm_output, args.vllm_python),
        child_command(args, "fastllm", fastllm_output, sys.executable),
    ]
    for command in commands:
        print(f"SUBPROCESS COMMAND: {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_DIR, check=True)
    with open(vllm_output, encoding="utf-8") as handle:
        vllm_result = json.load(handle)
    with open(fastllm_output, encoding="utf-8") as handle:
        fastllm_result = json.load(handle)
    return compare_results(args, vllm_result, fastllm_result)


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
