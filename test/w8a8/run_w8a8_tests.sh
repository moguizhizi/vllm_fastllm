#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
log_dir=${W8A8_LOG_DIR:-"${repo_dir}/test/w8a8/logs/$(date -u +%Y%m%dT%H%M%SZ)"}
optest=${W8A8_OPTEST:-"${repo_dir}/optest"}
model=${W8A8_MODEL:-}
suite=${1:-ops}
mkdir -p "${log_dir}"

run_logged() {
    local name=$1
    shift
    local log_file="${log_dir}/${name}.log"
    {
        printf 'UTC: %s\n' "$(date -u +%FT%TZ)"
        printf 'PWD: %q\n' "${repo_dir}"
        printf 'COMMAND:'
        printf ' %q' "$@"
        printf '\n\n'
    } | tee "${log_file}"
    (cd "${repo_dir}" && "$@") 2>&1 | tee -a "${log_file}"
}

detect_arch() {
    local cc major minor
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed -n '1p')
    major=${cc%%.*}; minor=${cc#*.}
    printf '%d\n' "$((10#${major} * 10 + 10#${minor}))"
}

run_ops_functional() {
    local arch
    arch=$(detect_arch)
    if [[ "${arch}" == 90 ]]; then
        run_logged sm90_int8_w8a8_function \
            "${optest}" --op linear_int8_w8a8 --device cuda:0 \
            --param batch=17 --param in=4096 --param out=4096 \
            --param input_type=bf16 --param has_bias=1 --param check=1 \
            --warmup 0 --iters 1
        run_logged sm90_fp8_blockwise_function env \
            FASTLLM_CUDA_CUTLASS_LINEAR_FP8=1 \
            FASTLLM_CUDA_CUTLASS_LINEAR_FP8_MIN_BATCH=1 \
            "${optest}" --op linear_fp8_block128 --device cuda:0 \
            --param batch=17 --param in=4096 --param out=4096 \
            --param block=128 --param weight_layout=separate \
            --param input_type=bf16 --param check=1 --warmup 0 --iters 1
        run_logged sm90_fp8_grouped_moe_function env \
            FASTLLM_CUDA_MOE_GROUPED_INDEXED=1 \
            FASTLLM_CUDA_MOE_GROUPED_INDEXED_MIN_BATCH=1 \
            FASTLLM_CUDA_MOE_CUTLASS_FP8_SM90=1 \
            "${optest}" --op mergemoe_fp8 --device cuda:0 \
            --param batch=128 --param hidden=2048 --param inter=768 \
            --param experts=16 --param topk=4 --param block=128 \
            --param input_type=bf16 --param weight_type=fp8 \
            --param scale_layout=perchannel \
            --param path=check_fp8 --warmup 0 --iters 1
    elif [[ "${arch}" == 120 ]]; then
        run_logged sm120_fp8_standard_function \
            "${optest}" --op linear_fp8_block128 --device cuda:0 \
            --param batch=17 --param in=4096 --param out=4096 \
            --param weight_layout=perchannel --param input_type=bf16 \
            --param check=4 --warmup 0 --iters 1
        run_logged sm120_fp8_blockwise_fallback_guard env \
            FASTLLM_CUDA_CUTLASS_LINEAR_FP8=1 \
            "${optest}" --op linear_fp8_block128 --device cuda:0 \
            --param batch=17 --param in=4096 --param out=4096 \
            --param block=128 --param weight_layout=separate \
            --param input_type=bf16 --param check=1 --warmup 0 --iters 1
    else
        echo "unsupported test GPU: SM${arch}; expected SM90 or SM120" >&2
        exit 2
    fi
}

run_ops_performance() {
    local arch
    arch=$(detect_arch)
    if [[ "${arch}" == 90 ]]; then
        for batch in 1 16 64 256 1024; do
            run_logged "sm90_int8_w8a8_m${batch}_perf" \
                "${optest}" --op linear_int8_w8a8 --device cuda:0 \
                --param batch="${batch}" --param in=4096 --param out=4096 \
                --param input_type=bf16 --param has_bias=1 \
                --warmup 20 --iters 200
            run_logged "sm90_fp8_blockwise_m${batch}_perf" env \
                FASTLLM_CUDA_CUTLASS_LINEAR_FP8=1 \
                FASTLLM_CUDA_CUTLASS_LINEAR_FP8_MIN_BATCH=1 \
                "${optest}" --op linear_fp8_block128 --device cuda:0 \
                --param batch="${batch}" --param in=4096 --param out=4096 \
                --param block=128 --param weight_layout=separate \
                --param input_type=bf16 --warmup 20 --iters 200
        done
        run_logged sm90_fp8_grouped_moe_perf env \
            FASTLLM_CUDA_MOE_GROUPED_INDEXED=1 \
            FASTLLM_CUDA_MOE_GROUPED_INDEXED_MIN_BATCH=1 \
            FASTLLM_CUDA_MOE_CUTLASS_FP8_SM90=1 \
            "${optest}" --op mergemoe_fp8 --device cuda:0 \
            --param batch=256 --param hidden=2048 --param inter=768 \
            --param experts=16 --param topk=4 --param block=128 \
            --param input_type=bf16 --param weight_type=fp8 \
            --param scale_layout=perchannel \
            --param path=operator --warmup 20 --iters 200
        run_logged sm90_fp8_grouped_moe_fallback_perf env \
            FASTLLM_CUDA_MOE_GROUPED_INDEXED=1 \
            FASTLLM_CUDA_MOE_GROUPED_INDEXED_MIN_BATCH=1 \
            FASTLLM_CUDA_MOE_CUTLASS_FP8_SM90=0 \
            "${optest}" --op mergemoe_fp8 --device cuda:0 \
            --param batch=256 --param hidden=2048 --param inter=768 \
            --param experts=16 --param topk=4 --param block=128 \
            --param input_type=bf16 --param weight_type=fp8 \
            --param scale_layout=perchannel \
            --param path=operator --warmup 20 --iters 200
    elif [[ "${arch}" == 120 ]]; then
        for batch in 1 16 64 256 1024; do
            run_logged "sm120_fp8_standard_m${batch}_perf" \
                "${optest}" --op linear_fp8_block128 --device cuda:0 \
                --param batch="${batch}" --param in=4096 --param out=4096 \
                --param weight_layout=perchannel --param input_type=bf16 \
                --warmup 20 --iters 200
            run_logged "sm120_fp8_blockwise_m${batch}_perf" env \
                FASTLLM_CUDA_CUTLASS_LINEAR_FP8=1 \
                FASTLLM_CUDA_CUTLASS_LINEAR_FP8_MIN_BATCH=1 \
                "${optest}" --op linear_fp8_block128 --device cuda:0 \
                --param batch="${batch}" --param in=4096 --param out=4096 \
                --param block=128 --param weight_layout=separate \
                --param input_type=bf16 --warmup 20 --iters 200
        done
    fi
}

require_model() {
    [[ -n "${model}" ]] || { echo "W8A8_MODEL is required" >&2; exit 2; }
}

run_forward() {
    require_model
    run_logged forward_check env FASTLLM_CUDA_CUTLASS_LINEAR_FP8=1 \
        FASTLLM_CUDA_MOE_GROUPED_INDEXED=1 \
        python test/basic/forward_check.py --model "${model}" --tokens 8 \
        --hf_device cuda --flm_dtype auto --flm_atype bfloat16 --flm_device cuda
}

run_model_performance() {
    require_model
    run_logged model_prefill env FASTLLM_CUDA_CUTLASS_LINEAR_FP8=1 \
        FASTLLM_CUDA_MOE_GROUPED_INDEXED=1 \
        python test/benchmark/prefill.py "${model}" --dtype auto \
        --atype bfloat16 --device cuda --moe_device cuda \
        --prompt-repeat 256 --max-tokens 16
    run_logged model_decode env FASTLLM_CUDA_CUTLASS_LINEAR_FP8=1 \
        FASTLLM_CUDA_MOE_GROUPED_INDEXED=1 \
        python test/benchmark/decode.py "${model}" --dtype auto \
        --atype bfloat16 --device cuda --moe_device cuda \
        --batch-size 32 --max_batch 32 --prefill-length 512 --max-tokens 64
}

case "${suite}" in
    ops-functional) run_ops_functional ;;
    ops-performance) run_ops_performance ;;
    ops) run_ops_functional; run_ops_performance ;;
    forward) run_forward ;;
    model-performance) run_model_performance ;;
    all) run_ops_functional; run_ops_performance; run_forward; run_model_performance ;;
    *) echo "usage: $0 {ops-functional|ops-performance|ops|forward|model-performance|all}" >&2; exit 2 ;;
esac

printf 'Logs: %s\n' "${log_dir}"
