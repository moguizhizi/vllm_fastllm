#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
log_dir=${NVFP4_LOG_DIR:-"${repo_dir}/test/nvfp4/logs/$(date -u +%Y%m%dT%H%M%SZ)"}
suite=${1:-ops}
model_path=${NVFP4_MODEL:-}
gpu_profile=${NVFP4_GPU_PROFILE:-sm100}
optest=${NVFP4_OPTEST:-"${repo_dir}/optest"}
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

run_ops() {
    if [[ "${gpu_profile}" == sm100 || "${gpu_profile}" == sm120 ]]; then
        run_logged op_dense_w4a4 env \
            FASTLLM_CUDA_NVFP4_TRACE=1 \
            FASTLLM_CUDA_NVFP4_W4A4=1 \
            FASTLLM_CUDA_NVFP4_W4A4_MIN_ROWS=32 \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=32 --param in=1024 --param out=1024 \
            --param input_type=bf16 --warmup 10 --iters 100 \
            --atol 0.20 --rtol 0.20
        run_logged op_swiglu_fp4_quant "${optest}" \
            --op nvfp4_swiglu_quant --device cuda:0 \
            --param rows=32 --param hidden=1024 --param input_type=bf16 \
            --warmup 10 --iters 100 --atol 0.20 --rtol 0.20
        if [[ "${gpu_profile}" == sm100 ]]; then
            run_logged op_grouped_moe_w4a4_check env \
                FASTLLM_CUDA_NVFP4_TRACE=1 \
                FASTLLM_CUDA_MOE_NVFP4_W4A4=1 \
                FASTLLM_CUDA_MOE_NVFP4_W4A4_MIN_BATCH=16 \
                "${optest}" --op mergemoe_fp8 --device cuda:0 \
                --param weight_type=nvfp4 --param path=check_nvfp4 \
                --param batch=32 --param topk=2 --param experts=8 \
                --param hidden=256 --param inter=256 --param input_type=bf16 \
                --warmup 0 --iters 1
            run_logged op_grouped_moe_w4a4_perf env \
                FASTLLM_CUDA_NVFP4_TRACE=1 \
                FASTLLM_CUDA_MOE_NVFP4_W4A4=1 \
                FASTLLM_CUDA_MOE_NVFP4_W4A4_MIN_BATCH=16 \
                "${optest}" --op mergemoe_fp8 --device cuda:0 \
                --param weight_type=nvfp4 --param path=operator \
                --param batch=32 --param topk=2 --param experts=8 \
                --param hidden=256 --param inter=256 --param input_type=bf16 \
                --warmup 10 --iters 100
        fi
    else
        run_logged op_dense_marlin_w4a16 env \
            FASTLLM_CUDA_NVFP4_TRACE=1 \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=8 --param in=1024 --param out=1024 \
            --param input_type=fp16 --warmup 10 --iters 100 \
            --atol 0.05 --rtol 0.05
    fi
}

require_model() {
    if [[ -z "${model_path}" ]]; then
        echo "NVFP4_MODEL must point to a downloaded NVFP4 Hugging Face model." >&2
        exit 2
    fi
}

run_forward() {
    require_model
    run_logged forward_check env FASTLLM_CUDA_NVFP4_TRACE=1 \
        python test/basic/forward_check.py --model "${model_path}" \
        --tokens 8 --hf_device cuda --flm_dtype auto \
        --flm_atype bfloat16 --flm_device cuda
}

run_model() {
    require_model
    run_logged model_prefill env FASTLLM_CUDA_NVFP4_TRACE=1 \
        python test/benchmark/prefill.py "${model_path}" \
        --dtype auto --atype bfloat16 --device cuda --moe_device cuda \
        --prompt-repeat 256 --max-tokens 16
    run_logged model_decode env FASTLLM_CUDA_NVFP4_TRACE=1 \
        python test/benchmark/decode.py "${model_path}" \
        --dtype auto --atype bfloat16 --device cuda --moe_device cuda \
        --batch-size 32 --max_batch 32 --prefill-length 512 --max-tokens 64
}

case "${suite}" in
    ops) run_ops ;;
    forward) run_forward ;;
    model) run_model ;;
    all) run_ops; run_forward; run_model ;;
    *) echo "usage: $0 {ops|forward|model|all}; NVFP4_GPU_PROFILE=sm100|sm120|preblackwell" >&2; exit 2 ;;
esac

printf 'Logs: %s\n' "${log_dir}"
