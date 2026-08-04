#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
log_dir=${NVFP4_LOG_DIR:-"${repo_dir}/test/nvfp4/logs/$(date -u +%Y%m%dT%H%M%SZ)"}
suite=${1:-ops}
model_path=${NVFP4_MODEL:-}
gpu_profile=${NVFP4_GPU_PROFILE:-auto}
cuda_arch=${NVFP4_CUDA_ARCH:-auto}
optest=${NVFP4_OPTEST:-"${repo_dir}/optest"}
nvcc_path=${NVFP4_NVCC:-$(command -v nvcc || true)}
mkdir -p "${log_dir}"

resolve_gpu_profile() {
    local compute_cap major minor detected_arch
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed -n '1p')
    major=${compute_cap%%.*}
    minor=${compute_cap#*.}
    if [[ ! "${major}" =~ ^[0-9]+$ || ! "${minor}" =~ ^[0-9]+$ ]]; then
        echo "cannot detect GPU compute capability: ${compute_cap}" >&2
        exit 2
    fi
    detected_arch=$((10#${major} * 10 + 10#${minor}))
    if [[ "${gpu_profile}" == auto ]]; then
        if ((detected_arch >= 120 && detected_arch < 130)); then
            gpu_profile=sm120
        elif ((detected_arch >= 100 && detected_arch < 120)); then
            gpu_profile=sm100
        else
            gpu_profile=preblackwell
        fi
    fi
    if [[ "${cuda_arch}" == auto ]]; then
        cuda_arch=${detected_arch}
    fi
    printf 'GPU profile: %s; CUDA_ARCH: %s; compute capability: %s\n' \
        "${gpu_profile}" "${cuda_arch}" "${compute_cap}"
}

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

run_build() {
    local -a install_args
    if [[ -z "${nvcc_path}" ]]; then
        echo "nvcc not found; set NVFP4_NVCC=/path/to/nvcc" >&2
        exit 2
    fi
    run_logged environment bash -lc \
        'git rev-parse HEAD; git status --short --branch; nvidia-smi'
    run_logged nvcc_version "${nvcc_path}" --version
    install_args=(-DUSE_CUDA=ON -DUNIT_TEST=ON
        "-DCMAKE_CUDA_COMPILER=${nvcc_path}"
        "-DCUDA_ARCH=${cuda_arch}")
    run_logged install bash "${repo_dir}/install.sh" "${install_args[@]}"
}

run_ops_functional() {
    if [[ "${gpu_profile}" == sm100 || "${gpu_profile}" == sm120 ]]; then
        run_logged op_dense_w4a4_decode_padding_bias env \
            FASTLLM_CUDA_NVFP4_TRACE=1 \
            FASTLLM_CUDA_NVFP4_W4A4=1 \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=1 --param in=1008 --param out=1000 \
            --param bias=1 --param input_type=bf16 \
            --warmup 0 --iters 1 --atol 0.20 --rtol 0.20
        run_logged op_swiglu_fp4_quant "${optest}" \
            --op nvfp4_swiglu_quant --device cuda:0 \
            --param rows=32 --param hidden=1024 --param input_type=bf16 \
            --warmup 0 --iters 1 --atol 0.20 --rtol 0.20
        run_logged op_swiglu_fp4_quant_fp16 "${optest}" \
            --op nvfp4_swiglu_quant --device cuda:0 \
            --param rows=32 --param hidden=1024 --param input_type=fp16 \
            --warmup 0 --iters 1 --atol 0.20 --rtol 0.20
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
        fi
    else
        run_logged op_dense_marlin_w4a16_check env \
            FASTLLM_CUDA_NVFP4_TRACE=1 \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=8 --param in=1024 --param out=1024 \
            --param bias=0 --param input_type=fp16 --warmup 0 --iters 1 \
            --atol 0.05 --rtol 0.05
    fi
}

run_ops_performance() {
    if [[ "${gpu_profile}" == sm100 || "${gpu_profile}" == sm120 ]]; then
        run_logged op_dense_w4a4_decode_perf env \
            FASTLLM_CUDA_NVFP4_W4A4=1 \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=1 --param in=1024 --param out=1024 \
            --param bias=0 --param input_type=bf16 --warmup 20 --iters 200 \
            --atol 0.20 --rtol 0.20
        run_logged op_dense_w4a4_prefill_perf env \
            FASTLLM_CUDA_NVFP4_W4A4=1 \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=32 --param in=1024 --param out=1024 \
            --param bias=0 --param input_type=bf16 --warmup 20 --iters 200 \
            --atol 0.20 --rtol 0.20
        run_logged op_swiglu_fp4_quant_perf "${optest}" \
            --op nvfp4_swiglu_quant --device cuda:0 \
            --param rows=32 --param hidden=1024 --param input_type=bf16 \
            --warmup 20 --iters 200 --atol 0.20 --rtol 0.20
        if [[ "${gpu_profile}" == sm100 ]]; then
            run_logged op_grouped_moe_w4a4_perf env \
                FASTLLM_CUDA_MOE_NVFP4_W4A4=1 \
                FASTLLM_CUDA_MOE_NVFP4_W4A4_MIN_BATCH=16 \
                "${optest}" --op mergemoe_fp8 --device cuda:0 \
                --param weight_type=nvfp4 --param path=operator \
                --param batch=32 --param topk=2 --param experts=8 \
                --param hidden=256 --param inter=256 --param input_type=bf16 \
                --warmup 20 --iters 200
        fi
    else
        run_logged op_dense_marlin_w4a16_perf env \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=8 --param in=1024 --param out=1024 \
            --param bias=0 --param input_type=fp16 --warmup 20 --iters 200 \
            --atol 0.05 --rtol 0.05
    fi
}

run_ops() {
    run_ops_functional
    run_ops_performance
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

resolve_gpu_profile

case "${suite}" in
    build) run_build ;;
    ops-functional) run_ops_functional ;;
    ops-performance) run_ops_performance ;;
    ops) run_ops ;;
    forward) run_forward ;;
    model-performance|model) run_model ;;
    all) run_build; run_ops; run_forward; run_model ;;
    *) echo "usage: $0 {build|ops-functional|ops-performance|ops|forward|model-performance|all}; NVFP4_GPU_PROFILE=auto|sm100|sm120|preblackwell" >&2; exit 2 ;;
esac

printf 'Logs: %s\n' "${log_dir}"
