// Apache-2.0. Native FastLLM adaptation of vLLM scaled_mm SM90 INT8 and
// SM120 FP8 kernels.  The public wrappers intentionally reject every layout
// except dynamic per-token activation scale + symmetric per-output-channel
// weight scale.
#include "fastllm-cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(FASTLLM_ENABLE_CUTLASS_W8A8)
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

namespace fastllm_w8a8_dense_detail {
using namespace cute;

template <typename T> __device__ float ToFloat(T value);
template <> __device__ float ToFloat(half value) { return __half2float(value); }
template <> __device__ float ToFloat(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename Input, typename Quant, int MaxValue>
__global__ void QuantizePerToken(const Input *input, Quant *quant,
                                 float *scales, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float local = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
        local = fmaxf(local, fabsf(ToFloat(input[(size_t)row * cols + col])));
    for (int delta = 16; delta > 0; delta >>= 1)
        local = fmaxf(local, __shfl_down_sync(0xffffffff, local, delta));
    __shared__ float warpMax[8];
    if ((threadIdx.x & 31) == 0) warpMax[threadIdx.x >> 5] = local;
    __syncthreads();
    if (threadIdx.x < 32) {
        local = threadIdx.x < 8 ? warpMax[threadIdx.x] : 0.0f;
        for (int delta = 16; delta > 0; delta >>= 1)
            local = fmaxf(local, __shfl_down_sync(0xffffffff, local, delta));
        if (threadIdx.x == 0) warpMax[0] = local;
    }
    __syncthreads();
    float scale = warpMax[0] == 0.0f ? 1.0f : warpMax[0] / (float)MaxValue;
    if (threadIdx.x == 0) scales[row] = scale;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float value = ToFloat(input[(size_t)row * cols + col]) / scale;
        if constexpr (std::is_same_v<Quant, int8_t>) {
            int q = max(-MaxValue, min(MaxValue, __float2int_rn(value)));
            quant[(size_t)row * cols + col] = (int8_t)q;
        } else {
            quant[(size_t)row * cols + col] = Quant(value);
        }
    }
}

template <typename Output, typename Accumulator>
__global__ void ApplyScales(const Accumulator *accumulator,
                            const float *tokenScales,
                            const float *channelScales,
                            const float *bias, Output *output,
                            int rows, int cols) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)rows * cols;
    if (index >= count) return;
    int row = index / cols, col = index % cols;
    float value = (float)accumulator[index] * tokenScales[row] * channelScales[col];
    if (bias != nullptr) value += bias[col];
    output[index] = Output(value);
}

template <typename Kernel>
struct EnableSm90Only : Kernel {
    template <typename... Args> CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
        Kernel::operator()(std::forward<Args>(args)...);
#elif defined(__CUDA_ARCH__)
        asm("trap;");
#endif
    }
};

template <typename Kernel>
struct EnableSm120Only : Kernel {
    template <typename... Args> CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && (__CUDA_ARCH__ < 1210)
        Kernel::operator()(std::forward<Args>(args)...);
#elif defined(__CUDA_ARCH__)
        asm("trap;");
#endif
    }
};

template <typename Element, typename Accumulator, typename Arch,
          typename Tile, typename Cluster, typename MainloopSchedule,
          typename EpilogueSchedule,
          typename EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto>
struct DenseKernel {
    using ElementA = Element;
    using ElementB = Element;
    using ElementD = Accumulator;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::RowMajor;
    static constexpr int Alignment = 128 / cutlass::sizeof_bits<Element>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<Accumulator>::value;
    using Operation = cutlass::epilogue::fusion::LinearCombination<
        Accumulator, Accumulator, void, Accumulator>;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch, cutlass::arch::OpClassTensorOp, Tile, Cluster,
        EpilogueTile,
        Accumulator, Accumulator, void, LayoutD, AlignmentD,
        Accumulator, LayoutD, AlignmentD, EpilogueSchedule, Operation>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, cutlass::arch::OpClassTensorOp,
        Element, LayoutA, Alignment, Element, LayoutB, Alignment,
        Accumulator, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;
    using Base = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, Mainloop, Epilogue>;
    using EnabledKernel = std::conditional_t<std::is_same_v<Arch, cutlass::arch::Sm90>,
        EnableSm90Only<Base>, EnableSm120Only<Base>>;
    struct GemmKernel : EnabledKernel {};
};

template <typename Definition>
bool Run(typename Definition::ElementA const *a,
         typename Definition::ElementB const *b,
         typename Definition::ElementD *d,
         int m, int n, int k, cudaStream_t stream) {
    using Kernel = typename Definition::GemmKernel;
    using Adapter = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using StrideA = typename Kernel::StrideA;
    using StrideB = typename Kernel::StrideB;
    using StrideD = typename Kernel::StrideD;
    StrideA strideA = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
    StrideB strideB = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
    StrideD strideD = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));
    typename Kernel::MainloopArguments mainloop{a, strideA, b, strideB};
    typename Kernel::EpilogueArguments epilogue{{}, d, strideD, d, strideD};
    cutlass::KernelHardwareInfo hw;
    hw.device_id = FastllmCudaGetDevice();
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);
    typename Kernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                    make_shape(m, n, k, 1), mainloop, epilogue, hw};
    Adapter gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    size_t workspaceBytes = Adapter::get_workspace_size(args);
    void *workspace = workspaceBytes ? FastllmCudaMalloc(workspaceBytes) : nullptr;
    if (workspaceBytes && workspace == nullptr) return false;
    cutlass::Status status = gemm.run(args, workspace, stream);
    if (workspace) FastllmCudaFree(workspace);
    return status == cutlass::Status::kSuccess && cudaGetLastError() == cudaSuccess;
}

template <typename Element, typename Accumulator>
bool Dispatch(int arch, Element const *a, Element const *b, Accumulator *d,
              int m, int n, int k, cudaStream_t stream) {
#if defined(FASTLLM_CUTLASS_W8A8_SM90)
    if constexpr (std::is_same_v<Element, int8_t>) {
        if (arch != 90) return false;
        if (m <= 32 && n < 8192) {
            using D = DenseKernel<Element, Accumulator, cutlass::arch::Sm90,
                Shape<_64,_64,_256>, Shape<_1,_8,_1>,
                cutlass::gemm::KernelTmaWarpSpecialized,
                cutlass::epilogue::TmaWarpSpecialized>;
            return Run<D>(a, b, d, m, n, k, stream);
        }
        if (m <= 32) {
            using D = DenseKernel<Element, Accumulator, cutlass::arch::Sm90,
                Shape<_64,_128,_256>, Shape<_1,_4,_1>,
                cutlass::gemm::KernelTmaWarpSpecialized,
                cutlass::epilogue::TmaWarpSpecialized>;
            return Run<D>(a, b, d, m, n, k, stream);
        }
        if (m <= 64) {
            using D = DenseKernel<Element, Accumulator, cutlass::arch::Sm90,
                Shape<_64,_64,_256>, Shape<_1,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecialized,
                cutlass::epilogue::TmaWarpSpecialized>;
            return Run<D>(a, b, d, m, n, k, stream);
        }
        if (m <= 128) {
            using D = DenseKernel<Element, Accumulator, cutlass::arch::Sm90,
                Shape<_64,_128,_128>, Shape<_2,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                cutlass::epilogue::TmaWarpSpecialized>;
            return Run<D>(a, b, d, m, n, k, stream);
        }
        using D = DenseKernel<Element, Accumulator, cutlass::arch::Sm90,
            Shape<_128,_128,_128>, Shape<_2,_1,_1>,
            cutlass::gemm::KernelTmaWarpSpecializedPingpong,
            cutlass::epilogue::TmaWarpSpecialized>;
        return Run<D>(a, b, d, m, n, k, stream);
    }
#endif
#if defined(FASTLLM_CUTLASS_W8A8_SM120)
    if constexpr (std::is_same_v<Element, cutlass::float_e4m3_t>) {
        if (arch != 120) return false;
        if (m <= 16) {
            using D = DenseKernel<Element, Accumulator, cutlass::arch::Sm120,
                Shape<_16,_64,_128>, Shape<_1,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                cutlass::epilogue::collective::EpilogueScheduleAuto,
                Shape<_16,_32>>;
            return Run<D>(a, b, d, m, n, k, stream);
        }
        if (m <= 32) {
            using D = DenseKernel<Element, Accumulator, cutlass::arch::Sm120,
                Shape<_32,_64,_128>, Shape<_1,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                cutlass::epilogue::collective::EpilogueScheduleAuto,
                Shape<_32,_32>>;
            return Run<D>(a, b, d, m, n, k, stream);
        }
        if (m <= 256) {
            using D = DenseKernel<Element, Accumulator, cutlass::arch::Sm120,
                Shape<_64,_64,_128>, Shape<_1,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                cutlass::epilogue::collective::EpilogueScheduleAuto>;
            return Run<D>(a, b, d, m, n, k, stream);
        }
        using D = DenseKernel<Element, Accumulator, cutlass::arch::Sm120,
            Shape<_128,_128,_128>, Shape<_1,_1,_1>,
            cutlass::gemm::collective::KernelScheduleAuto,
            cutlass::epilogue::collective::EpilogueScheduleAuto>;
        return Run<D>(a, b, d, m, n, k, stream);
    }
#endif
    return false;
}

template <typename Input, typename Quant, typename Accumulator, int MaxValue>
bool Execute(const fastllm::Data &input, fastllm::Data &weight,
             const fastllm::Data &bias, fastllm::Data &output,
             int m, int k, int n, int arch) {
    Quant *quant = (Quant*)FastllmCudaMalloc((size_t)m * k * sizeof(Quant));
    float *tokenScales = (float*)FastllmCudaMalloc((size_t)m * sizeof(float));
    Accumulator *accumulator = (Accumulator*)FastllmCudaMalloc(
        (size_t)m * n * sizeof(Accumulator));
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight, bias, n);
    if (!quant || !tokenScales || !accumulator || weight.extraCudaData.size() < 2 ||
        weight.extraCudaData[0] == nullptr || weight.extraCudaData[1] == nullptr) {
        FastllmCudaFree(quant); FastllmCudaFree(tokenScales);
        FastllmCudaFree(accumulator);
        return false;
    }
    const float *channelScales = (const float*)weight.extraCudaData[0];
    const float *biasData = (const float*)weight.extraCudaData[1];
    cudaStream_t stream = 0;
    QuantizePerToken<Input, Quant, MaxValue><<<m, 256, 0, stream>>>(
        (const Input*)input.cudaData, quant, tokenScales, m, k);
    bool ok = cudaGetLastError() == cudaSuccess &&
              Dispatch<Quant, Accumulator>(arch, quant,
                  (const Quant*)weight.cudaData, accumulator, m, n, k, stream);
    if (ok) {
        int threads = 256;
        int blocks = std::min<size_t>(4096, ((size_t)m * n + threads - 1) / threads);
        if (input.dataType == fastllm::DataType::FLOAT16)
            ApplyScales<<<blocks, threads, 0, stream>>>(accumulator, tokenScales,
                channelScales, biasData, (half*)output.cudaData, m, n);
        else
            ApplyScales<<<blocks, threads, 0, stream>>>(accumulator, tokenScales,
                channelScales, biasData, (__nv_bfloat16*)output.cudaData, m, n);
        ok = cudaGetLastError() == cudaSuccess;
    }
    FastllmCudaFree(quant); FastllmCudaFree(tokenScales);
    FastllmCudaFree(accumulator);
    return ok;
}
} // namespace fastllm_w8a8_dense_detail
#endif

bool FastllmCudaCutlassLinearInt8W8A8Sm90(
    const fastllm::Data &input, fastllm::Data &weight,
    const fastllm::Data &bias, fastllm::Data &output, int m, int k, int n) {
#if defined(FASTLLM_ENABLE_CUTLASS_W8A8) && defined(FASTLLM_CUTLASS_W8A8_SM90)
    using namespace fastllm_w8a8_dense_detail;
    if (FastllmCudaRuntimeArch() != 90 || m <= 0 || k <= 0 || n <= 0 ||
        k % 16 || n % 8 || input.cudaData == nullptr || weight.cudaData == nullptr ||
        output.cudaData == nullptr || weight.dataType != fastllm::DataType::INT8_W8A8 ||
        weight.dims != std::vector<int>({n, k}) || weight.perChannelAxis != 0 ||
        !weight.mins.empty() || !weight.zeros.empty() || weight.scales.size() != (size_t)n ||
        (input.dataType != fastllm::DataType::FLOAT16 && input.dataType != fastllm::DataType::BFLOAT16) ||
        output.dataType != input.dataType ||
        (!bias.dims.empty() && (bias.dataType != fastllm::DataType::FLOAT32 ||
                               bias.cudaData == nullptr || bias.Count(0) != n))) return false;
    if (input.dataType == fastllm::DataType::FLOAT16)
        return Execute<half, int8_t, int32_t, 127>(input, weight, bias, output, m, k, n, 90);
    return Execute<__nv_bfloat16, int8_t, int32_t, 127>(input, weight, bias, output, m, k, n, 90);
#else
    (void)input; (void)weight; (void)bias; (void)output; (void)m; (void)k; (void)n;
    return false;
#endif
}

bool FastllmCudaCutlassLinearFp8W8A8Sm120(
    const fastllm::Data &input, fastllm::Data &weight,
    const fastllm::Data &bias, fastllm::Data &output, int m, int k, int n) {
#if defined(FASTLLM_ENABLE_CUTLASS_W8A8) && defined(FASTLLM_CUTLASS_W8A8_SM120)
    using namespace fastllm_w8a8_dense_detail;
    if (FastllmCudaRuntimeArch() != 120 || m <= 0 || k <= 0 || n <= 0 ||
        k % 16 || n % 8 || input.cudaData == nullptr || weight.cudaData == nullptr ||
        output.cudaData == nullptr || weight.dataType != fastllm::DataType::FP8_E4M3 ||
        weight.dims != std::vector<int>({n, k}) || weight.blockK != 1 ||
        weight.blockM != k || weight.scales.size() != (size_t)n ||
        (input.dataType != fastllm::DataType::FLOAT16 && input.dataType != fastllm::DataType::BFLOAT16) ||
        output.dataType != input.dataType ||
        (!bias.dims.empty() && (bias.dataType != fastllm::DataType::FLOAT32 ||
                               bias.cudaData == nullptr || bias.Count(0) != n))) return false;
    if (input.dataType == fastllm::DataType::FLOAT16)
        return Execute<half, cutlass::float_e4m3_t, float, 448>(input, weight, bias, output, m, k, n, 120);
    return Execute<__nv_bfloat16, cutlass::float_e4m3_t, float, 448>(input, weight, bias, output, m, k, n, 120);
#else
    (void)input; (void)weight; (void)bias; (void)output; (void)m; (void)k; (void)n;
    return false;
#endif
}
