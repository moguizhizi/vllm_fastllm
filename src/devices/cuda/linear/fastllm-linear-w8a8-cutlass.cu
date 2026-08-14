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
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
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

using BackendState = FastllmCudaFp8W8A8BackendState;

/**
 * SM120 W8A8执行阶段按GPU复用的临时显存。
 *
 * FP8权重已经是CUTLASS直接消费的[N,K]格式，不需要像NVFP4一样保留
 * 第二份重排权重。本结构只缓存动态量化激活、per-token scale、FP32
 * 累加输出和可能存在的CUTLASS workspace；各缓冲区只扩容、不缩小。
 */
struct ExecutionScratch {
    void *quantized = nullptr;
    float *tokenScales = nullptr;
    void *accumulator = nullptr;
    void *workspace = nullptr;
    size_t quantizedBytes = 0;
    size_t tokenScaleBytes = 0;
    size_t accumulatorBytes = 0;
    size_t workspaceBytes = 0;
};

std::mutex scratchMutex;
std::map<int, ExecutionScratch> executionScratch;
std::mutex backendMutex;
std::map<std::pair<const fastllm::Data *, int>, BackendState> backendStates;

static bool Enabled() {
    const char *value = std::getenv("FASTLLM_CUDA_W8A8");
    return value == nullptr || value[0] == '\0' || value[0] != '0';
}

static bool StrictEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_W8A8_STRICT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool TraceEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_W8A8_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool ForceGemmFailureForTest() {
    const char *value = std::getenv(
        "FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[noreturn]] static void ThrowFixedBackend(const char *reason) {
    throw std::runtime_error(
        std::string("fixed SM120 FP8 W8A8 CUTLASS backend failed: ") + reason);
}

static void Trace(const char *path, const char *reason,
                  int m, int n, int k) {
    if (!TraceEnabled()) return;
    std::fprintf(stderr,
                 "[fastllm][w8a8] path=%s reason=%s m=%d n=%d k=%d sm=120\n",
                 path, reason, m, n, k);
}

static BackendState GetBackendState(const fastllm::Data &weight, int device) {
    std::lock_guard<std::mutex> guard(backendMutex);
    auto it = backendStates.find({&weight, device});
    return it == backendStates.end() ? BackendState::Uninitialized : it->second;
}

static void SetBackendState(const fastllm::Data &weight, int device,
                            BackendState state) {
    std::lock_guard<std::mutex> guard(backendMutex);
    backendStates[{&weight, device}] = state;
}

static void ReleaseScratch(ExecutionScratch &scratch) {
    if (scratch.quantized != nullptr) FastllmCudaFree(scratch.quantized);
    if (scratch.tokenScales != nullptr) FastllmCudaFree(scratch.tokenScales);
    if (scratch.accumulator != nullptr) FastllmCudaFree(scratch.accumulator);
    if (scratch.workspace != nullptr) FastllmCudaFree(scratch.workspace);
    scratch = ExecutionScratch();
}

/**
 * 获取或扩容当前GPU上的W8A8执行临时区。
 *
 * 首次GEMM会完成分配，后续正式推理直接复用稳定地址；CUDA Graph捕获
 * 期间禁止扩容。申请新空间失败时保留原临时区，避免破坏其他调用。
 *
 * @param quantizedBytes   FP8动态量化激活所需字节数。
 * @param tokenScaleBytes  per-token FP32 scale所需字节数。
 * @param accumulatorBytes FP32/INT32累加输出所需字节数。
 * @return 容量足够时返回当前GPU临时区，否则返回nullptr。
 */
static ExecutionScratch *GetExecutionScratch(size_t quantizedBytes,
                                             size_t tokenScaleBytes,
                                             size_t accumulatorBytes) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return nullptr;
    std::lock_guard<std::mutex> guard(scratchMutex);
    ExecutionScratch &scratch = executionScratch[device];
    if (scratch.quantized != nullptr && scratch.tokenScales != nullptr &&
        scratch.accumulator != nullptr &&
        scratch.quantizedBytes >= quantizedBytes &&
        scratch.tokenScaleBytes >= tokenScaleBytes &&
        scratch.accumulatorBytes >= accumulatorBytes) {
        return &scratch;
    }
    if (FastllmCudaGraphIsCapturing()) return nullptr;

    ExecutionScratch replacement;
    replacement.quantized = FastllmCudaMalloc(quantizedBytes);
    replacement.tokenScales = static_cast<float *>(
        FastllmCudaMalloc(tokenScaleBytes));
    replacement.accumulator = FastllmCudaMalloc(accumulatorBytes);
    if (replacement.quantized == nullptr || replacement.tokenScales == nullptr ||
        replacement.accumulator == nullptr) {
        ReleaseScratch(replacement);
        return nullptr;
    }
    replacement.quantizedBytes = quantizedBytes;
    replacement.tokenScaleBytes = tokenScaleBytes;
    replacement.accumulatorBytes = accumulatorBytes;
    ReleaseScratch(scratch);
    scratch = replacement;
    return &scratch;
}

static void *GetWorkspace(ExecutionScratch &scratch, size_t bytes) {
    if (bytes == 0) return nullptr;
    if (scratch.workspace != nullptr && scratch.workspaceBytes >= bytes) {
        return scratch.workspace;
    }
    if (FastllmCudaGraphIsCapturing()) return nullptr;
    void *replacement = FastllmCudaMalloc(bytes);
    if (replacement == nullptr) return nullptr;
    if (scratch.workspace != nullptr) FastllmCudaFree(scratch.workspace);
    scratch.workspace = replacement;
    scratch.workspaceBytes = bytes;
    return scratch.workspace;
}

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
         int m, int n, int k, cudaStream_t stream,
         ExecutionScratch &scratch) {
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
    void *workspace = GetWorkspace(scratch, workspaceBytes);
    if (workspaceBytes && workspace == nullptr) return false;
    cutlass::Status status = gemm.run(args, workspace, stream);
    return status == cutlass::Status::kSuccess && cudaGetLastError() == cudaSuccess;
}

template <typename Element, typename Accumulator>
bool Dispatch(int arch, Element const *a, Element const *b, Accumulator *d,
              int m, int n, int k, cudaStream_t stream,
              ExecutionScratch &scratch) {
#if defined(FASTLLM_CUTLASS_W8A8_SM90)
    if constexpr (std::is_same_v<Element, int8_t>) {
        if (arch != 90) return false;
        if (m <= 32 && n < 8192) {
            using D = DenseKernel<Element, Accumulator, cutlass::arch::Sm90,
                Shape<_64,_64,_256>, Shape<_1,_8,_1>,
                cutlass::gemm::KernelTmaWarpSpecialized,
                cutlass::epilogue::TmaWarpSpecialized>;
            return Run<D>(a, b, d, m, n, k, stream, scratch);
        }
        if (m <= 32) {
            using D = DenseKernel<Element, Accumulator, cutlass::arch::Sm90,
                Shape<_64,_128,_256>, Shape<_1,_4,_1>,
                cutlass::gemm::KernelTmaWarpSpecialized,
                cutlass::epilogue::TmaWarpSpecialized>;
            return Run<D>(a, b, d, m, n, k, stream, scratch);
        }
        if (m <= 64) {
            using D = DenseKernel<Element, Accumulator, cutlass::arch::Sm90,
                Shape<_64,_64,_256>, Shape<_1,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecialized,
                cutlass::epilogue::TmaWarpSpecialized>;
            return Run<D>(a, b, d, m, n, k, stream, scratch);
        }
        if (m <= 128) {
            using D = DenseKernel<Element, Accumulator, cutlass::arch::Sm90,
                Shape<_64,_128,_128>, Shape<_2,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                cutlass::epilogue::TmaWarpSpecialized>;
            return Run<D>(a, b, d, m, n, k, stream, scratch);
        }
        using D = DenseKernel<Element, Accumulator, cutlass::arch::Sm90,
            Shape<_128,_128,_128>, Shape<_2,_1,_1>,
            cutlass::gemm::KernelTmaWarpSpecializedPingpong,
            cutlass::epilogue::TmaWarpSpecialized>;
        return Run<D>(a, b, d, m, n, k, stream, scratch);
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
            return Run<D>(a, b, d, m, n, k, stream, scratch);
        }
        if (m <= 32) {
            using D = DenseKernel<Element, Accumulator, cutlass::arch::Sm120,
                Shape<_32,_64,_128>, Shape<_1,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                cutlass::epilogue::collective::EpilogueScheduleAuto,
                Shape<_32,_32>>;
            return Run<D>(a, b, d, m, n, k, stream, scratch);
        }
        if (m <= 256) {
            using D = DenseKernel<Element, Accumulator, cutlass::arch::Sm120,
                Shape<_64,_64,_128>, Shape<_1,_1,_1>,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                cutlass::epilogue::collective::EpilogueScheduleAuto>;
            return Run<D>(a, b, d, m, n, k, stream, scratch);
        }
        using D = DenseKernel<Element, Accumulator, cutlass::arch::Sm120,
            Shape<_128,_128,_128>, Shape<_1,_1,_1>,
            cutlass::gemm::collective::KernelScheduleAuto,
            cutlass::epilogue::collective::EpilogueScheduleAuto>;
        return Run<D>(a, b, d, m, n, k, stream, scratch);
    }
#endif
    return false;
}

template <typename Input, typename Quant, typename Accumulator, int MaxValue>
bool Execute(const fastllm::Data &input, fastllm::Data &weight,
             const fastllm::Data &bias, fastllm::Data &output,
             int m, int k, int n, int arch) {
    ExecutionScratch *scratch = GetExecutionScratch(
        (size_t)m * k * sizeof(Quant), (size_t)m * sizeof(float),
        (size_t)m * n * sizeof(Accumulator));
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight, bias, n);
    if (scratch == nullptr || weight.extraCudaData.size() < 2 ||
        weight.extraCudaData[0] == nullptr || weight.extraCudaData[1] == nullptr) {
        return false;
    }
    Quant *quant = static_cast<Quant *>(scratch->quantized);
    float *tokenScales = scratch->tokenScales;
    Accumulator *accumulator = static_cast<Accumulator *>(scratch->accumulator);
    const float *channelScales = (const float*)weight.extraCudaData[0];
    const float *biasData = (const float*)weight.extraCudaData[1];
    cudaStream_t stream = 0;
    QuantizePerToken<Input, Quant, MaxValue><<<m, 256, 0, stream>>>(
        (const Input*)input.cudaData, quant, tokenScales, m, k);
    bool ok = cudaGetLastError() == cudaSuccess &&
              Dispatch<Quant, Accumulator>(arch, quant,
                  (const Quant*)weight.cudaData, accumulator, m, n, k, stream,
                  *scratch);
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
    return ok;
}
} // namespace fastllm_w8a8_dense_detail
#endif

/**
 * 清除指定FP8权重记录的W8A8固定后端状态。
 *
 * FP8权重本身不需要额外的CUTLASS重排副本，但后端选择状态以Data地址和
 * GPU编号为键缓存。张量销毁、覆盖或迁移设备前必须删除全部对应记录，
 * 防止后续复用同一Data地址时继承旧权重的Cutlass或Rejected状态。
 *
 * @param weight 即将销毁、覆盖或迁移的权重；nullptr表示无需处理。
 */
void FastllmCudaReleaseFp8W8A8BackendState(const fastllm::Data *weight) {
#if defined(FASTLLM_ENABLE_CUTLASS_W8A8)
    using namespace fastllm_w8a8_dense_detail;
    if (weight == nullptr) return;
    std::lock_guard<std::mutex> guard(backendMutex);
    for (auto it = backendStates.begin(); it != backendStates.end();) {
        if (it->first.first == weight) {
            it = backendStates.erase(it);
        } else {
            ++it;
        }
    }
#else
    (void)weight;
#endif
}

FastllmCudaFp8W8A8BackendState FastllmCudaGetFp8W8A8BackendState(
        const fastllm::Data &weight, int device) {
#if defined(FASTLLM_ENABLE_CUTLASS_W8A8)
    return fastllm_w8a8_dense_detail::GetBackendState(weight, device);
#else
    (void)weight;
    (void)device;
    return FastllmCudaFp8W8A8BackendState::Rejected;
#endif
}

/**
 * 在标准FP8权重上传当前GPU后预先记录SM120 W8A8候选后端。
 *
 * 标准FP8权重已经是CUTLASS直接消费的[N,K]布局，因此本阶段不复制或
 * 重排权重，也不释放原始CUDA表示；这里只检查与M无关的GPU、权重形状
 * 和per-channel scale条件并记录Prepared。首次真实GEMM负责分配执行
 * scratch、同步验证内核，随后固定为Cutlass或Rejected。
 *
 * @param weight 已上传CUDA的标准FP8_E4M3二维线性权重[N,K]。
 * @return true表示权重已处于Prepared/Cutlass；false表示不是目标格式或
 *         已固定交给Legacy。
 */
bool FastllmCudaTryPrepareFp8W8A8Weight(fastllm::Data &weight) {
#if defined(FASTLLM_ENABLE_CUTLASS_W8A8) && defined(FASTLLM_CUTLASS_W8A8_SM120)
    using namespace fastllm_w8a8_dense_detail;
    if (weight.dataType != fastllm::DataType::FP8_E4M3 ||
        weight.weightType != fastllm::WeightType::LINEAR ||
        weight.dims.size() != 2) return false;
    const int device = FastllmCudaGetDevice();
    const BackendState state = GetBackendState(weight, device);
    if (state == BackendState::Prepared || state == BackendState::Cutlass) return true;
    if (state == BackendState::Rejected) return false;
    const int n = weight.dims[0], k = weight.dims[1];
    const bool supported = Enabled() && FastllmCudaRuntimeArch() == 120 &&
        n > 0 && k > 0 && n % 8 == 0 && k % 16 == 0 &&
        weight.blockK == 1 && weight.blockM == k &&
        weight.scales.size() == (size_t)n && weight.cudaData != nullptr;
    if (!supported) {
        SetBackendState(weight, device, BackendState::Rejected);
        Trace("fallback", "weight-load backend selection rejected", 1, n, k);
        if (StrictEnabled()) {
            throw std::runtime_error(
                "strict SM120 FP8 W8A8 weight-load selection failed");
        }
        return false;
    }
    SetBackendState(weight, device, BackendState::Prepared);
    Trace("w8a8-cutlass", "weight prepared; first GEMM validation pending",
          1, n, k);
    return true;
#else
    (void)weight;
    return false;
#endif
}

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
    // 本入口只接管标准per-channel FP8权重，其他FP8布局继续交给原有后端。
    if (weight.dataType != fastllm::DataType::FP8_E4M3) return false;

    const int device = FastllmCudaGetDevice();
    const BackendState state = GetBackendState(weight, device);
    if (state == BackendState::Rejected) {
        Trace("fallback", "backend fixed to legacy", m, n, k);
        if (StrictEnabled()) {
            throw std::runtime_error(
                "strict SM120 FP8 W8A8 requires the CUTLASS backend");
        }
        return false;
    }

    const bool selecting = state == BackendState::Uninitialized ||
                           state == BackendState::Prepared;
    const char *failure = nullptr;
    if (selecting && !Enabled()) failure = "backend disabled";
    else if (FastllmCudaRuntimeArch() != 120) failure = "runtime SM is not 120";
    else if (m <= 0 || k <= 0 || n <= 0) failure = "invalid GEMM dimensions";
    else if (k % 16 != 0) failure = "K is not aligned to 16";
    else if (n % 8 != 0) failure = "N is not aligned to 8";
    else if (input.cudaData == nullptr || weight.cudaData == nullptr ||
             output.cudaData == nullptr) failure = "CUDA tensor pointer is null";
    else if (weight.dims != std::vector<int>({n, k})) failure = "weight shape mismatch";
    else if (weight.blockK != 1 || weight.blockM != k ||
             weight.scales.size() != (size_t)n) failure = "weight scale layout mismatch";
    else if (input.dataType != fastllm::DataType::FLOAT16 &&
             input.dataType != fastllm::DataType::BFLOAT16) failure = "input is not FP16/BF16";
    else if (output.dataType != input.dataType) failure = "output dtype differs from input";
    else if (!bias.dims.empty() &&
             (bias.dataType != fastllm::DataType::FLOAT32 ||
              bias.cudaData == nullptr || bias.Count(0) != n)) {
        failure = "bias must be empty or FP32[N] on CUDA";
    }

    if (failure != nullptr) {
        Trace("fallback", failure, m, n, k);
        if (selecting) {
            SetBackendState(weight, device, BackendState::Rejected);
            if (StrictEnabled()) {
                throw std::runtime_error(
                    std::string("strict SM120 FP8 W8A8 selection failed: ") + failure);
            }
            return false;
        }
        ThrowFixedBackend(failure);
    }

    const bool ok = input.dataType == fastllm::DataType::FLOAT16
        ? Execute<half, cutlass::float_e4m3_t, float, 448>(
              input, weight, bias, output, m, k, n, 120)
        : Execute<__nv_bfloat16, cutlass::float_e4m3_t, float, 448>(
              input, weight, bias, output, m, k, n, 120);

    // CUDA调用异步返回。后端只在首次真实GEMM同步验证成功后固定，避免
    // 把异步内核错误误判为成功并在后续请求中静默混用Legacy。
    cudaError_t validation = selecting && ok
        ? cudaStreamSynchronize(0) : (ok ? cudaSuccess : cudaErrorUnknown);
    const bool forcedFailure = ok && validation == cudaSuccess &&
                               ForceGemmFailureForTest();
    if (forcedFailure && !selecting) {
        // 测试固定后端运行失败时先排空刚提交的合法kernel，避免故障注入
        // 留下异步工作影响后续case；生产环境不设置该测试开关。
        validation = cudaStreamSynchronize(0);
    }
    if (!ok || validation != cudaSuccess || forcedFailure) {
        const char *reason = !ok ? "CUTLASS launch failed" :
            (validation != cudaSuccess ? cudaGetErrorString(validation) :
             "injected CUTLASS GEMM failure");
        Trace("fallback", reason, m, n, k);
        if (selecting) {
            SetBackendState(weight, device, BackendState::Rejected);
            if (StrictEnabled()) {
                throw std::runtime_error(
                    std::string("strict SM120 FP8 W8A8 warmup failed: ") + reason);
            }
            return false;
        }
        ThrowFixedBackend(reason);
    }

    if (selecting) SetBackendState(weight, device, BackendState::Cutlass);
    Trace("w8a8-cutlass", selecting ? "backend fixed" : "success", m, n, k);
    return true;
#else
    (void)input; (void)weight; (void)bias; (void)output; (void)m; (void)k; (void)n;
    return false;
#endif
}
