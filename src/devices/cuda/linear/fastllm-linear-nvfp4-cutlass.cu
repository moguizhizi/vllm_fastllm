#include "fastllm-cuda.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace {

struct WeightCache {
    uint8_t *weight = nullptr;
    uint8_t *scales = nullptr;
    float *alpha = nullptr;
    int sourceRows = 0;
    int sourceColumns = 0;
    int paddedRows = 0;
    int paddedColumns = 0;
};

struct ActivationScratch {
    uint8_t *activation = nullptr;
    uint8_t *scales = nullptr;
    size_t activationBytes = 0;
    size_t scaleBytes = 0;
};

struct OutputScratch {
    void *output = nullptr;
    size_t bytes = 0;
};

std::mutex cacheMutex;
std::map<std::pair<const fastllm::Data *, int>, WeightCache> weightCaches;
std::map<int, ActivationScratch> activationScratch;
std::map<int, OutputScratch> outputScratch;

enum class BackendState : uint8_t {
    Uninitialized,
    Cutlass,
    Rejected,
};

enum class FusionState : uint8_t {
    Uninitialized,
    Enabled,
    Disabled,
};

std::mutex stateMutex;
std::map<std::pair<const fastllm::Data *, int>, BackendState> backendStates;
std::map<std::pair<const fastllm::Data *, int>, FusionState> fusionStates;

static int RoundUp(int value, int alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static bool Enabled() {
    const char *value = std::getenv("FASTLLM_CUDA_NVFP4_W4A4");
    return value == nullptr || value[0] == '\0' || value[0] != '0';
}

static bool TraceEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_NVFP4_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static int MinRows() {
    const char *value = std::getenv("FASTLLM_CUDA_NVFP4_W4A4_MIN_ROWS");
    return value == nullptr ? 1 : std::max(1, std::atoi(value));
}

static int RuntimeArch() {
    int device = 0, major = 0, minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) return 0;
    return major * 10 + minor;
}

static void Trace(const char *path, const char *reason, int n, int m, int k, int arch) {
    if (TraceEnabled()) {
        std::fprintf(stderr, "[fastllm][nvfp4] path=%s reason=%s n=%d m=%d k=%d sm=%d\n",
                     path, reason, n, m, k, arch);
    }
}

static bool TraceSynchronize(const char *stage, bool force = false) {
    if (!force && !TraceEnabled()) return true;
    const cudaError_t status = cudaStreamSynchronize(cudaStreamPerThread);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "[fastllm][nvfp4] stage=%s status=%s\n",
                     stage, cudaGetErrorString(status));
        return false;
    }
    return true;
}

static bool SemanticsSupported(const fastllm::Data &input,
                               const fastllm::Data &weight,
                               const fastllm::Data &bias,
                               const fastllm::Data &output,
                               int n, int m, int k, bool checkSelectionPolicy,
                               const char **reason) {
    if (checkSelectionPolicy && !Enabled()) { *reason = "disabled"; return false; }
    if (checkSelectionPolicy && n < MinRows()) {
        *reason = "below W4A4 row threshold"; return false;
    }
    if (input.dataType != fastllm::DataType::FLOAT16 &&
        input.dataType != fastllm::DataType::BFLOAT16) {
        *reason = "activation is not FP16/BF16"; return false;
    }
    if (output.dataType != input.dataType) { *reason = "output dtype differs from input"; return false; }
    if (weight.dataType != fastllm::DataType::NVFP4_BLOCK_16 || weight.blockM != 16) {
        *reason = "weight is not NVFP4_BLOCK_16"; return false;
    }
    if (m <= 0 || k <= 0 || m % 16 != 0) {
        *reason = "K is not compatible with NVFP4 block-16"; return false;
    }
    const size_t expectedScales = (size_t)k * ((size_t)m / 16);
    if (weight.nvfp4GroupScales.size() != expectedScales ||
        !std::isfinite(weight.nvfp4GlobalScale) || weight.nvfp4GlobalScale <= 0.0f) {
        *reason = "original E4M3/global scales are unavailable"; return false;
    }
    if (!bias.dims.empty() &&
        (bias.dataType != fastllm::DataType::FLOAT32 || bias.Count(0) != (uint64_t)k)) {
        *reason = "bias must be empty or FP32 with N elements"; return false;
    }
    if (weight.dims.size() != 2 || weight.dims[0] != k || weight.dims[1] != m) {
        *reason = "weight shape mismatch"; return false;
    }
    if (weight.cudaData != nullptr &&
        (weight.cudaDataBorrowed || weight.multiDeviceData ||
         (weight.cpuData == nullptr && weight.numasData.empty()))) {
        *reason = "original CUDA weight has no releasable host source"; return false;
    }
    return true;
}

static int CurrentDevice() {
    int device = 0;
    return cudaGetDevice(&device) == cudaSuccess ? device : -1;
}

/**
 * 获取指定NVFP4权重在指定GPU上的CUTLASS后端状态。
 *
 * 状态以“Data对象地址 + CUDA设备号”为键保存，因此同一权重移动到
 * 另一张GPU后会拥有独立的后端生命周期。读取过程由stateMutex保护；
 * 如果尚未登记状态，则返回Uninitialized，但不会向状态表插入新记录。
 *
 * @param weight 用于标识模型权重对象。
 * @param device CUDA设备号。
 * @return 当前后端状态：Uninitialized、Cutlass或Rejected。
 */
static BackendState GetBackendState(const fastllm::Data &weight, int device) {
    std::lock_guard<std::mutex> guard(stateMutex);
    auto it = backendStates.find({&weight, device});
    return it == backendStates.end() ? BackendState::Uninitialized : it->second;
}

static void SetBackendState(const fastllm::Data &weight, int device, BackendState state) {
    std::lock_guard<std::mutex> guard(stateMutex);
    backendStates[{&weight, device}] = state;
}

static FusionState GetFusionState(const fastllm::Data &weight, int device) {
    std::lock_guard<std::mutex> guard(stateMutex);
    auto it = fusionStates.find({&weight, device});
    return it == fusionStates.end() ? FusionState::Uninitialized : it->second;
}

static void SetFusionState(const fastllm::Data &weight, int device, FusionState state) {
    std::lock_guard<std::mutex> guard(stateMutex);
    fusionStates[{&weight, device}] = state;
}

static void ReleaseCache(WeightCache &cache) {
    if (cache.weight != nullptr) FastllmCudaFree(cache.weight);
    if (cache.scales != nullptr) FastllmCudaFree(cache.scales);
    if (cache.alpha != nullptr) FastllmCudaFree(cache.alpha);
    cache = WeightCache();
}

static WeightCache *GetWeightCache(fastllm::Data &weight, int m, int k,
                                   bool releaseSourceAfterRepack) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return nullptr;
    std::lock_guard<std::mutex> guard(cacheMutex);
    auto key = std::make_pair((const fastllm::Data *)&weight, device);
    WeightCache &cache = weightCaches[key];
    const int paddedM = RoundUp(m, 32);
    const int paddedK = RoundUp(k, 32);
    if (cache.sourceRows == k && cache.sourceColumns == m &&
        cache.paddedRows == paddedK &&
        cache.paddedColumns == paddedM &&
        cache.weight != nullptr && cache.scales != nullptr && cache.alpha != nullptr) return &cache;
    if (FastllmCudaGraphIsCapturing()) return nullptr;
    ReleaseCache(cache);
    if (weight.cudaData == nullptr &&
        !weight.RestoreCudaDataForRepackedWeight()) return nullptr;
    cache.weight = static_cast<uint8_t *>(FastllmCudaMalloc((size_t)paddedK * paddedM / 2));
    cache.scales = static_cast<uint8_t *>(FastllmCudaMalloc(
        FastllmCudaNvfp4SwizzledScaleBytes(paddedK, paddedM)));
    cache.alpha = static_cast<float *>(FastllmCudaMalloc(sizeof(float)));
    const size_t sourceScaleBytes = (size_t)k * (m / 16);
    uint8_t *sourceScales = static_cast<uint8_t *>(FastllmCudaMalloc(sourceScaleBytes));
    if (cache.weight == nullptr || cache.scales == nullptr || cache.alpha == nullptr) {
        if (sourceScales != nullptr) FastllmCudaFree(sourceScales);
        ReleaseCache(cache); return nullptr;
    }
    const bool queued = sourceScales != nullptr &&
        cudaMemcpyAsync(sourceScales, weight.nvfp4GroupScales.data(), sourceScaleBytes,
                        cudaMemcpyHostToDevice, cudaStreamPerThread) == cudaSuccess &&
        FastllmCudaNvfp4Block16ToCutlassPadded(
            static_cast<const uint8_t *>(weight.cudaData), sourceScales, cache.weight,
            cache.scales, k, m, paddedK, paddedM, (void *)cudaStreamPerThread) &&
        cudaMemcpyAsync(cache.alpha, &weight.nvfp4GlobalScale,
                        sizeof(weight.nvfp4GlobalScale), cudaMemcpyHostToDevice,
                        cudaStreamPerThread) == cudaSuccess;
    const cudaError_t syncStatus = cudaStreamSynchronize(cudaStreamPerThread);
    if (sourceScales != nullptr) FastllmCudaFree(sourceScales);
    if (!queued || syncStatus != cudaSuccess) {
        ReleaseCache(cache);
        return nullptr;
    }
    cache.sourceRows = k;
    cache.sourceColumns = m;
    cache.paddedRows = paddedK;
    cache.paddedColumns = paddedM;
    const bool released = !releaseSourceAfterRepack ||
                          weight.ReleaseCudaDataForRepackedWeight();
    if (!released) {
        ReleaseCache(cache);
        return nullptr;
    }
    if (releaseSourceAfterRepack && released && TraceEnabled()) {
        std::fprintf(stderr,
                     "[fastllm][nvfp4] weight_source=released bytes=%zu name=%s\n",
                     (size_t)weight.expansionBytes, weight.name.c_str());
    }
    return &cache;
}

static void ReleaseWeightCacheForDevice(const fastllm::Data *weight, int device) {
    std::lock_guard<std::mutex> guard(cacheMutex);
    auto it = weightCaches.find({weight, device});
    if (it != weightCaches.end()) {
        int originalDevice = 0;
        cudaGetDevice(&originalDevice);
        cudaSetDevice(device);
        ReleaseCache(it->second);
        weightCaches.erase(it);
        cudaSetDevice(originalDevice);
    }
}

static ActivationScratch *GetActivationScratch(int n, int m) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return nullptr;
    const size_t activationBytes = (size_t)n * m / 2;
    const size_t scaleBytes = FastllmCudaNvfp4SwizzledScaleBytes(n, m);
    std::lock_guard<std::mutex> guard(cacheMutex);
    ActivationScratch &scratch = activationScratch[device];
    if (scratch.activationBytes >= activationBytes && scratch.scaleBytes >= scaleBytes &&
        scratch.activation != nullptr && scratch.scales != nullptr) return &scratch;
    if (FastllmCudaGraphIsCapturing()) return nullptr;
    uint8_t *newActivation = static_cast<uint8_t *>(FastllmCudaMalloc(activationBytes));
    uint8_t *newScales = static_cast<uint8_t *>(FastllmCudaMalloc(scaleBytes));
    if (newActivation == nullptr || newScales == nullptr) {
        if (newActivation != nullptr) FastllmCudaFree(newActivation);
        if (newScales != nullptr) FastllmCudaFree(newScales);
        return nullptr;
    }
    if (scratch.activation != nullptr) FastllmCudaFree(scratch.activation);
    if (scratch.scales != nullptr) FastllmCudaFree(scratch.scales);
    scratch.activation = newActivation;
    scratch.scales = newScales;
    scratch.activationBytes = activationBytes;
    scratch.scaleBytes = scaleBytes;
    return &scratch;
}

static OutputScratch *GetOutputScratch(size_t bytes) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return nullptr;
    std::lock_guard<std::mutex> guard(cacheMutex);
    OutputScratch &scratch = outputScratch[device];
    if (scratch.output != nullptr && scratch.bytes >= bytes) return &scratch;
    if (FastllmCudaGraphIsCapturing()) return nullptr;
    void *newOutput = FastllmCudaMalloc(bytes);
    if (newOutput == nullptr) return nullptr;
    if (scratch.output != nullptr) FastllmCudaFree(scratch.output);
    scratch.output = newOutput;
    scratch.bytes = bytes;
    return &scratch;
}

template <typename T>
__global__ void FinalizeOutputKernel(const T *source, T *destination,
                                     const float *bias, int rows, int columns,
                                     int sourceStride) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t elements = (size_t)rows * columns;
    if (index >= elements) return;
    const int row = index / columns;
    const int column = index - (size_t)row * columns;
    float value;
    if constexpr (std::is_same<T, half>::value) {
        value = __half2float(source[(size_t)row * sourceStride + column]);
        if (bias != nullptr) value += bias[column];
        destination[index] = __float2half_rn(value);
    } else {
        value = __bfloat162float(source[(size_t)row * sourceStride + column]);
        if (bias != nullptr) value += bias[column];
        destination[index] = __float2bfloat16_rn(value);
    }
}

static bool FinalizeOutput(const void *source, void *destination,
                           const float *bias, fastllm::DataType dataType,
                           int rows, int columns, int sourceStride) {
    const size_t elements = (size_t)rows * columns;
    const int threads = 256;
    const int blocks = (elements + threads - 1) / threads;
    if (dataType == fastllm::DataType::FLOAT16) {
        FinalizeOutputKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            static_cast<const half *>(source), static_cast<half *>(destination),
            bias, rows, columns, sourceStride);
    } else if (dataType == fastllm::DataType::BFLOAT16) {
        FinalizeOutputKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(source),
            static_cast<__nv_bfloat16 *>(destination), bias,
            rows, columns, sourceStride);
    } else {
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

bool FastllmCudaPrepareNvfp4W4A4Weight(
        fastllm::Data &weight, int inFeatures, int outFeatures,
        const uint8_t **packedWeight, const uint8_t **scales,
        const float **alpha) {
    if (packedWeight == nullptr || scales == nullptr || alpha == nullptr ||
        weight.dataType != fastllm::DataType::NVFP4_BLOCK_16 ||
        weight.blockM != 16 || inFeatures <= 0 || outFeatures <= 0 ||
        inFeatures % 32 != 0 || outFeatures % 32 != 0 ||
        weight.nvfp4GroupScales.size() !=
            (size_t)outFeatures * (inFeatures / 16) ||
        !std::isfinite(weight.nvfp4GlobalScale) ||
        weight.nvfp4GlobalScale <= 0.0f) return false;
    WeightCache *cache = GetWeightCache(weight, inFeatures, outFeatures, true);
    if (cache == nullptr || cache->paddedColumns != inFeatures ||
        cache->paddedRows != outFeatures) return false;
    *packedWeight = cache->weight;
    *scales = cache->scales;
    *alpha = cache->alpha;
    return true;
}

/**
 * 执行一次NVFP4 W4A4 CUTLASS线性计算。
 *
 * 本函数只负责单次计算，不决定后端的最终生命周期。执行流程为：检查
 * GPU与张量语义、取得CUTLASS重排权重、准备临时缓冲区、将FP16/BF16
 * 激活动态量化为NVFP4、调用对应架构的CUTLASS GEMM，最后处理padding
 * 和bias。首次后端选择时可强制同步，以便在释放原始权重前发现异步错误。
 *
 * 参数采用标准GEMM语义：output[M,N] = input[M,K] * weight[N,K]^T。
 *
 * @param input                FP16或BF16输入；普通路径形状为[m, k]，融合
 *                             路径形状为[m, 2 * k]。
 * @param weight               NVFP4_BLOCK_16权重，逻辑形状为[n, k]。
 * @param bias                 可选FP32偏置，长度为n。
 * @param output               输出张量，逻辑形状为[m, n]。
 * @param m                    GEMM的M维，激活行数，通常为token数。
 * @param n                    GEMM的N维，输出特征数。
 * @param k                    GEMM的K维，输入特征数。
 * @param siluMulInput         true表示把SwiGLU计算与激活量化融合。
 * @param checkSelectionPolicy true表示同时检查开关和最小行数等选择策略；
 *                             后端固定后的正式计算传false。
 * @param validateWarmup       true表示末尾强制同步，验证首次warmup的异步错误。
 * @return true表示本次CUTLASS计算及必要的后处理成功；false表示某个阶段
 *         不满足条件或执行失败，具体阶段通过Trace记录。
 */
static bool RunCutlassNvfp4W4A4(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output,
        int m, int n, int k, bool siluMulInput,
        bool checkSelectionPolicy, bool validateWarmup) {
    // 先检查运行架构、数据类型、权重格式、scale元数据和维度语义。
    const int arch = RuntimeArch();
    const char *reason = "unsupported";
    const bool supportedArch = arch >= 100 && arch < 130;
    if (!supportedArch ||
        !SemanticsSupported(input, weight, bias, output, m, k, n,
                            checkSelectionPolicy, &reason)) {
        Trace("fallback", !supportedArch ? "runtime SM is not 100-129" : reason,
              m, k, n, arch);
        return false;
    }
    if (input.dims.empty() ||
        input.dims.back() != (siluMulInput ? k * 2 : k)) {
        Trace("fallback", "activation shape mismatch", m, k, n, arch);
        return false;
    }

    // 获取或创建当前GPU专属的CUTLASS格式权重。此处暂不释放原始权重；
    // 是否释放由外层后端生命周期函数在warmup成功后决定。
    WeightCache *cache = GetWeightCache(weight, k, n, false);
    if (cache == nullptr) {
        Trace("fallback", "weight cache unavailable", m, k, n, arch);
        return false;
    }
    if (!TraceSynchronize("weight repack")) {
        Trace("fallback", "weight repack", m, k, n, arch);
        return false;
    }

    // 激活量化需要补齐后的A和scale缓冲区；输出特征被补齐时，GEMM先
    // 写入临时输出，随后再切回真实的n列。
    ActivationScratch *activation = GetActivationScratch(m, cache->paddedColumns);
    const bool paddedOutput = cache->paddedRows != n;
    OutputScratch *padded = paddedOutput
        ? GetOutputScratch((size_t)m * cache->paddedRows * sizeof(uint16_t)) : nullptr;
    if (activation == nullptr || (paddedOutput && padded == nullptr)) {
        Trace("fallback", "activation/output scratch unavailable", m, k, n, arch);
        return false;
    }

    // 接入FastLLM张量的设备指针，并为可选bias取得输入指针。
    void *inputData = FastllmCudaPrepareInput(input);
    void *outputData = FastllmCudaPrepareOutput(output);
    void *biasData = bias.dims.empty() ? nullptr : FastllmCudaPrepareInput(bias);
    void *gemmOutput = paddedOutput ? padded->output : outputData;
    const char *failureStage = "prepare input/output";
    bool ok = inputData != nullptr && outputData != nullptr &&
              (bias.dims.empty() || biasData != nullptr);
    if (ok) {
        // 普通路径只做动态NVFP4量化；融合路径先计算SwiGLU，再直接生成
        // CUTLASS所需的NVFP4激活和E4M3 block scale。
        failureStage = siluMulInput ? "SwiGLU activation quantization" :
                                      "activation quantization";
        ok = siluMulInput
            ? FastllmCudaSiluMulNvfp4QuantizePadded(
                  inputData, input.dataType, activation->activation, activation->scales,
                  m, k, cache->paddedColumns, 1.0f, (void *)cudaStreamPerThread)
            : FastllmCudaNvfp4QuantizeActivationPadded(
                  inputData, input.dataType, activation->activation, activation->scales,
                  m, k, cache->paddedColumns, 1.0f, (void *)cudaStreamPerThread);
        if (ok) ok = TraceSynchronize(failureStage);
    }
    if (ok) {
        // SM100/SM120使用各自实例化的CUTLASS内核。GEMM直接消费NVFP4
        // 激活、NVFP4权重、两侧block scale以及权重global scale。
        failureStage = arch < 120 ? "SM100 CUTLASS GEMM" : "SM120 CUTLASS GEMM";
        ok = arch < 120
            ? FastllmCudaNvfp4CutlassGemmSm100(
                  activation->activation, cache->weight, activation->scales, cache->scales,
                  cache->alpha, gemmOutput, output.dataType,
                  m, cache->paddedRows, cache->paddedColumns,
                  (void *)cudaStreamPerThread)
            : FastllmCudaNvfp4CutlassGemmSm120(
                  activation->activation, cache->weight, activation->scales, cache->scales,
                  cache->alpha, gemmOutput, output.dataType,
                  m, cache->paddedRows, cache->paddedColumns,
                  (void *)cudaStreamPerThread);
    }
    if (ok && (paddedOutput || biasData != nullptr)) {
        // 去掉输出N维padding，并在需要时同时加上FP32 bias。
        failureStage = "output finalize";
        ok = FinalizeOutput(gemmOutput, outputData, static_cast<const float *>(biasData),
                            output.dataType, m, n, cache->paddedRows);
    }
    if (ok && validateWarmup) {
        // CUDA调用通常异步返回。首次warmup必须同步，避免内核实际失败后
        // 外层仍将CUTLASS标记为固定后端并释放原始CUDA权重。
        failureStage = "warmup validation";
        ok = TraceSynchronize(failureStage, true);
    }

    // 与PrepareInput/PrepareOutput成对收尾，并记录本次最终路径或失败阶段。
    if (biasData != nullptr) FastllmCudaFinishInput(bias, biasData);
    if (inputData != nullptr) FastllmCudaFinishInput(input, inputData);
    if (outputData != nullptr) FastllmCudaFinishOutput(output, outputData);
    Trace(ok ? (siluMulInput ? "w4a4-swiglu-cutlass" : "w4a4-cutlass") : "fallback",
          ok ? "success" : failureStage, m, k, n, arch);
    return ok;
}

/**
 * 尝试使用CUTLASS执行NVFP4 W4A4线性计算。
 *
 * 该函数同时负责当前权重在当前GPU上的后端生命周期管理：第一次调用
 * 会执行带同步验证的warmup；成功后固定使用CUTLASS并释放原始CUDA
 * 权重，失败后固定拒绝CUTLASS，由上层改走Legacy后端。后端一旦固定为
 * CUTLASS，正式推理期间发生的执行错误将直接抛出，不再动态切换后端。
 *
 * 参数采用标准GEMM语义：output[M,N] = input[M,K] * weight[N,K]^T。
 *
 * @param input  FP16或BF16激活，逻辑形状为[m, k]。
 * @param weight NVFP4_BLOCK_16权重，逻辑形状为[n, k]；可能在warmup
 *               成功后释放其原始CUDA表示。
 * @param bias   可选的FP32偏置，长度为n。
 * @param output 输出张量，逻辑形状为[m, n]，类型与input相同。
 * @param m      GEMM的M维，激活行数，通常是本次参与计算的token数。
 * @param n      GEMM的N维，输出特征数。
 * @param k      GEMM的K维，输入特征数。
 * @return true表示本次已由CUTLASS完成；false表示CUTLASS未接管，调用方
 *         应使用已经选定的其他后端。
 */
bool TryCudaCutlassNvfp4W4A4(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output,
        int m, int n, int k) {
    // 本入口只处理每16个权重共享一组缩放因子的NVFP4权重。
    if (weight.dataType != fastllm::DataType::NVFP4_BLOCK_16 ||
        weight.blockM != 16) return false;
    const int device = CurrentDevice();
    if (device < 0) return false;

    // 后端状态按“权重对象 + GPU”保存。同一份权重换到另一张GPU后，
    // 需要在目标GPU上重新完成一次后端选择和权重重排。
    const BackendState state = GetBackendState(weight, device);
    if (state == BackendState::Rejected) {
        // 首次warmup已经确认CUTLASS不可用，后续固定交给Legacy路径，
        // 避免每次Linear都重复创建cache并再次尝试失败。
        Trace("fallback", "CUTLASS backend rejected during warmup", m, k, n,
              RuntimeArch());
        return false;
    }

    // 只有未初始化状态才执行带同步检查的warmup。后端固定后跳过
    // 选择策略检查，防止因不同batch行数在CUTLASS和Legacy之间切换。
    const bool warmup = state == BackendState::Uninitialized;
    bool ok = RunCutlassNvfp4W4A4(input, weight, bias, output, m, n, k, false,
                                  warmup, warmup);
    if (ok && warmup) {
        // warmup成功后，CUTLASS重排权重成为该GPU上的持久表示。
        // 释放原始CUDA权重，避免同时保存原始格式和重排格式。
        if (!weight.ReleaseCudaDataForRepackedWeight()) {
            ReleaseWeightCacheForDevice(&weight, device);
            SetBackendState(weight, device, BackendState::Rejected);
            return false;
        }
        SetBackendState(weight, device, BackendState::Cutlass);
        if (TraceEnabled()) {
            std::fprintf(stderr,
                "[fastllm][nvfp4] backend=cutlass state=fixed device=%d name=%s\n",
                device, weight.name.c_str());
        }
    } else if (!ok && warmup) {
        // 只允许在首次warmup阶段降级：销毁未通过验证的重排权重，
        // 并永久标记为Rejected，使后续调用固定走Legacy路径。
        ReleaseWeightCacheForDevice(&weight, device);
        SetBackendState(weight, device, BackendState::Rejected);
    } else if (!ok) {
        // 后端固定后再失败不能静默切换，否则同一层会在推理过程中
        // 改变权重表示和计算路径；此时直接报错暴露真实故障。
        throw std::runtime_error(
            "NVFP4 CUTLASS backend failed after warmup; runtime fallback is disabled");
    }
    return ok;
}

bool FastllmCudaCutlassNvfp4W4A4FromSwiglu(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output,
        int n, int m, int k) {
    const int device = CurrentDevice();
    if (device < 0) return false;
    const BackendState backend = GetBackendState(weight, device);
    const FusionState fusion = GetFusionState(weight, device);
    if (backend == BackendState::Rejected || fusion == FusionState::Disabled) return false;

    const bool backendWarmup = backend == BackendState::Uninitialized;
    const bool fusionWarmup = fusion == FusionState::Uninitialized;
    // 融合入口仍使用FastLLM旧维度命名，在这里映射为标准GEMM M/N/K。
    bool ok = RunCutlassNvfp4W4A4(input, weight, bias, output, n, k, m, true,
                                  backendWarmup, backendWarmup || fusionWarmup);
    if (ok) {
        if (backendWarmup) {
            if (!weight.ReleaseCudaDataForRepackedWeight()) {
                ReleaseWeightCacheForDevice(&weight, device);
                SetBackendState(weight, device, BackendState::Rejected);
                SetFusionState(weight, device, FusionState::Disabled);
                return false;
            }
            SetBackendState(weight, device, BackendState::Cutlass);
        }
        if (fusionWarmup) SetFusionState(weight, device, FusionState::Enabled);
    } else if (fusionWarmup) {
        // 融合能力单独降级。保留Linear后端的未初始化状态，随后由普通
        // SwiGLU + Linear路径独立完成CUTLASS warmup和后端选择。
        SetFusionState(weight, device, FusionState::Disabled);
    } else {
        throw std::runtime_error(
            "NVFP4 fused SwiGLU backend failed after warmup; runtime fallback is disabled");
    }
    return ok;
}

void FastllmCudaReleaseNvfp4W4A4Cache(const fastllm::Data *weight) {
    {
        std::lock_guard<std::mutex> guard(cacheMutex);
        bool hasCache = false;
        for (const auto &entry : weightCaches) {
            if (entry.first.first == weight) { hasCache = true; break; }
        }
        if (hasCache) {
            int originalDevice = 0;
            cudaGetDevice(&originalDevice);
            for (auto it = weightCaches.begin(); it != weightCaches.end();) {
                if (it->first.first == weight) {
                    cudaSetDevice(it->first.second);
                    ReleaseCache(it->second);
                    it = weightCaches.erase(it);
                } else {
                    ++it;
                }
            }
            cudaSetDevice(originalDevice);
        }
    }
    {
        std::lock_guard<std::mutex> guard(stateMutex);
        for (auto it = backendStates.begin(); it != backendStates.end();) {
            if (it->first.first == weight) it = backendStates.erase(it);
            else ++it;
        }
        for (auto it = fusionStates.begin(); it != fusionStates.end();) {
            if (it->first.first == weight) it = fusionStates.erase(it);
            else ++it;
        }
    }
}
