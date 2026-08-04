#include "fastllm-cuda.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>

namespace {

struct WeightCache {
    const void *source = nullptr;
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

static bool SemanticsSupported(const fastllm::Data &input,
                               const fastllm::Data &weight,
                               const fastllm::Data &bias,
                               const fastllm::Data &output,
                               int n, int m, int k, const char **reason) {
    if (!Enabled()) { *reason = "disabled"; return false; }
    if (n < MinRows()) { *reason = "below W4A4 row threshold"; return false; }
    if (input.dataType != fastllm::DataType::FLOAT16 &&
        input.dataType != fastllm::DataType::BFLOAT16) {
        *reason = "activation is not FP16/BF16"; return false;
    }
    if (output.dataType != input.dataType) { *reason = "output dtype differs from input"; return false; }
    if (weight.dataType != fastllm::DataType::NVFP4_BLOCK_16 || weight.blockM != 16) {
        *reason = "weight is not NVFP4_BLOCK_16"; return false;
    }
    if (!bias.dims.empty() &&
        (bias.dataType != fastllm::DataType::FLOAT32 || bias.Count(0) != (uint64_t)k)) {
        *reason = "bias must be empty or FP32 with N elements"; return false;
    }
    if (m <= 0 || k <= 0 || m % 16 != 0) {
        *reason = "K is not compatible with NVFP4 block-16"; return false;
    }
    if (weight.dims.size() != 2 || weight.dims[0] != k || weight.dims[1] != m) {
        *reason = "weight shape mismatch"; return false;
    }
    return true;
}

static void ReleaseCache(WeightCache &cache) {
    if (cache.weight != nullptr) FastllmCudaFree(cache.weight);
    if (cache.scales != nullptr) FastllmCudaFree(cache.scales);
    if (cache.alpha != nullptr) FastllmCudaFree(cache.alpha);
    cache = WeightCache();
}

static WeightCache *GetWeightCache(fastllm::Data &weight, int m, int k) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return nullptr;
    std::lock_guard<std::mutex> guard(cacheMutex);
    auto key = std::make_pair((const fastllm::Data *)&weight, device);
    WeightCache &cache = weightCaches[key];
    const int paddedM = RoundUp(m, 32);
    const int paddedK = RoundUp(k, 32);
    if (cache.source == weight.cudaData && cache.sourceRows == k &&
        cache.sourceColumns == m && cache.paddedRows == paddedK &&
        cache.paddedColumns == paddedM &&
        cache.weight != nullptr && cache.scales != nullptr && cache.alpha != nullptr) return &cache;
    if (FastllmCudaGraphIsCapturing()) return nullptr;
    ReleaseCache(cache);
    cache.weight = static_cast<uint8_t *>(FastllmCudaMalloc((size_t)paddedK * paddedM / 2));
    cache.scales = static_cast<uint8_t *>(FastllmCudaMalloc(
        FastllmCudaNvfp4SwizzledScaleBytes(paddedK, paddedM)));
    cache.alpha = static_cast<float *>(FastllmCudaMalloc(sizeof(float)));
    if (cache.weight == nullptr || cache.scales == nullptr || cache.alpha == nullptr) {
        ReleaseCache(cache); return nullptr;
    }
    const float one = 1.0f;
    if (!FastllmCudaNvfp4Block16ToCutlassPadded(
            static_cast<const uint8_t *>(weight.cudaData), cache.weight,
            cache.scales, k, m, paddedK, paddedM, (void *)cudaStreamPerThread) ||
        cudaMemcpyAsync(cache.alpha, &one, sizeof(one), cudaMemcpyHostToDevice,
                        cudaStreamPerThread) != cudaSuccess) {
        ReleaseCache(cache); return nullptr;
    }
    cache.source = weight.cudaData;
    cache.sourceRows = k;
    cache.sourceColumns = m;
    cache.paddedRows = paddedK;
    cache.paddedColumns = paddedM;
    return &cache;
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
        inFeatures % 32 != 0 || outFeatures % 32 != 0) return false;
    WeightCache *cache = GetWeightCache(weight, inFeatures, outFeatures);
    if (cache == nullptr || cache->paddedColumns != inFeatures ||
        cache->paddedRows != outFeatures) return false;
    *packedWeight = cache->weight;
    *scales = cache->scales;
    *alpha = cache->alpha;
    return true;
}

static bool RunCutlassNvfp4W4A4(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output,
        int n, int m, int k, bool siluMulInput) {
    const int arch = RuntimeArch();
    const char *reason = "unsupported";
    const bool supportedArch = arch >= 100 && arch < 130;
    if (!supportedArch ||
        !SemanticsSupported(input, weight, bias, output, n, m, k, &reason)) {
        Trace("fallback", !supportedArch ? "runtime SM is not 100-129" : reason,
              n, m, k, arch);
        return false;
    }
    if (input.dims.empty() ||
        input.dims.back() != (siluMulInput ? m * 2 : m)) {
        Trace("fallback", "activation shape mismatch", n, m, k, arch);
        return false;
    }

    WeightCache *cache = GetWeightCache(weight, m, k);
    if (cache == nullptr) {
        Trace("fallback", "weight cache unavailable", n, m, k, arch);
        return false;
    }
    ActivationScratch *activation = GetActivationScratch(n, cache->paddedColumns);
    const bool paddedOutput = cache->paddedRows != k;
    OutputScratch *padded = paddedOutput
        ? GetOutputScratch((size_t)n * cache->paddedRows * sizeof(uint16_t)) : nullptr;
    if (activation == nullptr || (paddedOutput && padded == nullptr)) {
        Trace("fallback", "activation/output scratch unavailable", n, m, k, arch);
        return false;
    }

    void *inputData = FastllmCudaPrepareInput(input);
    void *outputData = FastllmCudaPrepareOutput(output);
    void *biasData = bias.dims.empty() ? nullptr : FastllmCudaPrepareInput(bias);
    void *gemmOutput = paddedOutput ? padded->output : outputData;
    const char *failureStage = "prepare input/output";
    bool ok = inputData != nullptr && outputData != nullptr &&
              (bias.dims.empty() || biasData != nullptr);
    if (ok) {
        failureStage = siluMulInput ? "SwiGLU activation quantization" :
                                      "activation quantization";
        ok = siluMulInput
            ? FastllmCudaSiluMulNvfp4QuantizePadded(
                  inputData, input.dataType, activation->activation, activation->scales,
                  n, m, cache->paddedColumns, 1.0f, (void *)cudaStreamPerThread)
            : FastllmCudaNvfp4QuantizeActivationPadded(
                  inputData, input.dataType, activation->activation, activation->scales,
                  n, m, cache->paddedColumns, 1.0f, (void *)cudaStreamPerThread);
    }
    if (ok) {
        failureStage = arch < 120 ? "SM100 CUTLASS GEMM" : "SM120 CUTLASS GEMM";
        ok = arch < 120
            ? FastllmCudaNvfp4CutlassGemmSm100(
                  activation->activation, cache->weight, activation->scales, cache->scales,
                  cache->alpha, gemmOutput, output.dataType,
                  n, cache->paddedRows, cache->paddedColumns,
                  (void *)cudaStreamPerThread)
            : FastllmCudaNvfp4CutlassGemmSm120(
                  activation->activation, cache->weight, activation->scales, cache->scales,
                  cache->alpha, gemmOutput, output.dataType,
                  n, cache->paddedRows, cache->paddedColumns,
                  (void *)cudaStreamPerThread);
    }
    if (ok && (paddedOutput || biasData != nullptr)) {
        failureStage = "output finalize";
        ok = FinalizeOutput(gemmOutput, outputData, static_cast<const float *>(biasData),
                            output.dataType, n, k, cache->paddedRows);
    }

    if (biasData != nullptr) FastllmCudaFinishInput(bias, biasData);
    if (inputData != nullptr) FastllmCudaFinishInput(input, inputData);
    if (outputData != nullptr) FastllmCudaFinishOutput(output, outputData);
    Trace(ok ? (siluMulInput ? "w4a4-swiglu-cutlass" : "w4a4-cutlass") : "fallback",
          ok ? "success" : failureStage, n, m, k, arch);
    return ok;
}

bool TryCudaCutlassNvfp4W4A4(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output,
        int n, int m, int k) {
    return RunCutlassNvfp4W4A4(input, weight, bias, output, n, m, k, false);
}

bool FastllmCudaCutlassNvfp4W4A4FromSwiglu(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output,
        int n, int m, int k) {
    return RunCutlassNvfp4W4A4(input, weight, bias, output, n, m, k, true);
}

void FastllmCudaReleaseNvfp4W4A4Cache(const fastllm::Data *weight) {
    std::lock_guard<std::mutex> guard(cacheMutex);
    for (auto it = weightCaches.begin(); it != weightCaches.end();) {
        if (it->first.first == weight) {
            ReleaseCache(it->second);
            it = weightCaches.erase(it);
        } else {
            ++it;
        }
    }
}
