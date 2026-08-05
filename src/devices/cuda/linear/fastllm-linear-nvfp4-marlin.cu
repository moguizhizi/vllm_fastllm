/* Dense NVFP4 Marlin W4A16 integration, adapted from vLLM's
 * marlin_utils_fp4.py and quantization/marlin kernels (Apache-2.0). */
#include "fastllm-cuda.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>

namespace {

struct Cache {
    const void *source = nullptr;
    uint32_t *weight = nullptr;
    uint8_t *scales = nullptr;
    float *globalScale = nullptr;
    int *workspace = nullptr;
    int inFeatures = 0;
    int outFeatures = 0;
};

std::mutex cacheMutex;
std::map<std::pair<const fastllm::Data *, int>, Cache> caches;

enum class BackendState : uint8_t {
    Uninitialized,
    Marlin,
    Legacy,
};

std::mutex stateMutex;
std::map<std::pair<const fastllm::Data *, int>, BackendState> backendStates;

static bool TraceEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_NVFP4_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static BackendState GetBackendState(const fastllm::Data &weight, int device) {
    std::lock_guard<std::mutex> guard(stateMutex);
    auto it = backendStates.find({&weight, device});
    return it == backendStates.end() ? BackendState::Uninitialized : it->second;
}

static void SetBackendState(const fastllm::Data &weight, int device, BackendState state) {
    std::lock_guard<std::mutex> guard(stateMutex);
    backendStates[{&weight, device}] = state;
}

static void FixLegacyBackend(const fastllm::Data &weight, int device,
                             const char *reason) {
    SetBackendState(weight, device, BackendState::Legacy);
    if (TraceEnabled()) {
        std::fprintf(stderr,
            "[fastllm][nvfp4] backend=legacy state=fixed device=%d reason=%s name=%s\n",
            device, reason, weight.name.c_str());
    }
}

static int RuntimeArch() {
    int device = 0, major = 0, minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) return 0;
    return major * 10 + minor;
}

__global__ void Block16ToGptqKernel(const uint8_t *source, uint32_t *qweight,
                                    int inFeatures, int outFeatures,
                                    int bytesPerRow) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int packs = inFeatures / 8;
    if (index >= packs * outFeatures) return;
    const int pack = index / outFeatures;
    const int out = index - pack * outFeatures;
    const int block = pack / 2;
    const int within = (pack & 1) * 4;
    const uint8_t *src = source + (size_t)out * bytesPerRow
                       + (size_t)block * (8 + sizeof(float)) + within;
    qweight[index] = *reinterpret_cast<const uint32_t *>(src);
}

__global__ void Block16ToMarlinScaleKernel(const uint8_t *source,
                                           uint8_t *scales,
                                           int inFeatures, int outFeatures,
                                           int bytesPerRow) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups = inFeatures / 16;
    if (index >= groups * outFeatures) return;
    const int group = index / outFeatures;
    const int out = index - group * outFeatures;
    const uint8_t *block = source + (size_t)out * bytesPerRow
                         + (size_t)group * (8 + sizeof(float));
    const float value = *reinterpret_cast<const float *>(block + 8);
    // vLLM NVFP4 Marlin uses a special unsigned S0E5M3 byte, not ordinary
    // E4M3.  It is the high byte of FP16(scale * 2^7) after a one-bit shift.
    const half scaled = __float2half_rn(value * 128.0f);
    const uint16_t bits = reinterpret_cast<const uint16_t &>(scaled);
    const uint8_t encoded = (uint8_t)(bits >> 7);

    const int block64 = index & ~63;
    const int inner64 = index & 63;
    const int permuted64 = block64 + (inner64 & 7) * 8 + (inner64 >> 3);
    const int block4 = permuted64 & ~3;
    const int inner4 = permuted64 & 3;
    const int inverse4 = inner4 == 1 ? 2 : (inner4 == 2 ? 1 : inner4);
    scales[block4 + inverse4] = encoded;
}

static void Release(Cache &cache) {
    if (cache.weight != nullptr) FastllmCudaFree(cache.weight);
    if (cache.scales != nullptr) FastllmCudaFree(cache.scales);
    if (cache.globalScale != nullptr) FastllmCudaFree(cache.globalScale);
    if (cache.workspace != nullptr) FastllmCudaFree(cache.workspace);
    cache = Cache();
}

static Cache *GetCache(fastllm::Data &weight, int inFeatures, int outFeatures) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return nullptr;
    std::lock_guard<std::mutex> guard(cacheMutex);
    auto key = std::make_pair((const fastllm::Data *)&weight, device);
    Cache &cache = caches[key];
    if (cache.inFeatures == inFeatures && cache.outFeatures == outFeatures &&
        cache.weight != nullptr &&
        cache.scales != nullptr && cache.globalScale != nullptr &&
        cache.workspace != nullptr) return &cache;
    if (FastllmCudaGraphIsCapturing()) return nullptr;
    Release(cache);
    if (weight.cudaData == nullptr &&
        !weight.RestoreCudaDataForRepackedWeight()) return nullptr;

    const size_t qweightCount = (size_t)(inFeatures / 8) * outFeatures;
    uint32_t *standard = static_cast<uint32_t *>(FastllmCudaMalloc(qweightCount * sizeof(uint32_t)));
    cache.weight = static_cast<uint32_t *>(FastllmCudaMalloc(qweightCount * sizeof(uint32_t)));
    cache.scales = static_cast<uint8_t *>(FastllmCudaMalloc(
        (size_t)(inFeatures / 16) * outFeatures));
    cache.globalScale = static_cast<float *>(FastllmCudaMalloc(sizeof(float)));
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cache.workspace = static_cast<int *>(FastllmCudaMalloc(
        (size_t)std::max(1, sms * 4) * sizeof(int)));
    if (standard == nullptr || cache.weight == nullptr || cache.scales == nullptr ||
        cache.globalScale == nullptr || cache.workspace == nullptr) {
        if (standard != nullptr) FastllmCudaFree(standard);
        Release(cache); return nullptr;
    }

    const int bytesPerRow = (inFeatures / 16) * (8 + sizeof(float));
    const int threads = 256;
    Block16ToGptqKernel<<<(qweightCount + threads - 1) / threads, threads, 0,
                           cudaStreamPerThread>>>(
        static_cast<const uint8_t *>(weight.cudaData), standard,
        inFeatures, outFeatures, bytesPerRow);
    const size_t scaleCount = (size_t)(inFeatures / 16) * outFeatures;
    Block16ToMarlinScaleKernel<<<(scaleCount + threads - 1) / threads, threads, 0,
                                 cudaStreamPerThread>>>(
        static_cast<const uint8_t *>(weight.cudaData), cache.scales,
        inFeatures, outFeatures, bytesPerRow);
    // The FP4 and scale dequant fast paths intentionally omit exponent-bias
    // multiplies.  For FP16 output vLLM compensates them with 2^(14 - 7).
    const float globalScale = 128.0f;
    bool ok = cudaPeekAtLastError() == cudaSuccess &&
              cudaMemcpyAsync(cache.globalScale, &globalScale, sizeof(float),
                              cudaMemcpyHostToDevice, cudaStreamPerThread) == cudaSuccess &&
              cudaMemsetAsync(cache.workspace, 0,
                              (size_t)std::max(1, sms * 4) * sizeof(int),
                              cudaStreamPerThread) == cudaSuccess &&
              FastllmCudaGptqMarlinRepackBitsStream(
                  standard, cache.weight, inFeatures, outFeatures, 4,
                  (void *)cudaStreamPerThread);
    FastllmCudaFree(standard);
    if (!ok) { Release(cache); return nullptr; }
    cache.source = weight.cudaData;
    cache.inFeatures = inFeatures;
    cache.outFeatures = outFeatures;
    return &cache;
}

} // namespace

bool FastllmCudaTryMarlinHalfMatMulNVFP4(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output,
        int n, int m, int k) {
    if (weight.dataType != fastllm::DataType::NVFP4_BLOCK_16 ||
        weight.blockM != 16) return false;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    const BackendState state = GetBackendState(weight, device);
    if (state == BackendState::Legacy) return false;
    const bool warmup = state == BackendState::Uninitialized;
    const int arch = RuntimeArch();
    const bool canRun = arch >= 75 && arch < 100 &&
        input.dataType == fastllm::DataType::FLOAT16 &&
        output.dataType == fastllm::DataType::FLOAT16 &&
        weight.dataType == fastllm::DataType::NVFP4_BLOCK_16 &&
        weight.blockM == 16 && bias.dims.empty() && n > 0 &&
        m > 0 && k > 0 && m % 64 == 0 && k % 64 == 0 &&
        (weight.cudaData == nullptr ||
         (!weight.cudaDataBorrowed && !weight.multiDeviceData &&
          (weight.cpuData != nullptr || !weight.numasData.empty())));
    if (!canRun) {
        if (warmup) {
            FixLegacyBackend(weight, device, "Marlin unsupported");
            return false;
        }
        throw std::runtime_error(
            "NVFP4 Marlin backend received an unsupported shape or dtype after warmup");
    }
    Cache *cache = GetCache(weight, m, k);
    if (cache == nullptr) {
        if (warmup) FixLegacyBackend(weight, device, "Marlin cache unavailable");
        else throw std::runtime_error("NVFP4 Marlin cache disappeared after warmup");
        return false;
    }
    half *cudaInput = static_cast<half *>(FastllmCudaPrepareInput(input));
    half *cudaOutput = static_cast<half *>(FastllmCudaPrepareOutput(output));
    if (cudaInput == nullptr || cudaOutput == nullptr) {
        if (warmup) {
            Release(*cache);
            FixLegacyBackend(weight, device, "Marlin input/output unavailable");
            return false;
        }
        throw std::runtime_error("NVFP4 Marlin input/output preparation failed after warmup");
    }
    bool ok = FastllmCudaMarlinHalfNVFP4Gemm(
        cudaInput, cache->weight, cache->scales, cache->globalScale,
        cudaOutput, n, k, m, cache->workspace);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    if (ok && warmup) ok = cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
    if (ok && warmup) {
        if (!weight.ReleaseCudaDataForRepackedWeight()) ok = false;
        else {
            cache->source = nullptr;
            SetBackendState(weight, device, BackendState::Marlin);
            if (TraceEnabled()) {
                std::fprintf(stderr,
                    "[fastllm][nvfp4] backend=marlin state=fixed device=%d name=%s\n",
                    device, weight.name.c_str());
            }
        }
    }
    if (!ok && warmup) {
        Release(*cache);
        FixLegacyBackend(weight, device, "Marlin warmup failed");
    } else if (!ok) {
        throw std::runtime_error(
            "NVFP4 Marlin backend failed after warmup; runtime fallback is disabled");
    }
    return ok;
}

void FastllmCudaReleaseNvfp4MarlinCache(const fastllm::Data *weight) {
    {
        std::lock_guard<std::mutex> guard(cacheMutex);
        bool hasCache = false;
        for (const auto &entry : caches) {
            if (entry.first.first == weight) { hasCache = true; break; }
        }
        if (hasCache) {
            int originalDevice = 0;
            cudaGetDevice(&originalDevice);
            for (auto it = caches.begin(); it != caches.end();) {
                if (it->first.first == weight) {
                    cudaSetDevice(it->first.second);
                    Release(it->second);
                    it = caches.erase(it);
                } else ++it;
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
    }
}
