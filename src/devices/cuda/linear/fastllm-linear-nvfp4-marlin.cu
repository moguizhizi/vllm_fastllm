/* Dense NVFP4 Marlin W4A16 integration, adapted from vLLM's
 * marlin_utils_fp4.py and quantization/marlin kernels (Apache-2.0). */
#include "fastllm-cuda.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <map>
#include <mutex>

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
    if (cache.source == weight.cudaData && cache.inFeatures == inFeatures &&
        cache.outFeatures == outFeatures && cache.weight != nullptr &&
        cache.scales != nullptr && cache.globalScale != nullptr &&
        cache.workspace != nullptr) return &cache;
    if (FastllmCudaGraphIsCapturing()) return nullptr;
    Release(cache);

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
    const int arch = RuntimeArch();
    if (arch < 75 || arch >= 100 || input.dataType != fastllm::DataType::FLOAT16 ||
        output.dataType != fastllm::DataType::FLOAT16 ||
        weight.dataType != fastllm::DataType::NVFP4_BLOCK_16 ||
        weight.blockM != 16 || !bias.dims.empty() || n < 2 ||
        m <= 0 || k <= 0 || m % 64 != 0 || k % 64 != 0) return false;
    Cache *cache = GetCache(weight, m, k);
    if (cache == nullptr) return false;
    half *cudaInput = static_cast<half *>(FastllmCudaPrepareInput(input));
    half *cudaOutput = static_cast<half *>(FastllmCudaPrepareOutput(output));
    if (cudaInput == nullptr || cudaOutput == nullptr) return false;
    bool ok = FastllmCudaMarlinHalfNVFP4Gemm(
        cudaInput, cache->weight, cache->scales, cache->globalScale,
        cudaOutput, n, k, m, cache->workspace);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return ok;
}

void FastllmCudaReleaseNvfp4MarlinCache(const fastllm::Data *weight) {
    std::lock_guard<std::mutex> guard(cacheMutex);
    for (auto it = caches.begin(); it != caches.end();) {
        if (it->first.first == weight) {
            Release(it->second);
            it = caches.erase(it);
        } else ++it;
    }
}
