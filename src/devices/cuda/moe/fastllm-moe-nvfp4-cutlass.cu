/* NVFP4 routed MoE integration around vLLM's SM100/SM120 grouped W4A4 CUTLASS GEMM. */
#include "fastllm-cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

template <typename T>
__global__ void GatherRows(const T *input, const int *routeRows, T *output,
                           int rows, int columns) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)rows * columns;
    if (index < count) {
        int row = int(index / columns);
        int col = int(index - (size_t)row * columns);
        output[index] = input[(size_t)routeRows[row] * columns + col];
    }
}

template <typename T>
__global__ void ReduceRoutes(const T *parts, const int *routePositions,
                             const float *routeScales, T *output,
                             int batch, int topk, int hidden) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)batch * hidden;
    if (index >= count) return;
    int token = int(index / hidden);
    int col = int(index - (size_t)token * hidden);
    float value = 0.0f;
    for (int slot = 0; slot < topk; ++slot) {
        int route = routePositions[token * topk + slot];
        if constexpr (std::is_same_v<T, half>) {
            value += __half2float(parts[(size_t)route * hidden + col]) * routeScales[route];
        } else {
            value += __bfloat162float(parts[(size_t)route * hidden + col]) * routeScales[route];
        }
    }
    if constexpr (std::is_same_v<T, half>) output[index] = __float2half_rn(value);
    else output[index] = __float2bfloat16_rn(value);
}

struct QuantBuffer {
    uint8_t *values = nullptr;
    uint8_t *scales = nullptr;
};

static void ReleaseBuffers(std::vector<QuantBuffer> &buffers) {
    for (auto &buffer : buffers) {
        FastllmCudaFree(buffer.values);
        FastllmCudaFree(buffer.scales);
    }
    buffers.clear();
}

static int MinBatch() {
    const char *value = std::getenv("FASTLLM_CUDA_MOE_NVFP4_W4A4_MIN_BATCH");
    // 与vLLM一致，不按M过滤后端；环境变量仅保留给显式性能调优。
    return value ? std::max(1, std::atoi(value)) : 1;
}

static bool Enabled() {
    const char *value = std::getenv("FASTLLM_CUDA_MOE_NVFP4_W4A4");
    return value == nullptr || value[0] == '\0' || value[0] != '0';
}

static bool StrictEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

template <typename T>
bool Run(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2,
         fastllm::Data &output, fastllm::Data **weights, int weightsBatch,
         const int *routeRows, const float *routeScales,
         const int *routePositions, const int *expertStarts,
         const int *expertCounts, int batch, int topk, int totalTasks,
         int hidden, int inter) {
    const fastllm::DataType dtype = std::is_same_v<T, half>
        ? fastllm::DataType::FLOAT16 : fastllm::DataType::BFLOAT16;
    const int experts = weightsBatch / 2 - 1;
    const int arch = FastllmCudaRuntimeArch();
    // vLLM 的 SM120 grouped kernel 固定输出 BF16；SM100 同时支持 FP16/BF16。
    if (input.dataType != dtype || input.dataDevice != fastllm::DataDevice::CUDA ||
        (arch >= 120 && dtype != fastllm::DataType::BFLOAT16) ||
        batch < MinBatch() || topk <= 0 || totalTasks != batch * topk ||
        experts <= 0 || experts > 256 || hidden % 32 != 0 || inter % 32 != 0 ||
        routeRows == nullptr || routeScales == nullptr || routePositions == nullptr ||
        expertStarts == nullptr || expertCounts == nullptr ||
        FastllmCudaGraphIsCapturing()) return false;

    for (int e = 0; e < experts; ++e) {
        fastllm::Data *gate = weights[(e + 1) * 2];
        fastllm::Data *down = weights[(e + 1) * 2 + 1];
        if (!gate || !down || gate->dataType != fastllm::DataType::NVFP4_BLOCK_16 ||
            down->dataType != fastllm::DataType::NVFP4_BLOCK_16 ||
            gate->blockM != 16 || down->blockM != 16 ||
            gate->dims != std::vector<int>({inter * 2, hidden}) ||
            down->dims != std::vector<int>({hidden, inter})) return false;
    }

    w1.dataDevice = fastllm::DataDevice::CUDA;
    w1.dataDeviceIds = input.dataDeviceIds;
    w1.dataType = dtype;
    w1.UpdateUnitSize();
    w1.Resize({totalTasks, inter * 2});
    w1.Allocate(false);
    w2.dataDevice = fastllm::DataDevice::CUDA;
    w2.dataDeviceIds = input.dataDeviceIds;
    w2.dataType = dtype;
    w2.UpdateUnitSize();
    w2.Resize({totalTasks, hidden});
    w2.Allocate(false);
    output.dataDevice = fastllm::DataDevice::CUDA;
    output.dataDeviceIds = input.dataDeviceIds;
    output.dataType = dtype;
    output.UpdateUnitSize();
    output.Resize(input.dims);
    output.Allocate(false);
    fastllm::Data gathered;
    gathered.dataDevice = fastllm::DataDevice::CUDA;
    gathered.dataDeviceIds = input.dataDeviceIds;
    gathered.dataType = dtype;
    gathered.UpdateUnitSize();
    gathered.Resize({totalTasks, hidden});
    gathered.Allocate(false);

    int *cudaRouteRows = static_cast<int *>(FastllmCudaMalloc(sizeof(int) * (size_t)totalTasks));
    int *cudaRoutePositions = static_cast<int *>(FastllmCudaMalloc(sizeof(int) * (size_t)totalTasks));
    float *cudaRouteScales = static_cast<float *>(FastllmCudaMalloc(sizeof(float) * (size_t)totalTasks));
    if (!cudaRouteRows || !cudaRoutePositions || !cudaRouteScales) {
        FastllmCudaFree(cudaRouteRows); FastllmCudaFree(cudaRoutePositions);
        FastllmCudaFree(cudaRouteScales); return false;
    }
    cudaStream_t stream = cudaStreamPerThread;
    bool ok = cudaMemcpyAsync(cudaRouteRows, routeRows, sizeof(int) * (size_t)totalTasks,
                              cudaMemcpyHostToDevice, stream) == cudaSuccess &&
              cudaMemcpyAsync(cudaRoutePositions, routePositions, sizeof(int) * (size_t)totalTasks,
                              cudaMemcpyHostToDevice, stream) == cudaSuccess &&
              cudaMemcpyAsync(cudaRouteScales, routeScales, sizeof(float) * (size_t)totalTasks,
                              cudaMemcpyHostToDevice, stream) == cudaSuccess;
    T *cudaInput = static_cast<T *>(FastllmCudaPrepareInput(input));
    if (ok && cudaInput) {
        size_t elements = (size_t)totalTasks * hidden;
        GatherRows<<<(elements + 255) / 256, 256, 0, stream>>>(
            cudaInput, cudaRouteRows, static_cast<T *>(gathered.cudaData), totalTasks, hidden);
        ok = cudaGetLastError() == cudaSuccess;
    } else ok = false;

    std::vector<QuantBuffer> gateBuffers, downBuffers;
    std::vector<const uint8_t *> a, b, scaleA, scaleB;
    std::vector<const float *> alpha;
    std::vector<void *> d;
    std::vector<int> rows;
    std::vector<int> activeExperts;
    for (int e = 0; ok && e < experts; ++e) if (expertCounts[e] > 0) {
        int count = expertCounts[e], start = expertStarts[e];
        QuantBuffer quant;
        quant.values = static_cast<uint8_t *>(FastllmCudaMalloc((size_t)count * hidden / 2));
        quant.scales = static_cast<uint8_t *>(FastllmCudaMalloc(
            FastllmCudaNvfp4SwizzledScaleBytes(count, hidden)));
        const uint8_t *weight = nullptr, *weightScales = nullptr;
        const float *weightAlpha = nullptr;
        fastllm::Data &gate = *weights[(e + 1) * 2];
        ok = quant.values && quant.scales &&
             FastllmCudaNvfp4QuantizeActivation(
                 static_cast<T *>(gathered.cudaData) + (size_t)start * hidden,
                 dtype, quant.values, quant.scales, count, hidden, 1.0f, (void *)stream) &&
             FastllmCudaPrepareNvfp4W4A4Weight(
                 gate, hidden, inter * 2, &weight, &weightScales, &weightAlpha);
        gateBuffers.push_back(quant);
        if (ok) {
            activeExperts.push_back(e); rows.push_back(count);
            a.push_back(quant.values); b.push_back(weight);
            scaleA.push_back(quant.scales); scaleB.push_back(weightScales);
            alpha.push_back(weightAlpha);
            d.push_back(static_cast<T *>(w1.cudaData) + (size_t)start * inter * 2);
        }
    }
    if (ok) {
        ok = arch >= 120
            ? FastllmCudaNvfp4GroupedGemmSm120(
                  a.data(), b.data(), scaleA.data(), scaleB.data(), alpha.data(), d.data(),
                  rows.data(), (int)rows.size(), inter * 2, hidden, dtype, (void *)stream)
            : FastllmCudaNvfp4GroupedGemmSm100(
                  a.data(), b.data(), scaleA.data(), scaleB.data(), alpha.data(), d.data(),
                  rows.data(), (int)rows.size(), inter * 2, hidden, dtype, (void *)stream);
    }

    a.clear(); b.clear(); scaleA.clear(); scaleB.clear(); alpha.clear(); d.clear(); rows.clear();
    for (size_t i = 0; ok && i < activeExperts.size(); ++i) {
        int e = activeExperts[i], count = expertCounts[e], start = expertStarts[e];
        QuantBuffer quant;
        quant.values = static_cast<uint8_t *>(FastllmCudaMalloc((size_t)count * inter / 2));
        quant.scales = static_cast<uint8_t *>(FastllmCudaMalloc(
            FastllmCudaNvfp4SwizzledScaleBytes(count, inter)));
        const uint8_t *weight = nullptr, *weightScales = nullptr;
        const float *weightAlpha = nullptr;
        fastllm::Data &down = *weights[(e + 1) * 2 + 1];
        ok = quant.values && quant.scales &&
             FastllmCudaSiluMulNvfp4Quantize(
                 static_cast<T *>(w1.cudaData) + (size_t)start * inter * 2,
                 dtype, quant.values, quant.scales, count, inter, 1.0f, (void *)stream) &&
             FastllmCudaPrepareNvfp4W4A4Weight(
                 down, inter, hidden, &weight, &weightScales, &weightAlpha);
        downBuffers.push_back(quant);
        if (ok) {
            rows.push_back(count); a.push_back(quant.values); b.push_back(weight);
            scaleA.push_back(quant.scales); scaleB.push_back(weightScales);
            alpha.push_back(weightAlpha);
            d.push_back(static_cast<T *>(w2.cudaData) + (size_t)start * hidden);
        }
    }
    if (ok) {
        ok = arch >= 120
            ? FastllmCudaNvfp4GroupedGemmSm120(
                  a.data(), b.data(), scaleA.data(), scaleB.data(), alpha.data(), d.data(),
                  rows.data(), (int)rows.size(), hidden, inter, dtype, (void *)stream)
            : FastllmCudaNvfp4GroupedGemmSm100(
                  a.data(), b.data(), scaleA.data(), scaleB.data(), alpha.data(), d.data(),
                  rows.data(), (int)rows.size(), hidden, inter, dtype, (void *)stream);
    }
    if (ok) {
        size_t elements = (size_t)batch * hidden;
        ReduceRoutes<<<(elements + 255) / 256, 256, 0, stream>>>(
            static_cast<const T *>(w2.cudaData), cudaRoutePositions,
            cudaRouteScales, static_cast<T *>(output.cudaData), batch, topk, hidden);
        ok = cudaGetLastError() == cudaSuccess;
    }

    ReleaseBuffers(gateBuffers);
    ReleaseBuffers(downBuffers);
    FastllmCudaFree(cudaRouteRows); FastllmCudaFree(cudaRoutePositions);
    FastllmCudaFree(cudaRouteScales);
    if (cudaInput) FastllmCudaFinishInput(input, cudaInput);
    return ok;
}
} // namespace

bool FastllmCudaMergeMoeNvfp4W4A4Grouped(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2,
        fastllm::Data &output, fastllm::Data **weights, int weightsBatch,
        const int *routeRows, const float *routeScales,
        const int *routePositions, const int *expertStarts,
        const int *expertCounts, int batch, int topk, int totalTasks,
        int hidden, int inter) {
    int arch = FastllmCudaRuntimeArch();
    if (!Enabled() || arch < 100 || arch >= 130 || weights == nullptr || weightsBatch < 4 ||
        (weightsBatch & 1)) return false;
    bool ok = false;
    if (input.dataType == fastllm::DataType::FLOAT16)
        ok = Run<half>(input, w1, w2, output, weights, weightsBatch,
            routeRows, routeScales, routePositions, expertStarts, expertCounts,
            batch, topk, totalTasks, hidden, inter);
    else if (input.dataType == fastllm::DataType::BFLOAT16)
        ok = Run<__nv_bfloat16>(input, w1, w2, output, weights, weightsBatch,
            routeRows, routeScales, routePositions, expertStarts, expertCounts,
            batch, topk, totalTasks, hidden, inter);
    if (!ok) {
        for (int i = 2; i < weightsBatch; ++i) {
            fastllm::Data *weight = weights[i];
            if (weight != nullptr && weight->dataType == fastllm::DataType::NVFP4_BLOCK_16 &&
                weight->cudaData == nullptr &&
                !weight->RestoreCudaDataForRepackedWeight()) {
                throw ("NVFP4 grouped MoE fallback cannot restore an original weight.");
            }
        }
        // strict用于算子验证和性能测试：禁止Legacy静默兜底，确保测到的
        // 一定是grouped CUTLASS，而不是“测试通过但实际跑了fallback”。
        if (StrictEnabled()) {
            throw std::runtime_error(
                "strict NVFP4 grouped MoE requires the CUTLASS backend");
        }
    }
    const char *trace = std::getenv("FASTLLM_CUDA_NVFP4_TRACE");
    if (trace && trace[0] != '\0' && trace[0] != '0') {
        std::fprintf(stderr,
            "[fastllm][nvfp4] path=%s batch=%d topk=%d hidden=%d inter=%d sm=%d\n",
            ok ? "grouped-moe-w4a4-cutlass" : "grouped-moe-fallback",
            batch, topk, hidden, inter, arch);
    }
    return ok;
}
