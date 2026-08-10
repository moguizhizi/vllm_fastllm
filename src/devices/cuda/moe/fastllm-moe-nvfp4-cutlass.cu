/* NVFP4 routed MoE integration around vLLM's SM100/SM120 grouped W4A4 CUTLASS GEMM. */
#include "fastllm-cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
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

/**
 * 对grouped MoE第一层输出执行GeGLU，供后续独立NVFP4量化使用。
 *
 * vLLM仅为SwiGLU提供“激活+FP4量化”融合kernel；GeGLU走普通门控激活
 * 后再独立量化。本kernel实现其中的普通GeGLU阶段，并与grouped GEMM使用
 * 同一CUDA流，避免借用Data高级接口时改变流边界或产生额外同步。
 *
 * 输入每行按[gate, up]连续存放，数学语义为：
 * output[row,col] = GELU(input[row,col]) * input[row,hidden+col]。
 * GELU采用FastLLM现有GeGLU相同的erf精确形式。
 *
 * @tparam T      输入输出类型，仅支持half或__nv_bfloat16。
 * @param input   第一层输出，逻辑形状为[rows, 2 * hidden]。
 * @param output  GeGLU结果，逻辑形状为[rows, hidden]。
 * @param rows    当前expert分到的路由行数。
 * @param hidden  GeGLU输出宽度。
 */
template <typename T>
__global__ void GegluRows(const T *input, T *output, int rows, int hidden) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = (size_t)rows * hidden;
    if (index >= count) return;
    const int row = int(index / hidden);
    const int column = int(index - (size_t)row * hidden);
    const size_t inputOffset = (size_t)row * hidden * 2 + column;
    float gate, up;
    if constexpr (std::is_same_v<T, half>) {
        gate = __half2float(input[inputOffset]);
        up = __half2float(input[inputOffset + hidden]);
        output[index] = __float2half_rn(
            gate * 0.5f * (1.0f + erff(gate / 1.41421356237f)) * up);
    } else {
        gate = __bfloat162float(input[inputOffset]);
        up = __bfloat162float(input[inputOffset + hidden]);
        output[index] = __float2bfloat16_rn(
            gate * 0.5f * (1.0f + erff(gate / 1.41421356237f)) * up);
    }
}

/**
 * Grouped NVFP4 MoE在一张GPU、一个调用线程上的持久临时缓冲区。
 *
 * 路由表、gather结果以及两层激活量化结果均按需扩容并跨调用复用，避免
 * 每个token步、每个活跃expert重复cudaMalloc/cudaFree。scratch使用
 * thread_local隔离CPU调用线程；同一线程再按GPU编号隔离，和
 * cudaStreamPerThread的生命周期保持一致。
 */
struct MoeScratch {
    int *routeRows = nullptr;
    int *routePositions = nullptr;
    float *routeScales = nullptr;
    uint8_t *gathered = nullptr;
    uint8_t *gateValues = nullptr;
    uint8_t *gateScales = nullptr;
    uint8_t *downValues = nullptr;
    uint8_t *downScales = nullptr;
    uint8_t *activated = nullptr;
    size_t routeRowsCapacity = 0;
    size_t routePositionsCapacity = 0;
    size_t routeScalesCapacity = 0;
    size_t gatheredCapacity = 0;
    size_t gateValueCapacity = 0;
    size_t gateScaleCapacity = 0;
    size_t downValueCapacity = 0;
    size_t downScaleCapacity = 0;
    size_t activatedCapacity = 0;
};

static thread_local std::map<int, MoeScratch> moeScratch;

template <typename T>
static bool EnsureScratchBuffer(T *&buffer, size_t &capacity, size_t bytes) {
    if (buffer != nullptr && capacity >= bytes) return true;
    T *replacement = static_cast<T *>(FastllmCudaMalloc(bytes));
    if (replacement == nullptr) return false;
    if (buffer != nullptr) FastllmCudaFree(buffer);
    buffer = replacement;
    capacity = bytes;
    return true;
}

/** 获取并扩容当前GPU/线程的MoE scratch；正式计算阶段不再释放。 */
static MoeScratch *GetMoeScratch(size_t routeCount, size_t gatheredBytes,
                                 size_t gateValueBytes, size_t gateScaleBytes,
                                 size_t downValueBytes, size_t downScaleBytes,
                                 size_t activatedBytes) {
    const int device = FastllmCudaGetDevice();
    MoeScratch &scratch = moeScratch[device];
    const size_t routeBytes = routeCount * sizeof(int);
    if (!EnsureScratchBuffer(scratch.routeRows, scratch.routeRowsCapacity, routeBytes) ||
        !EnsureScratchBuffer(scratch.routePositions, scratch.routePositionsCapacity, routeBytes) ||
        !EnsureScratchBuffer(scratch.routeScales, scratch.routeScalesCapacity,
                             routeCount * sizeof(float)) ||
        !EnsureScratchBuffer(scratch.gathered, scratch.gatheredCapacity, gatheredBytes) ||
        !EnsureScratchBuffer(scratch.gateValues, scratch.gateValueCapacity, gateValueBytes) ||
        !EnsureScratchBuffer(scratch.gateScales, scratch.gateScaleCapacity, gateScaleBytes) ||
        !EnsureScratchBuffer(scratch.downValues, scratch.downValueCapacity, downValueBytes) ||
        !EnsureScratchBuffer(scratch.downScales, scratch.downScaleCapacity, downScaleBytes) ||
        !EnsureScratchBuffer(scratch.activated, scratch.activatedCapacity, activatedBytes)) {
        return nullptr;
    }
    return &scratch;
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
         int hidden, int inter, fastllm::MoeGateType gateType) {
    const fastllm::DataType dtype = std::is_same_v<T, half>
        ? fastllm::DataType::FLOAT16 : fastllm::DataType::BFLOAT16;
    const int experts = weightsBatch / 2 - 1;
    const int arch = FastllmCudaRuntimeArch();
    // 先验证架构、输入类型、路由规模和张量布局；任一条件不满足时交由
    // 外层决定是否进入Legacy，不在本函数中改变后端生命周期。
    // vLLM 的 SM120 grouped kernel 固定输出 BF16；SM100 同时支持 FP16/BF16。
    if (input.dataType != dtype || input.dataDevice != fastllm::DataDevice::CUDA ||
        (arch >= 120 && dtype != fastllm::DataType::BFLOAT16) ||
        (gateType != fastllm::MoeGateSwiglu &&
         gateType != fastllm::MoeGateGeglu) ||
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

    // 为两次grouped GEMM准备正式推理使用的中间张量：w1保存gate/up，
    // w2保存down结果；output保持与输入相同的batch/token布局。
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

    // 每个expert的scale布局都会把自身M补齐到128，因此不能只按totalTasks
    // 计算一块scale大小；先计算各expert在持久scale缓冲区中的独立偏移。
    std::vector<int> activeExperts;
    std::vector<size_t> gateScaleOffsets, downScaleOffsets;
    activeExperts.reserve(experts);
    gateScaleOffsets.reserve(experts);
    downScaleOffsets.reserve(experts);
    size_t gateScaleBytes = 0, downScaleBytes = 0;
    for (int e = 0; e < experts; ++e) if (expertCounts[e] > 0) {
        activeExperts.push_back(e);
        gateScaleOffsets.push_back(gateScaleBytes);
        downScaleOffsets.push_back(downScaleBytes);
        gateScaleBytes += FastllmCudaNvfp4SwizzledScaleBytes(expertCounts[e], hidden);
        downScaleBytes += FastllmCudaNvfp4SwizzledScaleBytes(expertCounts[e], inter);
    }
    if (activeExperts.empty()) return false;

    MoeScratch *scratch = GetMoeScratch(
        totalTasks, (size_t)totalTasks * hidden * sizeof(T),
        (size_t)totalTasks * hidden / 2, gateScaleBytes,
        (size_t)totalTasks * inter / 2, downScaleBytes,
        gateType == fastllm::MoeGateGeglu
            ? (size_t)totalTasks * inter * sizeof(T) : sizeof(T));
    if (scratch == nullptr) return false;

    // 路由元数据由CPU按expert连续排列；异步上传后，gather kernel把原始
    // token行重排为grouped GEMM要求的expert分组输入。
    int *cudaRouteRows = scratch->routeRows;
    int *cudaRoutePositions = scratch->routePositions;
    float *cudaRouteScales = scratch->routeScales;
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
            cudaInput, cudaRouteRows, reinterpret_cast<T *>(scratch->gathered),
            totalTasks, hidden);
        ok = cudaGetLastError() == cudaSuccess;
    } else ok = false;

    // 第一层：逐个活跃expert量化输入、取得对应CUTLASS重排权重，然后
    // 将各expert的指针和实际行数组成一次grouped W4A4 GEMM参数。
    std::vector<const uint8_t *> a, b, scaleA, scaleB;
    std::vector<const float *> alpha;
    std::vector<void *> d;
    std::vector<int> rows;
    a.reserve(activeExperts.size()); b.reserve(activeExperts.size());
    scaleA.reserve(activeExperts.size()); scaleB.reserve(activeExperts.size());
    alpha.reserve(activeExperts.size()); d.reserve(activeExperts.size());
    rows.reserve(activeExperts.size());
    for (size_t i = 0; ok && i < activeExperts.size(); ++i) {
        const int e = activeExperts[i];
        int count = expertCounts[e], start = expertStarts[e];
        uint8_t *quantValues = scratch->gateValues + (size_t)start * hidden / 2;
        uint8_t *quantScales = scratch->gateScales + gateScaleOffsets[i];
        const uint8_t *weight = nullptr, *weightScales = nullptr;
        const float *weightAlpha = nullptr;
        fastllm::Data &gate = *weights[(e + 1) * 2];
        ok = FastllmCudaNvfp4QuantizeActivation(
                 reinterpret_cast<T *>(scratch->gathered) + (size_t)start * hidden,
                 dtype, quantValues, quantScales, count, hidden, 1.0f, (void *)stream) &&
             FastllmCudaPrepareNvfp4W4A4Weight(
                 gate, hidden, inter * 2, &weight, &weightScales, &weightAlpha);
        if (ok) {
            rows.push_back(count);
            a.push_back(quantValues); b.push_back(weight);
            scaleA.push_back(quantScales); scaleB.push_back(weightScales);
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

    // 第二层：复用同一组参数容器和持久scratch。SwiGLU走融合激活量化，
    // GeGLU先计算门控激活再独立量化，最终调用down grouped W4A4 GEMM。
    a.clear(); b.clear(); scaleA.clear(); scaleB.clear(); alpha.clear(); d.clear(); rows.clear();
    for (size_t i = 0; ok && i < activeExperts.size(); ++i) {
        int e = activeExperts[i], count = expertCounts[e], start = expertStarts[e];
        uint8_t *quantValues = scratch->downValues + (size_t)start * inter / 2;
        uint8_t *quantScales = scratch->downScales + downScaleOffsets[i];
        const uint8_t *weight = nullptr, *weightScales = nullptr;
        const float *weightAlpha = nullptr;
        fastllm::Data &down = *weights[(e + 1) * 2 + 1];
        const T *gateInput = static_cast<T *>(w1.cudaData) +
                             (size_t)start * inter * 2;
        if (gateType == fastllm::MoeGateSwiglu) {
            // SwiGLU沿用vLLM对应的激活+FP4量化融合路径。
            ok = FastllmCudaSiluMulNvfp4Quantize(
                gateInput, dtype, quantValues, quantScales,
                count, inter, 1.0f, (void *)stream);
        } else {
            // vLLM的GeGLU路径并不融合量化：先完成门控激活，再调用普通
            // NVFP4动态量化。activated按路由顺序复用持久scratch。
            T *activated = reinterpret_cast<T *>(scratch->activated) +
                           (size_t)start * inter;
            const size_t elements = (size_t)count * inter;
            GegluRows<<<(elements + 255) / 256, 256, 0, stream>>>(
                gateInput, activated, count, inter);
            ok = cudaGetLastError() == cudaSuccess &&
                 FastllmCudaNvfp4QuantizeActivation(
                     activated, dtype, quantValues, quantScales,
                     count, inter, 1.0f, (void *)stream);
        }
        ok = ok && FastllmCudaPrepareNvfp4W4A4Weight(
            down, inter, hidden, &weight, &weightScales, &weightAlpha);
        if (ok) {
            rows.push_back(count); a.push_back(quantValues); b.push_back(weight);
            scaleA.push_back(quantScales); scaleB.push_back(weightScales);
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
        // grouped GEMM输出按路由顺序排列；根据原始token/topk位置和路由
        // 分数做加权归并，恢复为output[batch, hidden]。
        size_t elements = (size_t)batch * hidden;
        ReduceRoutes<<<(elements + 255) / 256, 256, 0, stream>>>(
            static_cast<const T *>(w2.cudaData), cudaRoutePositions,
            cudaRouteScales, static_cast<T *>(output.cudaData), batch, topk, hidden);
        ok = cudaGetLastError() == cudaSuccess;
    }

    if (cudaInput) FastllmCudaFinishInput(input, cudaInput);
    return ok;
}
} // namespace

/**
 * 执行一次NVFP4 W4A4 grouped MoE计算。
 *
 * 本函数是CUDA grouped W4A4 MoE的公开入口，负责检查运行时开关和GPU
 * 架构，并按输入类型分派到内部Run实现。内部流程依次完成路由行gather、
 * gate/up激活动态NVFP4量化、第一层grouped GEMM、SwiGLU或GeGLU门控、
 * down激活动态NVFP4量化、第二层grouped GEMM，以及top-k路由加权归并。
 * 本函数不构造路由表；调用方必须提前按expert连续排列路由元数据。
 *
 * 数学语义为：对每个token选中的top-k expert分别计算
 * down(activation(gate_up(input)))，再乘以对应routeScales并求和。
 * gate/up权重逻辑形状为[2 * inter, hidden]，down权重逻辑形状为
 * [hidden, inter]，两者均为NVFP4_BLOCK_16布局。
 *
 * 执行失败时，本函数恢复已经释放原始CUDA表示的NVFP4权重，使调用方
 * 可以安全进入Legacy路径；STRICT模式下直接抛出异常，禁止静默降级。
 *
 * @param input           FP16或BF16输入，逻辑形状为[batch, hidden]；SM120
 *                        当前仅接受BF16。
 * @param w1              第一层grouped GEMM临时输出，函数内调整为
 *                        [totalTasks, 2 * inter]。
 * @param w2              第二层grouped GEMM临时输出，函数内调整为
 *                        [totalTasks, hidden]。
 * @param output           最终MoE输出，逻辑形状与input一致。
 * @param weights          expert权重指针数组；第0对为可选shared expert，
 *                        routed expert e使用第e+1对gate/up和down权重。
 * @param weightsBatch     weights数组中的Data指针数量，必须为偶数且至少为4。
 * @param routeRows        按expert分组后的任务到原始input行号的映射，长度为
 *                        totalTasks。
 * @param routeScales      每条路由的top-k权重，按分组后任务顺序排列，长度为
 *                        totalTasks。
 * @param routePositions   原始[token, topk]位置到分组后任务位置的映射，长度
 *                        为batch * topk。
 * @param expertStarts     每个expert在分组任务数组中的起始位置。
 * @param expertCounts     每个expert本次参与计算的任务数量。
 * @param batch            输入token行数，即grouped MoE的M总规模。
 * @param topk             每个token选中的routed expert数量。
 * @param totalTasks       总路由任务数，必须等于batch * topk。
 * @param hidden           模型隐藏维度，必须按32对齐。
 * @param inter            单个expert的中间维度，必须按32对齐。
 * @param gateType         门控激活类型，当前支持MoeGateSwiglu和
 *                        MoeGateGeglu。
 * @return true表示两次grouped W4A4 GEMM和路由归并均成功；false表示条件
 *         不满足或执行失败，具体路径可通过FASTLLM_CUDA_NVFP4_TRACE查看。
 */
bool FastllmCudaMergeMoeNvfp4W4A4Grouped(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2,
        fastllm::Data &output, fastllm::Data **weights, int weightsBatch,
        const int *routeRows, const float *routeScales,
        const int *routePositions, const int *expertStarts,
        const int *expertCounts, int batch, int topk, int totalTasks,
        int hidden, int inter, fastllm::MoeGateType gateType) {
    int arch = FastllmCudaRuntimeArch();
    if (!Enabled() || arch < 100 || arch >= 130 || weights == nullptr || weightsBatch < 4 ||
        (weightsBatch & 1)) return false;
    bool ok = false;
    if (input.dataType == fastllm::DataType::FLOAT16)
        ok = Run<half>(input, w1, w2, output, weights, weightsBatch,
            routeRows, routeScales, routePositions, expertStarts, expertCounts,
            batch, topk, totalTasks, hidden, inter, gateType);
    else if (input.dataType == fastllm::DataType::BFLOAT16)
        ok = Run<__nv_bfloat16>(input, w1, w2, output, weights, weightsBatch,
            routeRows, routeScales, routePositions, expertStarts, expertCounts,
            batch, topk, totalTasks, hidden, inter, gateType);
    if (!ok) {
        // CUTLASS路径失败后恢复原始权重表示，保证后续Legacy计算不会读取
        // 已释放的cudaData；STRICT模式用于测试时阻止这种回退。
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
