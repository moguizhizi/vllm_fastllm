/* Apache-2.0: kernels adapted from vLLM nvfp4_quant_kernels.cu and
 * activation_nvfp4_quant_fusion_kernels.cu. */
#include "fastllm-cuda.cuh"
#define NVFP4_ENABLE_ELTS16 1
#include "fastllm-nvfp4-utils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace {

template <typename T>
__global__ void FastllmNvfp4QuantKernel(
        const T *__restrict__ input, uint8_t *__restrict__ output,
        uint8_t *__restrict__ outputScales, int rows, int columns,
        int paddedColumns,
        int roundedRows, int kTiles, float globalScale) {
    const int pack = blockIdx.y * blockDim.x + threadIdx.x;
    const int packs = paddedColumns / fastllm_nvfp4::kElementsPerThread;
    if (pack >= packs) return;
    for (int row = blockIdx.x; row < roundedRows; row += gridDim.x) {
        fastllm_nvfp4::PackedVec<T> values;
        const int element = pack * fastllm_nvfp4::kElementsPerThread;
        const bool valid = row < rows && element < columns;
        const T *source = valid ? input + (size_t)row * columns + element : input;
        fastllm_nvfp4::LoadOrZero(values, source, valid);
        uint8_t *scale = fastllm_nvfp4::QuantScaleAddress(
            row, pack, kTiles, outputScales);
        const auto packed = fastllm_nvfp4::Quantize(values, globalScale, scale);
        if (row < rows) {
            fastllm_nvfp4::StoreFp4(
                output + (size_t)row * (paddedColumns / 2) + element / 2, packed);
        }
    }
}

template <typename T>
__global__ void FastllmSiluMulNvfp4QuantKernel(
        const T *__restrict__ input, uint8_t *__restrict__ output,
        uint8_t *__restrict__ outputScales, int rows, int hidden,
        int paddedHidden,
        int roundedRows, int kTiles, float globalScale) {
    const int pack = blockIdx.y * blockDim.x + threadIdx.x;
    const int packs = paddedHidden / fastllm_nvfp4::kElementsPerThread;
    if (pack >= packs) return;
    for (int row = blockIdx.x; row < roundedRows; row += gridDim.x) {
        fastllm_nvfp4::PackedVec<T> gate, up;
        const int element = pack * fastllm_nvfp4::kElementsPerThread;
        const bool valid = row < rows && element < hidden;
        const T *rowInput = valid ? input + (size_t)row * hidden * 2 : input;
        fastllm_nvfp4::LoadOrZero(gate, rowInput + element, valid);
        fastllm_nvfp4::LoadOrZero(up, rowInput + hidden + element, valid);
        auto activated = fastllm_nvfp4::SiluMul(gate, up);
        uint8_t *scale = fastllm_nvfp4::QuantScaleAddress(
            row, pack, kTiles, outputScales);
        const auto packed = fastllm_nvfp4::Quantize(activated, globalScale, scale);
        if (row < rows) {
            fastllm_nvfp4::StoreFp4(
                output + (size_t)row * (paddedHidden / 2) + element / 2, packed);
        }
    }
}

// 将 FastLLM 的 FP4 权重和 checkpoint 原始 E4M3 group scale
// 重排为 CUTLASS 块缩放 GEMM 要求的两个布局。
// 这里只搬运并改变布局，不对 scale 做解码或二次量化。
__global__ void FastllmNvfp4Block16ToCutlassKernel(
        const uint8_t *__restrict__ source,
        const uint8_t *__restrict__ groupScales,
        uint8_t *__restrict__ packedWeight,
        uint8_t *__restrict__ weightScales, int rows, int columns,
        int paddedRows, int paddedColumns,
        int sourceBytesPerRow, int kTiles) {
    // K 维每相邻 16 个权重共享一个缩放因子。
    const int group = blockIdx.y * blockDim.x + threadIdx.x;
    const int groups = paddedColumns / 16;
    if (group >= groups) return;
    // CUTLASS 缩放因子布局要求行维度补齐到 128；遍历完整分块，
    // 同时将补齐权重矩阵之外的缩放因子初始化为零。
    for (int row = blockIdx.x; row < ((paddedRows + 127) / 128) * 128; row += gridDim.x) {
        uint64_t packed = 0;
        uint8_t scaleByte = 0;
        if (row < rows && group < columns / 16) {
            const uint8_t *block = source + (size_t)row * sourceBytesPerRow
                                 + (size_t)group * (8 + sizeof(float));
            // 旧 fallback 仍使用 12 字节 block；此处只读其中的 8 字节 FP4。
            // 相邻 block 间隔 12 字节，因此用两次 32-bit 读取避免非对齐 uint64_t 访问。
            const uint32_t packedLo = *reinterpret_cast<const uint32_t *>(block);
            const uint32_t packedHi = *reinterpret_cast<const uint32_t *>(block + 4);
            packed = uint64_t(packedLo) | (uint64_t(packedHi) << 32);
            // 原始 E4M3 字节直接进入 CUTLASS swizzled scale 布局。
            scaleByte = groupScales[(size_t)row * (columns / 16) + group];
        }
        // 补齐区域使用零 FP4 权重和零缩放因子。
        if (row < paddedRows) {
            reinterpret_cast<uint64_t *>(
                packedWeight + (size_t)row * (paddedColumns / 2))[group] = packed;
        }
        *fastllm_nvfp4::SwizzledScaleAddress(row, group, kTiles, weightScales) = scaleByte;
    }
}

static bool FastllmNvfp4RuntimeSupported() {
    int device = 0, major = 0, minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
        return false;
    }
    const int arch = major * 10 + minor;
    return arch >= 100 && arch < 130;
}

template <typename T>
static bool LaunchQuant(const void *input, uint8_t *output, uint8_t *scales,
                        int rows, int columns, int paddedColumns, float globalScale,
                        cudaStream_t stream) {
    const int packs = paddedColumns / fastllm_nvfp4::kElementsPerThread;
    const int threads = std::min(256, packs);
    const int roundedRows = (rows + 127) / 128 * 128;
    const int kTiles = (paddedColumns + 63) / 64;
    dim3 block(threads);
    dim3 grid(std::min(roundedRows, 128), (packs + threads - 1) / threads);
    FastllmNvfp4QuantKernel<T><<<grid, block, 0, stream>>>(
        static_cast<const T *>(input), output, scales, rows, columns,
        paddedColumns, roundedRows, kTiles, globalScale);
    return cudaGetLastError() == cudaSuccess;
}

template <typename T>
static bool LaunchSiluMulQuant(const void *input, uint8_t *output, uint8_t *scales,
                               int rows, int hidden, int paddedHidden, float globalScale,
                               cudaStream_t stream) {
    const int packs = paddedHidden / fastllm_nvfp4::kElementsPerThread;
    const int threads = std::min(256, packs);
    const int roundedRows = (rows + 127) / 128 * 128;
    const int kTiles = (paddedHidden + 63) / 64;
    dim3 block(threads);
    dim3 grid(std::min(roundedRows, 128), (packs + threads - 1) / threads);
    FastllmSiluMulNvfp4QuantKernel<T><<<grid, block, 0, stream>>>(
        static_cast<const T *>(input), output, scales, rows, hidden,
        paddedHidden, roundedRows, kTiles, globalScale);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

size_t FastllmCudaNvfp4SwizzledScaleBytes(int rows, int columns) {
    return rows > 0 && columns > 0 && columns % 16 == 0
        ? fastllm_nvfp4::SwizzledScaleBytes(rows, columns) : 0;
}

bool FastllmCudaNvfp4QuantizeActivation(
        const void *input, fastllm::DataType inputType,
        uint8_t *output, uint8_t *outputScales,
        int rows, int columns, float globalScale, void *streamPtr) {
    return FastllmCudaNvfp4QuantizeActivationPadded(
        input, inputType, output, outputScales, rows, columns, columns,
        globalScale, streamPtr);
}

bool FastllmCudaNvfp4QuantizeActivationPadded(
        const void *input, fastllm::DataType inputType,
        uint8_t *output, uint8_t *outputScales,
        int rows, int columns, int paddedColumns,
        float globalScale, void *streamPtr) {
    if (!FastllmNvfp4RuntimeSupported() || input == nullptr || output == nullptr ||
        outputScales == nullptr || rows <= 0 || columns <= 0 || columns % 16 != 0 ||
        paddedColumns < columns || paddedColumns % 32 != 0 ||
        !std::isfinite(globalScale) || globalScale <= 0.0f) {
        return false;
    }
    cudaStream_t stream = streamPtr == nullptr ? 0 : static_cast<cudaStream_t>(streamPtr);
    if (inputType == fastllm::DataType::FLOAT16) {
        return LaunchQuant<half>(input, output, outputScales, rows, columns,
                                 paddedColumns, globalScale, stream);
    }
    if (inputType == fastllm::DataType::BFLOAT16) {
        return LaunchQuant<__nv_bfloat16>(input, output, outputScales, rows, columns,
                                          paddedColumns, globalScale, stream);
    }
    return false;
}

bool FastllmCudaSiluMulNvfp4Quantize(
        const void *input, fastllm::DataType inputType,
        uint8_t *output, uint8_t *outputScales,
        int rows, int hidden, float globalScale, void *streamPtr) {
    return FastllmCudaSiluMulNvfp4QuantizePadded(
        input, inputType, output, outputScales, rows, hidden, hidden,
        globalScale, streamPtr);
}

bool FastllmCudaSiluMulNvfp4QuantizePadded(
        const void *input, fastllm::DataType inputType,
        uint8_t *output, uint8_t *outputScales,
        int rows, int hidden, int paddedHidden,
        float globalScale, void *streamPtr) {
    if (!FastllmNvfp4RuntimeSupported() || input == nullptr || output == nullptr ||
        outputScales == nullptr || rows <= 0 || hidden <= 0 || hidden % 16 != 0 ||
        paddedHidden < hidden || paddedHidden % 32 != 0 ||
        !std::isfinite(globalScale) || globalScale <= 0.0f) {
        return false;
    }
    cudaStream_t stream = streamPtr == nullptr ? 0 : static_cast<cudaStream_t>(streamPtr);
    if (inputType == fastllm::DataType::FLOAT16) {
        return LaunchSiluMulQuant<half>(input, output, outputScales, rows, hidden,
                                        paddedHidden, globalScale, stream);
    }
    if (inputType == fastllm::DataType::BFLOAT16) {
        return LaunchSiluMulQuant<__nv_bfloat16>(input, output, outputScales, rows, hidden,
                                                 paddedHidden, globalScale, stream);
    }
    return false;
}

bool FastllmCudaNvfp4Block16ToCutlass(
        const uint8_t *source, const uint8_t *groupScales,
        uint8_t *packedWeight, uint8_t *weightScales,
        int rows, int columns, void *streamPtr) {
    return FastllmCudaNvfp4Block16ToCutlassPadded(
        source, groupScales, packedWeight, weightScales,
        rows, columns, rows, columns,
        streamPtr);
}

bool FastllmCudaNvfp4Block16ToCutlassPadded(
        const uint8_t *source, const uint8_t *groupScales,
        uint8_t *packedWeight, uint8_t *weightScales,
        int rows, int columns, int paddedRows, int paddedColumns,
        void *streamPtr) {
    if (!FastllmNvfp4RuntimeSupported() || source == nullptr || packedWeight == nullptr ||
        groupScales == nullptr || weightScales == nullptr ||
        rows <= 0 || columns <= 0 || columns % 16 != 0) {
        return false;
    }
    if (paddedRows < rows || paddedColumns < columns || paddedRows % 32 != 0 ||
        paddedColumns % 32 != 0) return false;
    const int groups = paddedColumns / 16;
    const int threads = std::min(256, groups);
    const int sourceBytesPerRow = (columns / 16) * (8 + sizeof(float));
    const int kTiles = (paddedColumns + 63) / 64;
    dim3 grid(std::min((paddedRows + 127) / 128 * 128, 128),
              (groups + threads - 1) / threads);
    cudaStream_t stream = streamPtr == nullptr ? 0 : static_cast<cudaStream_t>(streamPtr);
    FastllmNvfp4Block16ToCutlassKernel<<<grid, threads, 0, stream>>>(
        source, groupScales, packedWeight, weightScales, rows, columns,
        paddedRows, paddedColumns, sourceBytesPerRow, kTiles);
    return cudaGetLastError() == cudaSuccess;
}
