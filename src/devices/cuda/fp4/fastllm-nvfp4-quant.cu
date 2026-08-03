/* Apache-2.0: kernels adapted from vLLM nvfp4_quant_kernels.cu and
 * activation_nvfp4_quant_fusion_kernels.cu. */
#include "fastllm-cuda.cuh"
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
        int roundedRows, int kTiles, float globalScale) {
    const int group = blockIdx.y * blockDim.x + threadIdx.x;
    const int groups = columns / 16;
    if (group >= groups) return;
    for (int row = blockIdx.x; row < roundedRows; row += gridDim.x) {
        fastllm_nvfp4::PackedVec<T> values;
        const bool valid = row < rows;
        fastllm_nvfp4::Load256OrZero(
            reinterpret_cast<fastllm_nvfp4::U32x8 &>(values),
            input + (size_t)row * columns + group * 16, valid);
        uint8_t *scale = fastllm_nvfp4::SwizzledScaleAddress(
            row, group, kTiles, outputScales);
        const uint64_t packed = fastllm_nvfp4::Quantize16(
            values, globalScale, scale);
        if (valid) {
            reinterpret_cast<uint64_t *>(output + (size_t)row * (columns / 2))[group] = packed;
        }
    }
}

template <typename T>
__global__ void FastllmSiluMulNvfp4QuantKernel(
        const T *__restrict__ input, uint8_t *__restrict__ output,
        uint8_t *__restrict__ outputScales, int rows, int hidden,
        int roundedRows, int kTiles, float globalScale) {
    const int group = blockIdx.y * blockDim.x + threadIdx.x;
    const int groups = hidden / 16;
    if (group >= groups) return;
    for (int row = blockIdx.x; row < roundedRows; row += gridDim.x) {
        fastllm_nvfp4::PackedVec<T> gate, up;
        const bool valid = row < rows;
        const T *rowInput = input + (size_t)row * hidden * 2;
        fastllm_nvfp4::Load256OrZero(
            reinterpret_cast<fastllm_nvfp4::U32x8 &>(gate),
            rowInput + group * 16, valid);
        fastllm_nvfp4::Load256OrZero(
            reinterpret_cast<fastllm_nvfp4::U32x8 &>(up),
            rowInput + hidden + group * 16, valid);
        auto activated = fastllm_nvfp4::SiluMul(gate, up);
        uint8_t *scale = fastllm_nvfp4::SwizzledScaleAddress(
            row, group, kTiles, outputScales);
        const uint64_t packed = fastllm_nvfp4::Quantize16(
            activated, globalScale, scale);
        if (valid) {
            reinterpret_cast<uint64_t *>(output + (size_t)row * (hidden / 2))[group] = packed;
        }
    }
}

__global__ void FastllmNvfp4Block16ToCutlassKernel(
        const uint8_t *__restrict__ source, uint8_t *__restrict__ packedWeight,
        uint8_t *__restrict__ weightScales, int rows, int columns,
        int sourceBytesPerRow, int kTiles) {
    const int group = blockIdx.y * blockDim.x + threadIdx.x;
    const int groups = columns / 16;
    if (group >= groups) return;
    for (int row = blockIdx.x; row < ((rows + 127) / 128) * 128; row += gridDim.x) {
        uint64_t packed = 0;
        uint8_t scaleByte = 0;
        if (row < rows) {
            const uint8_t *block = source + (size_t)row * sourceBytesPerRow
                                 + (size_t)group * (8 + sizeof(float));
            packed = *reinterpret_cast<const uint64_t *>(block);
            __nv_fp8_e4m3 scale(*reinterpret_cast<const float *>(block + 8));
            scaleByte = reinterpret_cast<const uint8_t &>(scale);
            reinterpret_cast<uint64_t *>(
                packedWeight + (size_t)row * (columns / 2))[group] = packed;
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
                        int rows, int columns, float globalScale,
                        cudaStream_t stream) {
    const int groups = columns / 16;
    const int threads = std::min(256, groups);
    const int roundedRows = (rows + 127) / 128 * 128;
    const int kTiles = (columns + 63) / 64;
    dim3 block(threads);
    dim3 grid(std::min(roundedRows, 128), (groups + threads - 1) / threads);
    FastllmNvfp4QuantKernel<T><<<grid, block, 0, stream>>>(
        static_cast<const T *>(input), output, scales, rows, columns,
        roundedRows, kTiles, globalScale);
    return cudaGetLastError() == cudaSuccess;
}

template <typename T>
static bool LaunchSiluMulQuant(const void *input, uint8_t *output, uint8_t *scales,
                               int rows, int hidden, float globalScale,
                               cudaStream_t stream) {
    const int groups = hidden / 16;
    const int threads = std::min(256, groups);
    const int roundedRows = (rows + 127) / 128 * 128;
    const int kTiles = (hidden + 63) / 64;
    dim3 block(threads);
    dim3 grid(std::min(roundedRows, 128), (groups + threads - 1) / threads);
    FastllmSiluMulNvfp4QuantKernel<T><<<grid, block, 0, stream>>>(
        static_cast<const T *>(input), output, scales, rows, hidden,
        roundedRows, kTiles, globalScale);
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
    if (!FastllmNvfp4RuntimeSupported() || input == nullptr || output == nullptr ||
        outputScales == nullptr || rows <= 0 || columns <= 0 || columns % 16 != 0 ||
        !std::isfinite(globalScale) || globalScale <= 0.0f) {
        return false;
    }
    cudaStream_t stream = streamPtr == nullptr ? 0 : static_cast<cudaStream_t>(streamPtr);
    if (inputType == fastllm::DataType::FLOAT16) {
        return LaunchQuant<half>(input, output, outputScales, rows, columns, globalScale, stream);
    }
    if (inputType == fastllm::DataType::BFLOAT16) {
        return LaunchQuant<__nv_bfloat16>(input, output, outputScales, rows, columns, globalScale, stream);
    }
    return false;
}

bool FastllmCudaSiluMulNvfp4Quantize(
        const void *input, fastllm::DataType inputType,
        uint8_t *output, uint8_t *outputScales,
        int rows, int hidden, float globalScale, void *streamPtr) {
    if (!FastllmNvfp4RuntimeSupported() || input == nullptr || output == nullptr ||
        outputScales == nullptr || rows <= 0 || hidden <= 0 || hidden % 16 != 0 ||
        !std::isfinite(globalScale) || globalScale <= 0.0f) {
        return false;
    }
    cudaStream_t stream = streamPtr == nullptr ? 0 : static_cast<cudaStream_t>(streamPtr);
    if (inputType == fastllm::DataType::FLOAT16) {
        return LaunchSiluMulQuant<half>(input, output, outputScales, rows, hidden, globalScale, stream);
    }
    if (inputType == fastllm::DataType::BFLOAT16) {
        return LaunchSiluMulQuant<__nv_bfloat16>(input, output, outputScales, rows, hidden, globalScale, stream);
    }
    return false;
}

bool FastllmCudaNvfp4Block16ToCutlass(
        const uint8_t *source, uint8_t *packedWeight, uint8_t *weightScales,
        int rows, int columns, void *streamPtr) {
    if (!FastllmNvfp4RuntimeSupported() || source == nullptr || packedWeight == nullptr ||
        weightScales == nullptr || rows <= 0 || columns <= 0 || columns % 32 != 0) {
        return false;
    }
    const int groups = columns / 16;
    const int threads = std::min(256, groups);
    const int sourceBytesPerRow = groups * (8 + sizeof(float));
    const int kTiles = (columns + 63) / 64;
    dim3 grid(std::min((rows + 127) / 128 * 128, 128),
              (groups + threads - 1) / threads);
    cudaStream_t stream = streamPtr == nullptr ? 0 : static_cast<cudaStream_t>(streamPtr);
    FastllmNvfp4Block16ToCutlassKernel<<<grid, threads, 0, stream>>>(
        source, packedWeight, weightScales, rows, columns,
        sourceBytesPerRow, kTiles);
    return cudaGetLastError() == cudaSuccess;
}
