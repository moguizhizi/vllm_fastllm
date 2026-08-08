/* Apache-2.0: 忠实移植自 vLLM nvfp4_blockwise_moe_kernel.cu 的 SM120 路径。 */
#include "fastllm-cuda.cuh"

#if defined(FASTLLM_ENABLE_CUTLASS_NVFP4_SM120)
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include <algorithm>
#include <climits>
#include <map>
#include <vector>

// NVCC 12.8会为__global__模板生成host stub。这里必须使用具名namespace，
// 否则stub中的匿名namespace名称可能与CUTE头文件内部的匿名namespace冲突。
namespace fastllm_nvfp4_moe_sm120 {
using namespace cute;

struct GroupedConfigSm120 {
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;
    using ElementType = cutlass::float_e2m1_t;
    using ElementSFType = cutlass::float_ue4m3_t;
    using ElementA = cutlass::nv_float4_t<ElementType>;
    using ElementB = cutlass::nv_float4_t<ElementType>;
    // 与 vLLM 一致：SM120 grouped NVFP4 当前固定输出 BF16。
    using ElementC = cutlass::bfloat16_t;
    using ElementD = ElementC;
    using Accumulator = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = LayoutC;
    static constexpr int AlignmentA = 32;
    static constexpr int AlignmentB = 32;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    using Arch = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
    using Cluster = Shape<_1, _1, _1>;
    using Tile = Shape<_128, _128, _128>;
    using Fusion = cutlass::epilogue::fusion::LinearCombination<
        ElementD, Accumulator, ElementC, Accumulator>;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch, OperatorClass, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator, Accumulator, ElementC, LayoutC *, AlignmentC,
        ElementD, LayoutD *, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, OperatorClass, ElementA, LayoutA *, AlignmentA,
        ElementB, LayoutB *, AlignmentB, Accumulator, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Mainloop, Epilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    static_assert(
        cute::is_base_of_v<
            cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative,
            typename Kernel::CollectiveMainloop::DispatchPolicy::Schedule>,
        "SM120 NVFP4 grouped GEMM requires the cooperative pointer-array schedule");
};

template <typename StrideA, typename StrideB, typename StrideD,
          typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
__global__ void SetupMetadataSm120(int32_t *shapes, StrideA *strideA,
                                   StrideB *strideB, StrideD *strideD,
                                   LayoutSFA *layoutA, LayoutSFB *layoutB,
                                   const int *rows, int groups, int n, int k) {
    int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    shapes[group * 3] = rows[group];
    shapes[group * 3 + 1] = n;
    shapes[group * 3 + 2] = k;
    reinterpret_cast<int64_t *>(strideA)[group] = k;
    reinterpret_cast<int64_t *>(strideB)[group] = k;
    reinterpret_cast<int64_t *>(strideD)[group] = n;
    layoutA[group] = ScaleConfig::tile_atom_to_shape_SFA(
        cute::make_shape(rows[group], n, k, 1));
    layoutB[group] = ScaleConfig::tile_atom_to_shape_SFB(
        cute::make_shape(rows[group], n, k, 1));
}

/** SM120 grouped GEMM复用的device metadata和CUTLASS workspace。 */
struct GroupedScratchSm120 {
    uint8_t *metadata = nullptr;
    void *workspace = nullptr;
    size_t metadataCapacity = 0;
    size_t workspaceCapacity = 0;
};

static thread_local std::map<int, GroupedScratchSm120> groupedScratchSm120;

template <typename T>
static bool EnsureBufferSm120(T *&buffer, size_t &capacity, size_t bytes) {
    if (bytes == 0 || (buffer != nullptr && capacity >= bytes)) return true;
    T *replacement = static_cast<T *>(FastllmCudaMalloc(bytes));
    if (replacement == nullptr) return false;
    if (buffer != nullptr) FastllmCudaFree(buffer);
    buffer = replacement;
    capacity = bytes;
    return true;
}

static size_t AlignMetadataSm120(size_t offset, size_t alignment) {
    return (offset + alignment - 1) / alignment * alignment;
}

/** 获取当前GPU/线程的持久metadata；thread_local防止per-thread stream互相覆盖。 */
static GroupedScratchSm120 *GetGroupedScratchSm120(size_t metadataBytes) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return nullptr;
    GroupedScratchSm120 &scratch = groupedScratchSm120[device];
    return EnsureBufferSm120(
        scratch.metadata, scratch.metadataCapacity, metadataBytes) ? &scratch : nullptr;
}

bool RunGroupedSm120(const uint8_t *const *hostA, const uint8_t *const *hostB,
                     const uint8_t *const *hostScaleA,
                     const uint8_t *const *hostScaleB,
                     const float *const *hostAlpha, void *const *hostD,
                     const int *hostRows, int groups, int n, int k,
                     cudaStream_t stream) {
    using Config = GroupedConfigSm120;
    using Gemm = Config::Gemm;
    using Kernel = Config::Kernel;
    using ShapeType = Config::ProblemShape::UnderlyingProblemShape;
    using StrideA = Kernel::InternalStrideA;
    using StrideB = Kernel::InternalStrideB;
    using StrideD = Kernel::InternalStrideD;
    using LayoutSFA = Kernel::CollectiveMainloop::InternalLayoutSFA;
    using LayoutSFB = Kernel::CollectiveMainloop::InternalLayoutSFB;
    using ScaleConfig = Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    // 所有device metadata放入一块持久buffer，只在容量增长时重新分配。
    size_t metadataBytes = 0;
#define RESERVE_META_SM120(type, count) \
    do { metadataBytes = AlignMetadataSm120(metadataBytes, alignof(type)); \
         metadataBytes += sizeof(type) * (size_t)(count); } while (0)
    RESERVE_META_SM120(const Config::ElementType *, groups); // A
    RESERVE_META_SM120(const Config::ElementType *, groups); // B
    RESERVE_META_SM120(const Config::ElementSFType *, groups); // scale A
    RESERVE_META_SM120(const Config::ElementSFType *, groups); // scale B
    RESERVE_META_SM120(const float *, groups); // alpha
    RESERVE_META_SM120(Config::ElementD *, groups); // D
    RESERVE_META_SM120(int, groups); // rows
    RESERVE_META_SM120(int32_t, (size_t)groups * 3); // shapes
    RESERVE_META_SM120(StrideA, groups);
    RESERVE_META_SM120(StrideB, groups);
    RESERVE_META_SM120(StrideD, groups);
    RESERVE_META_SM120(LayoutSFA, groups);
    RESERVE_META_SM120(LayoutSFB, groups);
#undef RESERVE_META_SM120
    GroupedScratchSm120 *scratch = GetGroupedScratchSm120(metadataBytes);
    if (scratch == nullptr) return false;
    size_t metadataOffset = 0;
#define TAKE_META_SM120(name, type, count) \
    metadataOffset = AlignMetadataSm120(metadataOffset, alignof(type)); \
    auto name = reinterpret_cast<type *>(scratch->metadata + metadataOffset); \
    metadataOffset += sizeof(type) * (size_t)(count)
    TAKE_META_SM120(a, const Config::ElementType *, groups);
    TAKE_META_SM120(b, const Config::ElementType *, groups);
    TAKE_META_SM120(sa, const Config::ElementSFType *, groups);
    TAKE_META_SM120(sb, const Config::ElementSFType *, groups);
    TAKE_META_SM120(alpha, const float *, groups);
    TAKE_META_SM120(d, Config::ElementD *, groups);
    TAKE_META_SM120(rows, int, groups);
    TAKE_META_SM120(shapes, int32_t, (size_t)groups * 3);
    TAKE_META_SM120(strideA, StrideA, groups);
    TAKE_META_SM120(strideB, StrideB, groups);
    TAKE_META_SM120(strideD, StrideD, groups);
    TAKE_META_SM120(layoutA, LayoutSFA, groups);
    TAKE_META_SM120(layoutB, LayoutSFB, groups);
#undef TAKE_META_SM120
    std::vector<const Config::ElementType *> castA(groups), castB(groups);
    std::vector<const Config::ElementSFType *> castSA(groups), castSB(groups);
    std::vector<Config::ElementD *> castD(groups);
    for (int i = 0; i < groups; ++i) {
        castA[i] = reinterpret_cast<const Config::ElementType *>(hostA[i]);
        castB[i] = reinterpret_cast<const Config::ElementType *>(hostB[i]);
        castSA[i] = reinterpret_cast<const Config::ElementSFType *>(hostScaleA[i]);
        castSB[i] = reinterpret_cast<const Config::ElementSFType *>(hostScaleB[i]);
        castD[i] = static_cast<Config::ElementD *>(hostD[i]);
    }
    bool ok = true;
#define COPY_META(dst, src, bytes) \
    do { if (ok) ok = cudaMemcpyAsync((dst), (src), (bytes), cudaMemcpyHostToDevice, stream) == cudaSuccess; } while (0)
    COPY_META(a, castA.data(), sizeof(void *) * groups);
    COPY_META(b, castB.data(), sizeof(void *) * groups);
    COPY_META(sa, castSA.data(), sizeof(void *) * groups);
    COPY_META(sb, castSB.data(), sizeof(void *) * groups);
    COPY_META(alpha, hostAlpha, sizeof(void *) * groups);
    COPY_META(d, castD.data(), sizeof(void *) * groups);
    COPY_META(rows, hostRows, sizeof(int) * groups);
#undef COPY_META
    if (ok) {
        SetupMetadataSm120<StrideA, StrideB, StrideD, LayoutSFA, LayoutSFB, ScaleConfig>
            <<<1, std::min(256, groups), 0, stream>>>(
                shapes, strideA, strideB, strideD, layoutA, layoutB,
                rows, groups, n, k);
        ok = cudaGetLastError() == cudaSuccess;
    }

    if (ok) {
        typename Kernel::MainloopArguments mainloop{
            a, strideA, b, strideB, sa, layoutA, sb, layoutB};
        typename Kernel::EpilogueArguments epilogue{{}, nullptr, strideD, d, strideD};
        epilogue.thread.alpha_ptr_array = const_cast<float **>(alpha);
        epilogue.thread.dAlpha = {_0{}, _0{}, 1};
        epilogue.thread.beta = 0.0f;
        cutlass::KernelHardwareInfo hw;
        int device = 0;
        cudaGetDevice(&device);
        hw.device_id = device;
        cudaDeviceGetAttribute(&hw.sm_count, cudaDevAttrMultiProcessorCount, device);
        typename Kernel::TileSchedulerArguments scheduler;
        using Raster = cutlass::gemm::kernel::detail::RasterOrderOptions;
        scheduler.raster_order = Raster::AlongM;
        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {groups, reinterpret_cast<ShapeType *>(shapes), nullptr},
            mainloop, epilogue, hw, scheduler};
        Gemm gemm;
        size_t workspaceBytes = Gemm::get_workspace_size(args);
        ok = EnsureBufferSm120(
                 scratch->workspace, scratch->workspaceCapacity, workspaceBytes) &&
             gemm.can_implement(args) == cutlass::Status::kSuccess &&
             gemm.initialize(args, scratch->workspace, stream) == cutlass::Status::kSuccess &&
             gemm.run(args, scratch->workspace, stream) == cutlass::Status::kSuccess &&
             cudaGetLastError() == cudaSuccess;
    }
    return ok;
}
} // namespace fastllm_nvfp4_moe_sm120
#endif

bool FastllmCudaNvfp4GroupedGemmSm120(
        const uint8_t *const *a, const uint8_t *const *b,
        const uint8_t *const *scaleA, const uint8_t *const *scaleB,
        const float *const *alpha, void *const *d, const int *rows,
        int groups, int n, int k, fastllm::DataType outputType,
        void *streamPtr) {
#if defined(FASTLLM_ENABLE_CUTLASS_NVFP4_SM120)
    if (!a || !b || !scaleA || !scaleB || !alpha || !d || !rows ||
        groups <= 0 || groups > 256 || n <= 0 || k <= 0 ||
        n % 32 != 0 || k % 32 != 0 ||
        outputType != fastllm::DataType::BFLOAT16) return false;
    cudaStream_t stream = streamPtr ? static_cast<cudaStream_t>(streamPtr) : 0;
    return fastllm_nvfp4_moe_sm120::RunGroupedSm120(
        a, b, scaleA, scaleB, alpha, d, rows, groups, n, k, stream);
#else
    (void)a; (void)b; (void)scaleA; (void)scaleB; (void)alpha; (void)d;
    (void)rows; (void)groups; (void)n; (void)k; (void)outputType; (void)streamPtr;
#endif
    return false;
}
