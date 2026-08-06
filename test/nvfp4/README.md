# NVFP4 验证

本目录增加 NVFP4 专用的 vLLM 对照测试入口，不修改通用 `forward_check.py`、
`prefill.py`、`decode.py`。
运行脚本会把命令、UTC 时间、工作目录和完整输出一并写入 `test/nvfp4/logs/`。

## 调用条件

| 路径 | 精确条件 |
|---|---|
| dense CUTLASS W4A4 | CUDA 12.8+；CUDA 12.x 编译目标 `100a/120a/121a`，CUDA 13.x 为 `100f/120f/121f`；当前运行时 SM100-SM119 或 SM120-SM129；权重 `NVFP4_BLOCK_16`、blockM=16；输入/输出同为 FP16 或 BF16；支持空 bias 或 FP32 bias；原始 K 为 16 的倍数；K/N 不足 32 时内部补零并裁剪；M>=1 |
| 动态激活 FP4 | SM100 或 SM120-SM129；FP16/BF16 输入；列数为 16 的倍数；dense 正式入口还受上述 K/N、M 条件约束 |
| SwiGLU+FP4 融合 | SM100 或 SM120-SM129；FP16/BF16 gate/up；hidden 为 16 的倍数；已接入 Dense `SwigluLinearAdd` 和 grouped MoE |
| grouped MoE W4A4 | CUDA 12.8+；当前仅 SM100；FP16/BF16；SwiGLU；仅 `NVFP4_BLOCK_16` experts；hidden/inter 为 32 的倍数；expert 数 1-256；batch 默认至少 16；无 shared expert/EP |
| dense Marlin W4A16 | SM75-SM99；FP16 输入/输出；权重 `NVFP4_BLOCK_16`、blockM=16；无 bias；M>=1；K/N 为 64 的倍数 |

环境变量：

动态激活量化与 vLLM 保持一致：CUDA 12.8 使用 PACK8/128-bit load，CUDA
12.9+ 使用 PACK16/256-bit load；两条路径均由编译器版本自动选择，无需手动宏。

- `FASTLLM_CUDA_NVFP4_W4A4=0`：关闭 dense W4A4。
- `FASTLLM_CUDA_NVFP4_W4A4_STRICT=1`：dense W4A4必须命中CUTLASS；预重排或
  首次GEMM验证失败时直接报错，禁止Legacy/Marlin fallback。
- `FASTLLM_CUDA_NVFP4_SWIGLU_QUANT=0`：关闭 Dense SwiGLU+FP4 量化融合。
- `FASTLLM_CUDA_NVFP4_TRACE=1`：在 log 中打印命中或回退原因；该模式会
  插入CUDA流同步，只用于功能诊断，不用于正式性能测试。
- `FASTLLM_CUDA_MOE_NVFP4_W4A4=0`：关闭 grouped MoE W4A4。
- `FASTLLM_CUDA_MOE_NVFP4_W4A4_MIN_BATCH=N`：grouped MoE 门槛，默认 16。

## 权重显存生命周期

每个NVFP4 `Data + device` 在权重进入GPU时选择候选后端。dense CUTLASS候选会立即
完成重排并释放原始 `cudaData`，状态记为Prepared；首个真实GEMM同步验证成功后固定
为Cutlass。验证失败则销毁候选cache、从CPU/mmap恢复原始权重，并固定使用legacy。
正式推理开始后不再通过修改环境变量切换已固定后端，运行阶段kernel失败直接报错。

dense W4A4 的首次后端选择不设置 M 门槛。M 只用于选择 CUTLASS 内部 tile/config，
不会导致CUTLASS与legacy在不同batch之间切换。上传阶段负责重排并释放原始CUDA
权重；首个真实GEMM仍负责验证激活量化、GEMM、后处理和异步CUDA状态。

SwiGLU 融合能力单独记录：融合 warmup 失败只关闭融合，随后执行普通 SwiGLU，再由
Linear 独立选择 CUTLASS/Marlin/legacy。它不会把 Linear 一起降级。

CPU/mmap 原始数据继续保留，用于目标GPU重建。NVFP4权重进入GPU时逐个完成上传、
预重排和原始CUDA表示释放，首个真实Linear只负责同步验证GEMM并固定后端。权重从
GPU0移动到GPU1时先销毁GPU0 cache，再从host源直接上传到GPU1并立即预重排，
不再执行“host→GPU0原始权重→GPU1”的中转。

`ops-functional` 会检查：上传预重排后立即释放原始CUDA权重、正式推理不切换后端、
首次GEMM验证失败后恢复并固定legacy、融合拒绝不影响Linear；存在两张GPU时还检查
跨GPU直接重建及GPU0/GPU1输出数值。SwiGLU用“普通SwiGLU+独立NVFP4量化”
作为融合量化的对照；激活量化覆盖真实K=4096/7168/14336。SM100和SM120的
grouped MoE覆盖M=2/64/224、topk=1/8、experts=40/64。

`ops-performance` 的dense W4A4覆盖M=1/16/64/128/256/512/1024；grouped MoE
使用同一组M/topk/experts矩阵。测试开启strict检查：CUTLASS失败时直接报错，
不允许legacy fallback伪装成CUTLASS成绩。

## 模型和显卡

- dense W4A4 首选：`nm-testing/TinyLlama-1.1B-Chat-v1.0-NVFP4`；B200/B300（SM100）或 RTX PRO 6000 Blackwell（SM120，96GB）。
- grouped MoE W4A4：`RedHatAI/Qwen3-30B-A3B-NVFP4`；模型卡明确给出vLLM
  离线用法，Qwen3-MoE也由FastLLM支持。单卡约18GB权重，优先在RTX 5090
  （SM120）验证，再在B200/B300（SM100）做架构交叉验证。
- Marlin W4A16：`nm-testing/fp4_nvfp4a16-e2e`；A100、H100、RTX 4090 等 SM75-SM99 GPU。

先由有对应 GPU 的机器完成构建。本次代码迁移按要求不在本机编译。

## 命令

以下命令都把原始命令和完整输出保存到 `NVFP4_LOG_DIR`。建议同一次验证固定同一个目录。

远端编译。脚本通过 `nvidia-smi` 自动识别 B200/B300、RTX 5090/RTX PRO 6000
等 GPU，把实际算力显式传给 `CUDA_ARCH`，并通过仓库现有的 `install.sh`
编译 FastLLM 和 `optest`，避免旧版 CMake 产生无效的 `compute_native`：

```bash
export NVFP4_LOG_DIR="$PWD/test/nvfp4/logs/remote-sm120-$(date -u +%Y%m%dT%H%M%SZ)"
test/nvfp4/run_nvfp4_tests.sh build
```

算子功能测试：

```bash
test/nvfp4/run_nvfp4_tests.sh ops-functional
```

算子性能测试：

```bash
test/nvfp4/run_nvfp4_tests.sh ops-performance
```

Blackwell 算子功能与性能：

```bash
NVFP4_GPU_PROFILE=sm100 test/nvfp4/run_nvfp4_tests.sh ops
NVFP4_GPU_PROFILE=sm120 test/nvfp4/run_nvfp4_tests.sh ops
```

Marlin 算子功能与性能：

```bash
NVFP4_GPU_PROFILE=preblackwell test/nvfp4/run_nvfp4_tests.sh ops
```

forward_check 只使用 vLLM 作为参考实现。vLLM 和 FastLLM 在两个独立进程中依次
加载同一个 NVFP4 模型，避免两份模型同时占用显存；检查输入 token、贪心生成 token、
首 token 以及 top-k logprob。运行环境必须能 `import vllm`。若 vLLM 安装在另一个
Python 环境，用 `NVFP4_VLLM_PYTHON` 指定：

```bash
NVFP4_MODEL=/models/TinyLlama-1.1B-Chat-v1.0-NVFP4 \
NVFP4_VLLM_PYTHON=/path/to/vllm/python \
  test/nvfp4/run_nvfp4_tests.sh forward
```

整体模型 prefill/decode：

```bash
NVFP4_MODEL=/models/TinyLlama-1.1B-Chat-v1.0-NVFP4 \
  test/nvfp4/run_nvfp4_tests.sh model-performance
```

一次执行全部：

```bash
NVFP4_MODEL=/models/TinyLlama-1.1B-Chat-v1.0-NVFP4 \
  test/nvfp4/run_nvfp4_tests.sh all
```

主要日志文件：

```text
environment.log
nvcc_version.log
install.log
op_dense_w4a4_decode_padding_bias.log
op_dense_w4a4_decode_perf.log
op_dense_w4a4_prefill_perf.log
op_swiglu_fp4_quant.log
op_swiglu_fp4_quant_fp16.log
op_swiglu_fp4_quant_perf.log
forward_check_vllm.log
forward-vllm-results/vllm.json
forward-vllm-results/fastllm.json
model_prefill.log
model_decode.log
```

指定已有 `optest` 或日志目录：

```bash
NVFP4_OPTEST=/opt/fastllm/optest \
NVFP4_LOG_DIR=/logs/nvfp4-run-001 \
  test/nvfp4/run_nvfp4_tests.sh ops
```

一般不需要设置 `NVFP4_GPU_PROFILE` 或 `NVFP4_CUDA_ARCH`。前者只在自动识别错误或
需要强制测试某条路径时覆盖；后者只在需要交叉编译或指定多个架构时传给 `install.sh`。
