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
| dense Marlin W4A16 | SM75-SM99；FP16 输入/输出；权重 `NVFP4_BLOCK_16`、blockM=16；无 bias；M>=2；K/N 为 64 的倍数 |

环境变量：

动态激活量化与 vLLM 保持一致：CUDA 12.8 使用 PACK8/128-bit load，CUDA
12.9+ 使用 PACK16/256-bit load；两条路径均由编译器版本自动选择，无需手动宏。

- `FASTLLM_CUDA_NVFP4_W4A4=0`：关闭 dense W4A4。
- `FASTLLM_CUDA_NVFP4_W4A4_MIN_ROWS=N`：调试时覆盖 dense W4A4 的 M 门槛，默认 1。
- `FASTLLM_CUDA_NVFP4_SWIGLU_QUANT=0`：关闭 Dense SwiGLU+FP4 量化融合。
- `FASTLLM_CUDA_NVFP4_TRACE=1`：在 log 中打印命中或回退原因。
- `FASTLLM_CUDA_MOE_NVFP4_W4A4=0`：关闭 grouped MoE W4A4。
- `FASTLLM_CUDA_MOE_NVFP4_W4A4_MIN_BATCH=N`：grouped MoE 门槛，默认 16。

## 权重显存生命周期

首次命中 W4A4 时，FastLLM 将 `NVFP4_BLOCK_16` 重排为 CUTLASS FP4 权重和
E4M3 scale。重排流同步完成后，原始 `cudaData` 会从 CUDA 内存池移除并真正释放，后续按
`Data + device + shape` 命中重排缓存，不再保留两份 GPU 权重。CPU/mmap 原始数据
继续保留；W4A4 失败需要 legacy fallback 时，会按需恢复原始 CUDA 权重。

`ops-functional` 的 dense 用例会同时检查：原始 CUDA 权重已释放，以及关闭 W4A4
后 fallback 能恢复原始权重。

## 模型和显卡

- dense W4A4 首选：`nm-testing/TinyLlama-1.1B-Chat-v1.0-NVFP4`；B200/B300（SM100）或 RTX PRO 6000 Blackwell（SM120，96GB）。
- grouped MoE W4A4：`nvidia/Qwen3-30B-A3B-NVFP4`；当前路径用 B200/B300（SM100）。它不是“小总参数模型”，但 active 参数约 3B，是目前更现实的 MoE 检查点。
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
