# NVFP4 验证

本目录只增加测试入口，不复制 `forward_check.py`、`prefill.py`、`decode.py`。
运行脚本会把命令、UTC 时间、工作目录和完整输出一并写入 `test/nvfp4/logs/`。

## 调用条件

| 路径 | 精确条件 |
|---|---|
| dense CUTLASS W4A4 | CUDA 12.9+；CUDA 12.x 编译目标 `100a/120a/121a`，CUDA 13.x 为 `100f/120f/121f`；当前运行时 SM100 或 SM120-SM129；权重 `NVFP4_BLOCK_16`、blockM=16；输入/输出同为 FP16 或 BF16；无 bias；K/N 为 32 的倍数；M 默认至少 32 |
| 动态激活 FP4 | SM100 或 SM120-SM129；FP16/BF16 输入；列数为 16 的倍数；dense 正式入口还受上述 K/N、M 条件约束 |
| SwiGLU+FP4 融合 | SM100 或 SM120-SM129；FP16/BF16 gate/up；hidden 为 16 的倍数 |
| grouped MoE W4A4 | CUDA 12.9+；当前仅 SM100；FP16/BF16；SwiGLU；仅 `NVFP4_BLOCK_16` experts；hidden/inter 为 32 的倍数；expert 数 1-256；batch 默认至少 16；无 shared expert/EP |
| dense Marlin W4A16 | SM75-SM99；FP16 输入/输出；权重 `NVFP4_BLOCK_16`、blockM=16；无 bias；M>=2；K/N 为 64 的倍数 |

环境变量：

- `FASTLLM_CUDA_NVFP4_W4A4=0`：关闭 dense W4A4。
- `FASTLLM_CUDA_NVFP4_W4A4_MIN_ROWS=N`：调整 dense W4A4 的 M 门槛，默认 32。
- `FASTLLM_CUDA_NVFP4_TRACE=1`：在 log 中打印命中或回退原因。
- `FASTLLM_CUDA_MOE_NVFP4_W4A4=0`：关闭 grouped MoE W4A4。
- `FASTLLM_CUDA_MOE_NVFP4_W4A4_MIN_BATCH=N`：grouped MoE 门槛，默认 16。

## 模型和显卡

- dense W4A4 首选：`nm-testing/TinyLlama-1.1B-Chat-v1.0-NVFP4`；B200/B300（SM100）或 RTX PRO 6000 Blackwell（SM120，96GB）。这是最小、最便于 forward_check 的 W4A4 模型。
- grouped MoE W4A4：`nvidia/Qwen3-30B-A3B-NVFP4`；当前路径用 B200/B300（SM100）。它不是“小总参数模型”，但 active 参数约 3B，是目前更现实的 MoE 检查点。
- Marlin W4A16：`nm-testing/fp4_nvfp4a16-e2e`；A100、H100、RTX 4090 等 SM75-SM99 GPU。

先由有对应 GPU 的机器完成构建。本次代码迁移按要求不在本机编译。

## 命令

Blackwell 算子功能与性能：

```bash
NVFP4_GPU_PROFILE=sm100 test/nvfp4/run_nvfp4_tests.sh ops
NVFP4_GPU_PROFILE=sm120 test/nvfp4/run_nvfp4_tests.sh ops
```

Marlin 算子功能与性能：

```bash
NVFP4_GPU_PROFILE=preblackwell test/nvfp4/run_nvfp4_tests.sh ops
```

forward_check：

```bash
NVFP4_MODEL=/models/TinyLlama-1.1B-Chat-v1.0-NVFP4 \
  test/nvfp4/run_nvfp4_tests.sh forward
```

整体模型 prefill/decode：

```bash
NVFP4_MODEL=/models/TinyLlama-1.1B-Chat-v1.0-NVFP4 \
  test/nvfp4/run_nvfp4_tests.sh model
```

一次执行全部：

```bash
NVFP4_MODEL=/models/TinyLlama-1.1B-Chat-v1.0-NVFP4 \
  test/nvfp4/run_nvfp4_tests.sh all
```

指定已有 `optest` 或日志目录：

```bash
NVFP4_OPTEST=/opt/fastllm/optest \
NVFP4_LOG_DIR=/logs/nvfp4-run-001 \
  test/nvfp4/run_nvfp4_tests.sh ops
```
