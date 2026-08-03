# CUDA W8A8 验证

结论：测试只在对应 GPU 机器运行；脚本不负责编译。每条命令、UTC 时间和完整输出都写入 `test/w8a8/logs/<timestamp>/`。

## 严格调用条件

| 路径 | 条件 |
|---|---|
| SM90 INT8 dense | 编译含 `90a`，CUDA 12.3+；运行时精确 SM90；FP16/BF16 输入输出同 dtype；2D signed-I8 权重；权重 symmetric、无 zero/min；每输出通道一个 FP32 scale；激活动态 per-token symmetric INT8；K%16=0、N%8=0；bias 为空或 FP32[N] |
| SM90 FP8 blockwise dense | 编译含 `90a`，CUDA 12.3+；运行时精确 SM90；FP16/BF16；FP8 E4M3 权重；activation/weight block 都是 128x128；M/N/K 正数，K/N 为 128 倍数；bias 为空或 FP32[N] |
| SM90 FP8 grouped MoE | 运行时精确 SM90；当前仅 BF16；SwiGLU；无 shared expert；expert 为 standard per-channel FP8 E4M3（不是 blockwise）；合法 route index；默认 batch>=16；hidden/inter 为 16 倍数；不满足直接回现有 MoE 路径 |
| SM120 standard FP8 dense | 编译含 `120a`/`120f`，CUDA 12.9+；运行时精确 SM120，不含 SM121；FP16/BF16；FP8 E4M3 权重；weight `blockK=1, blockM=K` 且 scale 数=N；激活动态 per-token；K%16=0、N%8=0；bias 为空或 FP32[N] |

## 命令

算子功能：

```bash
W8A8_LOG_DIR="$PWD/test/w8a8/logs/sm90-function" \
  test/w8a8/run_w8a8_tests.sh ops-functional
```

算子性能：

```bash
W8A8_LOG_DIR="$PWD/test/w8a8/logs/sm90-perf" \
  test/w8a8/run_w8a8_tests.sh ops-performance
```

forward_check：

```bash
W8A8_MODEL=/models/Qwen2.5-0.5B-Instruct-quantized.w8a8 \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/int8-forward" \
  test/w8a8/run_w8a8_tests.sh forward
```

整体 prefill/decode：

```bash
W8A8_MODEL=/models/Qwen3-30B-A3B-FP8 \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/fp8-e2e" \
  test/w8a8/run_w8a8_tests.sh model-performance
```

## 模型（魔塔优先）

| 用途 | ModelScope 模型 |
|---|---|
| SM90 INT8 dense，最小 forward_check | `neuralmagic/Qwen2.5-0.5B-Instruct-quantized.w8a8` |
| SM90/SM120 FP8 blockwise dense | `Qwen/Qwen3-30B-A3B-FP8`（block=128；同时可测 MoE 现有 blockwise 路径） |
| SM90 standard FP8 grouped MoE | ModelScope 未找到可直接匹配 FastLLM 独立 expert 布局的模型；HF fallback：`nm-testing/Mixtral-8x7B-Instruct-v0.1-FP8-Dynamic` |
| SM120 standard FP8 dense，较小 | `neuralmagic/Qwen2.5-7B-FP8-dynamic` |

注意：SM90 grouped CUTLASS 的目标格式是 standard per-token/per-channel FP8；Qwen3 官方 FP8 是 128x128 blockwise，不能拿它证明 standard grouped kernel 命中。
HF Mixtral fallback 的 `config.json` 是 compressed-tensors `float-quantized`：FP8 weight 为 static symmetric channel，activation 为 dynamic symmetric token。

开关：`FASTLLM_CUDA_MOE_CUTLASS_FP8_SM90=0` 可单独关闭 SM90 grouped CUTLASS，便于和原 grouped CUDA fallback 做 A/B 性能对比。
