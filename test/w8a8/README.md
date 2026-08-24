# CUDA W8A8 验证

结论：测试只在对应GPU机器运行。`build`负责编译安装；其他suite把每条
命令、UTC时间和完整输出写入`test/w8a8/logs/<timestamp>/`。

SM120标准FP8 W8A8采用固定后端生命周期：每份权重在每张GPU上只进行
一次真实CUTLASS GEMM选择。首次同步验证成功后固定为CUTLASS；失败则固定
交给Legacy，正式推理期间不再混切。FP8权重本身已经是CUTLASS可消费格式，
不会复制第二份权重；设备scale、类型化bias、量化激活和workspace临时区会
缓存复用。scale、bias与输出类型转换已融合进CUTLASS epilogue，不再分配
完整的FP32中间矩阵或额外启动缩放kernel。

## 严格调用条件

| 路径 | 条件 |
|---|---|
| SM90 INT8 dense | 编译含 `90a`，CUDA 12.3+；运行时精确 SM90；FP16/BF16 输入输出同 dtype；2D signed-I8 权重；权重 symmetric、无 zero/min；每输出通道一个 FP32 scale；激活动态 per-token symmetric INT8；K%16=0、N%8=0；bias 为空或 FP32[N] |
| SM90 FP8 blockwise dense | 编译含 `90a`，CUDA 12.3+；运行时精确 SM90；FP16/BF16；FP8 E4M3 权重；activation/weight block 都是 128x128；M/N/K 正数，K/N 为 128 倍数；bias 为空或 FP32[N] |
| SM90 FP8 grouped MoE | 运行时精确 SM90；当前仅 BF16；SwiGLU；无 shared expert；expert 为 standard per-channel FP8 E4M3（不是 blockwise）；合法 route index；默认 batch>=16；hidden/inter 为 16 倍数；不满足直接回现有 MoE 路径 |
| SM120 standard FP8 dense | 编译含 `120a`/`120f`，CUDA 12.9+；运行时精确 SM120，不含 SM121；FP16/BF16；FP8 E4M3 权重；weight `blockK=1, blockM=K` 且 scale 数为N（per-channel）或1（tensorwise）；激活动态 per-token；K%16=0、N%8=0；bias 为空或 FP32[N] |

## 命令

编译安装：

```bash
W8A8_NVCC=/usr/local/cuda-13.0/bin/nvcc \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/sm120-build" \
  test/w8a8/run_w8a8_tests.sh build
```

算子功能（FastLLM分支、固定后端生命周期及vLLM官方形状/scale组合）：

```bash
W8A8_VLLM_PYTHON=/root/miniconda3/bin/python \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/sm120-function" \
  test/w8a8/run_w8a8_tests.sh ops-functional
```

其中SM120生命周期包含四类用例：首次GEMM故障后固定为`Rejected`且
不再重试；已固定为`Cutlass`后的运行故障必须抛错而不能fallback；权重
`Data`析构后必须删除以其地址为键的后端状态，防止新对象继承陈旧记录；
权重从GPU0迁移到GPU1时必须清理旧卡状态，并从主机scale源在新卡重建缓存。
故障通过测试专用环境变量注入，但调用的仍是正式W8A8 CUTLASS入口和状态机。

算子性能（直接生成FastLLM/vLLM Markdown、CSV和JSON对比）：

```bash
W8A8_VLLM_PYTHON=/root/miniconda3/bin/python \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/sm120-perf" \
  test/w8a8/run_w8a8_tests.sh ops-performance
```

forward_check：

```bash
W8A8_MODEL=/root/autodl-tmp/neuralmagic/Qwen2___5-7B-FP8-dynamic \
W8A8_VLLM_PYTHON=/root/miniconda3/bin/python \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/sm120-forward" \
  test/w8a8/run_w8a8_tests.sh forward
```

整体模型性能（相同HTTP接口、相同Token ID，生成FastLLM/vLLM对比表）：

```bash
W8A8_MODEL=/root/autodl-tmp/neuralmagic/Qwen2___5-7B-FP8-dynamic \
W8A8_VLLM_PYTHON=/root/miniconda3/bin/python \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/sm120-model" \
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
