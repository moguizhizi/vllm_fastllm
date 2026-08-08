# 项目技能

## Skills

### Available skills

- git-commit-zh-split: 当用户要求提交代码、整理提交、准备 commit、拆分 commit、push，或指定提交与推送规范时使用。默认使用中文提交信息，将差异较大的改动拆分为多个提交；推送前先执行 fetch、stash、rebase、stash pop，再 push。 (file: /home/tf/codexworkspace/fastllm3/fastllm/.codex/skills/git-commit-zh-split/SKILL.md)
- fastllm-triton-ops: 当需要在 FastLLM CUDA 后端添加、扩展、调试、验证或测速 Triton 生成的算子时使用，覆盖 Python 编译服务、cubin 缓存、C++ fallback 接入和验证流程。 (file: /root/github/newgit2/fastllm/.codex/skills/fastllm-triton-ops/SKILL.md)

### How to use skills

- Discovery: 上面列出的 skill 是当前仓库可用的项目级 skill。
- Trigger rules: 当用户直接提到 skill 名称，或请求明显匹配 skill 描述时，必须使用对应 skill。
- Missing/blocked: 如果 skill 文件缺失或无法读取，简短说明后继续采用最佳替代方案。
- Context hygiene: 先读 `SKILL.md`，只在需要时再加载更多上下文。

## 代码注释规范

### 函数级别注释

- 新增或补充函数级别注释时，统一使用中文 Doxygen `/** ... */` 格式。
- 首行用一句话说明函数职责；随后说明职责边界、主要执行流程及关键失败行为。
- 涉及矩阵、张量或量化布局时，明确写出数学语义、逻辑形状、数据类型和布局约束。
- 使用 `@param` 逐项说明参数的语义、形状、类型及特殊取值；参数说明较长时按同一缩进换行。
- 使用 `@return` 说明返回值含义；返回失败时，应说明如何取得具体失败原因。
- 不复述代码表面行为；注释重点解释约束、生命周期、后端选择、同步和回退等不直观逻辑。
- 简单函数可以适当精简，但整体格式保持一致。推荐模板如下：

```cpp
/**
 * 执行一次NVFP4 W4A4 CUTLASS线性计算。
 *
 * 本函数只负责单次计算，不决定后端的最终生命周期。执行流程为：检查
 * GPU与张量语义、取得CUTLASS重排权重、准备临时缓冲区、将FP16/BF16
 * 激活动态量化为NVFP4、调用对应架构的CUTLASS GEMM，最后处理padding
 * 和bias。首次后端选择时可强制同步，以便在释放原始权重前发现异步错误。
 *
 * 参数采用标准GEMM语义：output[M,N] = input[M,K] * weight[N,K]^T。
 *
 * @param input                FP16或BF16输入；普通路径形状为[m, k]，融合
 *                             路径形状为[m, 2 * k]。
 * @param weight               NVFP4_BLOCK_16权重，逻辑形状为[n, k]。
 * @param bias                 可选FP32偏置，长度为n。
 * @param output               输出张量，逻辑形状为[m, n]。
 * @param m                    GEMM的M维，激活行数，通常为token数。
 * @param n                    GEMM的N维，输出特征数。
 * @param k                    GEMM的K维，输入特征数。
 * @param siluMulInput         true表示把SwiGLU计算与激活量化融合。
 * @param checkBackendPolicy   true表示首次选择时检查CUTLASS后端开关；后端
 *                             选择不受M影响，固定后的正式计算传false。
 * @param validateWarmup       true表示末尾强制同步，验证首次warmup的异步错误。
 * @return true表示本次CUTLASS计算及必要的后处理成功；false表示某个阶段
 *         不满足条件或执行失败，具体阶段通过Trace记录。
 */
```
