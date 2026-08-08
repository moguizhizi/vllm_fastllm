# 通用算子性能对比框架

`operator_benchmark.py`不绑定FastLLM、vLLM、CUDA或某一种算子。它按JSON配置
执行相同case的多个实现，提取性能指标，并输出逐维度对比结果。

适用场景：

- 对比同一算子修改前后的性能，例如`before`与`after`两个二进制。
- 对比相同参数下FastLLM与vLLM的算子性能。
- 同时比较多个候选实现、多个shape、dtype、batch、topk或expert数量。

## 快速运行

仓库提供了一个不需要GPU的最小示例：

```bash
cd /path/to/vllm_fastllm

python3 test/benchmark/operator_compare/operator_benchmark.py \
  --config test/benchmark/operator_compare/example.json \
  --output-prefix /tmp/operator-report
```

输出：

```text
/tmp/operator-report.md
/tmp/operator-report.json
/tmp/operator-report-results.csv
/tmp/operator-report-comparison.csv
/tmp/operator-report-logs/*.log
```

每份原始日志都包含UTC时间、工作目录、完整命令、配置中的环境变量、退出码和
命令输出。

## 配置结构

最小配置如下：

```json
{
  "name": "Linear优化前后对比",
  "baseline": "before",
  "repeat": 5,
  "aggregate": "median",
  "metrics": [
    {
      "name": "latency_ms",
      "unit": "ms",
      "goal": "min",
      "pattern": "latency: avg_ms=(?P<value>[0-9.eE+-]+)"
    }
  ],
  "implementations": [
    {
      "name": "before",
      "command": ["/opt/fastllm-before/optest", "--op", "{op}"]
    },
    {
      "name": "after",
      "command": ["{repo}/optest", "--op", "{op}"]
    }
  ],
  "cases": [
    {
      "name": "linear_m1",
      "dimensions": {"op": "linear_nvfp4", "M": 1, "N": 4096, "K": 4096}
    }
  ]
}
```

主要字段：

| 字段 | 含义 |
| --- | --- |
| `baseline` | 基准实现；其他实现都与它比较 |
| `repeat` | 框架层重复执行整条命令的次数 |
| `aggregate` | 多次结果的聚合方式：`median`、`mean`或`min` |
| `metrics` | 从输出提取的指标，可配置单位、优化方向和正则 |
| `implementations` | 被测实现，可指向不同程序、Python环境或Git版本产物 |
| `cases` | 测试矩阵；`dimensions`会原样进入CSV和Markdown |

### 模板变量

命令、工作目录和环境变量支持`{name}`模板。可用变量来自：

1. 顶层`variables`。
2. 当前case的`variables`。
3. 当前case的`dimensions`。
4. 内置的`{case}`、`{config_dir}`、`{repo}`和`{output_prefix}`。

后面的值覆盖前面的同名值。`command`必须写成JSON数组，每一项对应一个命令行
参数；框架默认不经过shell，避免空格和引号产生歧义。确实需要管道时显式使用：

```json
"command": ["bash", "-lc", "command_a | command_b"]
```

### 指标提取

正则可以使用第一个捕获组：

```json
"pattern": "avg_ms=([0-9.eE+-]+)"
```

也可以使用Python命名捕获组`value`：

```json
"pattern": "avg_ms=(?P<value>[0-9.eE+-]+)"
```

`goal=min`表示越小越好，例如延迟；`goal=max`表示越大越好，例如吞吐。
统一定义：

```text
goal=min：speedup = baseline / candidate
          improvement_pct = (baseline - candidate) / baseline * 100%
goal=max：speedup = candidate / baseline
          improvement_pct = (candidate - baseline) / baseline * 100%
```

因此无论指标方向如何，`speedup > 1`和正的`improvement_pct`都表示候选实现更好。
默认取输出中最后一次匹配；设置`"match": "first"`可取第一次。设置
`"required": false`可允许某个指标缺失。

如果两个框架输出格式不同，可以按实现覆盖正则：

```json
{
  "name": "latency_ms",
  "unit": "ms",
  "goal": "min",
  "patterns": {
    "FastLLM": "latency: avg_ms=(?P<value>[0-9.eE+-]+)",
    "vLLM": "vllm_latency_ms=(?P<value>[0-9.eE+-]+)"
  }
}
```

也可以在某个implementation内用`metric_patterns`覆盖：

```json
"metric_patterns": {
  "latency_ms": "latency_us=(?P<value>[0-9.eE+-]+)"
}
```

注意：框架不会自动换算单位。不同实现必须输出相同单位，或由被测命令先完成换算。

## 算子修改前后对比

先分别保存两版构建产物，然后在同一配置中定义两个实现：

```json
"implementations": [
  {
    "name": "before",
    "command": [
      "/opt/fastllm-before/optest", "--op", "linear_nvfp4",
      "--device", "cuda:0",
      "--param", "batch={M}", "--param", "in={K}", "--param", "out={N}",
      "--warmup", "20", "--iters", "200"
    ]
  },
  {
    "name": "after",
    "command": [
      "{repo}/optest", "--op", "linear_nvfp4",
      "--device", "cuda:0",
      "--param", "batch={M}", "--param", "in={K}", "--param", "out={N}",
      "--warmup", "20", "--iters", "200"
    ]
  }
]
```

case只保存变化的维度：

```json
"cases": [
  {"name": "m1", "dimensions": {"M": 1, "N": 4096, "K": 4096, "dtype": "bf16"}},
  {"name": "m128", "dimensions": {"M": 128, "N": 4096, "K": 4096, "dtype": "bf16"}},
  {"name": "m256", "dimensions": {"M": 256, "N": 4096, "K": 4096, "dtype": "bf16"}}
]
```

不要在同一目录反复覆盖`optest`。给不同版本保留不同的绝对路径，才能确保比较的
确实是两份代码。

## FastLLM与vLLM对比

同一套框架只需把实现名称和命令换成FastLLM、vLLM：

```json
{
  "baseline": "vLLM",
  "implementations": [
    {
      "name": "vLLM",
      "command": [
        "/opt/vllm/bin/python", "/opt/bench/vllm_linear.py",
        "--m", "{M}", "--n", "{N}", "--k", "{K}", "--dtype", "{dtype}"
      ],
      "metric_patterns": {
        "latency_ms": "vllm_latency_ms=(?P<value>[0-9.eE+-]+)"
      }
    },
    {
      "name": "FastLLM",
      "command": [
        "{repo}/optest", "--op", "linear_nvfp4", "--device", "cuda:0",
        "--param", "batch={M}", "--param", "in={K}", "--param", "out={N}",
        "--param", "input_type={dtype}", "--warmup", "20", "--iters", "200"
      ],
      "metric_patterns": {
        "latency_ms": "latency: avg_ms=(?P<value>[0-9.eE+-]+)"
      }
    }
  ]
}
```

这里`baseline=vLLM`，所以报告中的candidate为FastLLM，speedup含义是FastLLM相对
vLLM的加速比。框架不要求两个实现在同一个Python环境中。

## case级覆盖

不同算子可以共用一个配置。case可单独覆盖命令、环境变量、工作目录和超时：

```json
{
  "name": "moe_m64",
  "dimensions": {"op": "moe", "M": 64, "topk": 8, "experts": 64},
  "timeout": 300,
  "env": {"CUDA_VISIBLE_DEVICES": "0"},
  "commands": {
    "FastLLM": ["{repo}/optest", "--op", "mergemoe_fp8"],
    "vLLM": ["/opt/vllm/bin/python", "/opt/bench/vllm_moe.py"]
  },
  "implementation_env": {
    "FastLLM": {"FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT": "1"}
  }
}
```

## 常用命令

检查模板展开，不运行算子：

```bash
python3 test/benchmark/operator_compare/operator_benchmark.py \
  --config bench.json --output-prefix logs/report --dry-run
```

只跑部分case：

```bash
python3 test/benchmark/operator_compare/operator_benchmark.py \
  --config bench.json --output-prefix logs/report \
  --case 'dense_*' --case 'moe_m64*'
```

覆盖重复次数和聚合方法：

```bash
python3 test/benchmark/operator_compare/operator_benchmark.py \
  --config bench.json --output-prefix logs/report \
  --repeat 7 --aggregate median
```

单项失败后继续生成其余结果：

```bash
python3 test/benchmark/operator_compare/operator_benchmark.py \
  --config bench.json --output-prefix logs/report --keep-going
```

运行框架自测：

```bash
python3 -m unittest discover \
  -s test/benchmark/operator_compare -p 'test_*.py'
```

## 测试原则

- 两个实现必须使用相同输入shape、dtype、warmup、iters和同步边界。
- CUDA算子计时前后都应同步GPU；不要把异步kernel启动时间当成执行时间。
- 修改前后对比必须保留两份独立构建产物。
- 框架层`repeat`用于观察进程级波动；算子内部仍应执行足够多的warmup和iters。
- 性能测试不要开启会插入同步或打印大量日志的trace开关。
- 对比GPU算子时固定GPU、功耗模式和系统负载，并记录驱动、CUDA和代码commit。
