"""解析并校验FastLLM Attention Backend正式路径记录。"""


TRACE_PREFIX = "[fastllm][attention] "


def parse_attention_backend_traces(output):
    """把每个静态调用签名的首次Backend选择记录解析为字段字典。"""
    traces = []
    for line in output.splitlines():
        if not line.startswith(TRACE_PREFIX):
            continue
        fields = {}
        for item in line[len(TRACE_PREFIX):].split():
            key, separator, value = item.partition("=")
            if separator:
                fields[key] = value
        if fields.get("actual"):
            traces.append(fields)
    return traces


def actual_attention_backends(output):
    """返回日志中确认执行且非none的Backend名称有序集合。"""
    return sorted({
        trace["actual"] for trace in parse_attention_backend_traces(output)
        if trace["actual"] != "none"
    })


def attention_backend_confirmed(requested, actual_backends):
    """判断auto是否产生实际路径，或显式Backend是否确实执行。"""
    return (bool(actual_backends) if requested == "auto" else
            requested in actual_backends)
