#!/usr/bin/env python3
"""把W8A8算子功能测试日志汇总为MD、CSV、JSON和XLSX。"""

import argparse
import csv
import json
from pathlib import Path
import re
import sys


REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "test" / "nvfp4"))
from xlsx_report import write_xlsx as write_workbook  # noqa: E402


HEADERS = [
    "case", "backend", "category", "result", "op", "m", "n", "k",
    "input_type", "weight_layout", "bias", "check", "max_abs_diff",
    "mean_abs_diff", "details", "log",
]

FUNCTIONAL_LOG_PATTERNS = (
    re.compile(r"^sm90_.*_function$"),
    re.compile(r"^sm90_branch_"),
    re.compile(r"^sm90_semantics_"),
    re.compile(r"^sm90_tensorwise_"),
    re.compile(r"^sm90_dense_lifecycle_"),
    re.compile(r"^sm90_vllm_"),
    re.compile(r"^sm120_branch_"),
    re.compile(r"^sm120_semantics_"),
    re.compile(r"^sm120_tensorwise_"),
    re.compile(r"^sm120_fixed_backend_lifecycle$"),
    re.compile(r"^sm120_first_gemm_failure_rejected$"),
    re.compile(r"^sm120_fixed_cutlass_failure_no_fallback$"),
    re.compile(r"^sm120_destructor_clears_backend_state$"),
    re.compile(r"^sm120_gpu_migration_rebuilds_cache$"),
    re.compile(r"^sm120_vllm_"),
    re.compile(r"^sm120_fp8_blockwise_"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="汇总W8A8算子功能测试日志")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--output-prefix", default="operator-functional-summary")
    return parser.parse_args()


def parameter(text, name):
    match = re.search(rf"--param\s+{re.escape(name)}=([^\s]+)", text)
    return match.group(1) if match else None


def integer_parameter(text, name):
    value = parameter(text, name)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def last_float(text, name):
    matches = re.findall(rf"\b{re.escape(name)}=([0-9.eE+-]+)", text)
    return float(matches[-1]) if matches else None


def result_from_log(text):
    if re.search(r"(?:Summary:\s*FAIL|optest error:|Traceback \(most recent call last\))", text):
        return "FAIL"
    if re.search(r"Summary:\s*PASS", text):
        return "PASS"
    return "UNKNOWN"


def category_from_name(name):
    if "lifecycle" in name or "failure" in name or "destructor" in name or "migration" in name:
        return "Backend lifecycle"
    if "tensorwise" in name or "semantics" in name:
        return "Scale/dtype/bias semantics"
    if "fallback_guard" in name:
        return "Fallback guard"
    if "vllm" in name:
        return "vLLM official coverage"
    if "branch" in name:
        return "M branch boundary"
    return "Functional"


def failure_details(text):
    patterns = (
        r"optest error:\s*(.+)",
        r"RuntimeError:\s*(.+)",
        r"AssertionError:\s*(.+)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    summary = re.findall(r"Summary:\s*[^\n]+", text)
    return summary[-1].strip() if summary else ""


def parse_fastllm_log(path, text):
    command = re.search(r"^COMMAND:(.*)$", text, re.MULTILINE)
    command_text = command.group(1) if command else text
    op = re.search(r"--op\s+([^\s]+)", command_text)
    result = result_from_log(text)
    weight_layout = parameter(command_text, "weight_layout")
    if weight_layout == "perchannel":
        weight_layout = "per-channel"
    return {
        "case": path.stem,
        "backend": "FastLLM",
        "category": category_from_name(path.stem),
        "result": result,
        "op": op.group(1) if op else None,
        "m": integer_parameter(command_text, "batch"),
        "n": integer_parameter(command_text, "out"),
        "k": integer_parameter(command_text, "in"),
        "input_type": parameter(command_text, "input_type"),
        "weight_layout": weight_layout,
        "bias": integer_parameter(command_text, "has_bias"),
        "check": integer_parameter(command_text, "check"),
        "max_abs_diff": last_float(text, "max_abs_diff"),
        "mean_abs_diff": last_float(text, "mean_abs_diff"),
        "details": failure_details(text) if result != "PASS" else "",
        "log": path.name,
    }


def parse_vllm_official(path, text):
    rows = []
    layout_match = re.search(r"--scale-layout\s+([^\s]+)", text)
    suite_layout = layout_match.group(1) if layout_match else "all"
    pattern = re.compile(
        r"^PASS (?P<padded>padded )?m=(?P<m>\d+) n=(?P<n>\d+) "
        r"k=(?P<k>\d+) per_token=(?P<per_token>True|False) "
        r"per_channel=(?P<per_channel>True|False) bias=(?P<bias>True|False)$",
        re.MULTILINE,
    )
    for index, match in enumerate(pattern.finditer(text), 1):
        values = match.groupdict()
        rows.append({
            "case": f"vllm_scaled_mm_{index:03d}",
            "backend": "vLLM",
            "category": "vLLM padded coverage" if values["padded"] else "vLLM aligned coverage",
            "result": "PASS",
            "op": "cutlass_scaled_mm",
            "m": int(values["m"]), "n": int(values["n"]), "k": int(values["k"]),
            "input_type": "bf16",
            "weight_layout": "per-channel" if values["per_channel"] == "True" else "tensorwise",
            "bias": 1 if values["bias"] == "True" else 0,
            "check": f"A={'per-token' if values['per_token'] == 'True' else 'tensorwise'}",
            "max_abs_diff": None, "mean_abs_diff": None,
            "details": "", "log": path.name,
        })
    for index, dtype in enumerate(
            re.findall(r"^PASS output_dtype=([^\s]+)$", text, re.MULTILINE), 1):
        rows.append({
            "case": f"vllm_output_dtype_{index}", "backend": "vLLM",
            "category": "vLLM output dtype", "result": "PASS",
            "op": "cutlass_scaled_mm", "m": 512, "n": 512, "k": 512,
            "input_type": "bf16", "weight_layout": "per-channel", "bias": 1,
            "check": dtype, "max_abs_diff": None, "mean_abs_diff": None,
            "details": "", "log": path.name,
        })
    block_pattern = re.compile(
        r"^PASS block128 m=(?P<m>\d+) n=(?P<n>\d+) k=(?P<k>\d+) "
        r"output_dtype=(?P<dtype>[^\s]+)$",
        re.MULTILINE,
    )
    for index, match in enumerate(block_pattern.finditer(text), 1):
        values = match.groupdict()
        rows.append({
            "case": f"vllm_block128_{index:03d}",
            "backend": "vLLM", "category": "vLLM Block128 coverage",
            "result": "PASS", "op": "cutlass_scaled_mm_block128",
            "m": int(values["m"]), "n": int(values["n"]),
            "k": int(values["k"]), "input_type": "bf16",
            "weight_layout": "block128", "bias": 0,
            "check": values["dtype"], "max_abs_diff": None,
            "mean_abs_diff": None, "details": "", "log": path.name,
        })
    if suite_layout == "dense":
        suite_op = "cutlass_scaled_mm"
        suite_weight_layout = "per-channel"
    elif suite_layout == "block128":
        suite_op = "cutlass_scaled_mm_block128"
        suite_weight_layout = "block128"
    else:
        suite_op = "cutlass_scaled_mm_suite"
        suite_weight_layout = None

    rows.append({
        "case": path.stem, "backend": "vLLM", "category": "vLLM suite",
        "result": result_from_log(text), "op": suite_op,
        "m": None, "n": None, "k": None, "input_type": None,
        "weight_layout": suite_weight_layout, "bias": None, "check": None,
        "max_abs_diff": None, "mean_abs_diff": None,
        "details": failure_details(text), "log": path.name,
    })
    return rows


def collect_rows(log_dir):
    rows = []
    for path in sorted(log_dir.glob("*.log")):
        if path.name.startswith("operator-functional-summary"):
            continue
        if not any(pattern.search(path.stem) for pattern in FUNCTIONAL_LOG_PATTERNS):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.match(r"^sm(?:90|120)_vllm_official_functional", path.stem):
            rows.extend(parse_vllm_official(path, text))
        elif "--op " in text and "--param " in text:
            rows.append(parse_fastllm_log(path, text))
    return rows


def display_value(value):
    """将功能报告展示层的浮点值统一格式化为小数点后3位。"""
    return f"{value:.3f}" if isinstance(value, float) else value


def report_group(row):
    """按实际scale语义把功能结果分配到独立工作表。"""
    if row.get("weight_layout") in ("per-channel", "tensorwise"):
        return "Dense W8A8功能"
    if row.get("weight_layout") in ("separate", "block128"):
        return "Block128 W8A8功能"
    if row.get("op") in ("linear_fp8_block128",
                          "cutlass_scaled_mm_block128"):
        return "Block128 W8A8功能"
    if row.get("op") in ("linear_fp8_w8a8", "cutlass_scaled_mm"):
        return "Dense W8A8功能"
    return "其他W8A8功能"


def write_xlsx(path, rows):
    """生成按Dense、Block128和其他W8A8分类的多工作表报告。"""
    sheets = []
    for name in ("Dense W8A8功能", "Block128 W8A8功能", "其他W8A8功能"):
        selected = [row for row in rows if report_group(row) == name]
        if selected:
            sheets.append((
                name,
                HEADERS,
                [[row.get(header) for header in HEADERS] for row in selected],
            ))
    write_workbook(path, sheets)


def write_reports(log_dir, prefix, rows):
    output = log_dir / prefix
    (output.with_suffix(".json")).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(
            {key: display_value(value) for key, value in row.items()}
            for row in rows)
    passed = sum(row["result"] == "PASS" for row in rows)
    failed = sum(row["result"] == "FAIL" for row in rows)
    unknown = len(rows) - passed - failed
    lines = [
        "# W8A8算子功能测试汇总", "",
        f"> 共{len(rows)}项：PASS={passed}，FAIL={failed}，UNKNOWN={unknown}。", "",
        "| Case | 后端 | 类别 | 结果 | M | N | K | 输入 | 权重scale | Bias | Check | 最大绝对误差 | 平均绝对误差 | 说明 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        show = lambda key: "n/a" if row.get(key) is None else str(
            display_value(row[key]))
        details = show("details").replace("|", "\\|")
        lines.append(
            f"| {show('case')} | {show('backend')} | {show('category')} | {show('result')} | "
            f"{show('m')} | {show('n')} | {show('k')} | {show('input_type')} | "
            f"{show('weight_layout')} | {show('bias')} | {show('check')} | "
            f"{show('max_abs_diff')} | {show('mean_abs_diff')} | {details} |")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_xlsx(output.with_suffix(".xlsx"), rows)
    print(f"W8A8 functional summary: PASS={passed} FAIL={failed} UNKNOWN={unknown}")
    for suffix in ("md", "csv", "json", "xlsx"):
        print(f"  {output.with_suffix('.' + suffix)}")


def main():
    args = parse_args()
    log_dir = Path(args.log_dir).resolve()
    rows = collect_rows(log_dir)
    if not rows:
        raise RuntimeError(f"未在{log_dir}找到W8A8功能测试日志")
    write_reports(log_dir, args.output_prefix, rows)


if __name__ == "__main__":
    main()
