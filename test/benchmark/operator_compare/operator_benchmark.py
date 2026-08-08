#!/usr/bin/env python3
"""通用算子性能对比：执行多实现、多维度case并生成JSON/CSV/Markdown报告。"""

import argparse
import csv
import fnmatch
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone


def parse_args():
    parser = argparse.ArgumentParser(description="通用算子性能对比框架")
    parser.add_argument("--config", required=True, help="JSON配置文件")
    parser.add_argument("--output-prefix", required=True, help="输出文件前缀")
    parser.add_argument("--case", action="append", default=[],
                        help="只运行匹配glob的case，可重复指定")
    parser.add_argument("--implementation", action="append", default=[],
                        help="只运行指定实现；必须包含baseline")
    parser.add_argument("--repeat", type=int, default=0,
                        help="覆盖配置中的外层重复次数")
    parser.add_argument("--aggregate", choices=("median", "mean", "min"),
                        help="覆盖配置中的聚合方式")
    parser.add_argument("--dry-run", action="store_true", help="只打印展开后的命令")
    parser.add_argument("--keep-going", action="store_true", help="单项失败后继续")
    return parser.parse_args()


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = ("name", "baseline", "metrics", "implementations", "cases")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"配置缺少字段: {', '.join(missing)}")
    names = [item["name"] for item in config["implementations"]]
    if len(names) != len(set(names)):
        raise ValueError("implementation名称不能重复")
    if config["baseline"] not in names:
        raise ValueError(f"baseline不存在: {config['baseline']}")
    metric_names = [item["name"] for item in config["metrics"]]
    if len(metric_names) != len(set(metric_names)):
        raise ValueError("metric名称不能重复")
    for metric in config["metrics"]:
        if metric.get("goal", "min") not in ("min", "max"):
            raise ValueError(f"metric goal只能是min或max: {metric['name']}")
    return config


def format_value(value, context):
    if not isinstance(value, str):
        return str(value)
    try:
        return value.format_map(context)
    except KeyError as error:
        raise ValueError(f"模板变量不存在: {error.args[0]}；模板={value}") from error


def merge_context(config, case, config_path, output_prefix):
    context = {}
    context.update(config.get("variables", {}))
    context.update(case.get("variables", {}))
    context.update(case.get("dimensions", {}))
    context.update({
        "case": case["name"],
        "config_dir": str(config_path.parent.resolve()),
        "repo": str(Path(__file__).resolve().parents[3]),
        "output_prefix": str(output_prefix.resolve()),
    })
    return context


def resolve_command(config, case, implementation, context):
    override = case.get("commands", {}).get(implementation["name"])
    command = override if override is not None else implementation.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError(
            f"{case['name']}/{implementation['name']}的command必须是非空JSON数组")
    return [format_value(value, context) for value in command]


def resolve_cwd(config, case, implementation, context):
    value = case.get("implementation_cwd", {}).get(implementation["name"])
    if value is None:
        value = case.get("cwd", implementation.get("cwd", config.get("cwd", "{repo}")))
    return Path(format_value(value, context)).resolve()


def resolve_env(config, case, implementation, context):
    values = {}
    values.update(config.get("env", {}))
    values.update(implementation.get("env", {}))
    values.update(case.get("env", {}))
    values.update(case.get("implementation_env", {}).get(implementation["name"], {}))
    return {key: format_value(value, context) for key, value in values.items()}


def resolve_metric_pattern(metric, implementation):
    name = metric["name"]
    if name in implementation.get("metric_patterns", {}):
        return implementation["metric_patterns"][name]
    if implementation["name"] in metric.get("patterns", {}):
        return metric["patterns"][implementation["name"]]
    if "pattern" in metric:
        return metric["pattern"]
    raise ValueError(f"指标{name}没有适用于{implementation['name']}的正则表达式")


def parse_metric(text, metric, implementation):
    pattern = resolve_metric_pattern(metric, implementation)
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        if metric.get("required", True):
            raise ValueError(f"输出中找不到指标{metric['name']}；pattern={pattern}")
        return None
    selected = matches[0] if metric.get("match", "last") == "first" else matches[-1]
    if "value" in selected.groupdict():
        raw = selected.group("value")
    elif selected.lastindex:
        raw = selected.group(1)
    else:
        raw = selected.group(0)
    return float(raw)


def aggregate(samples, method):
    if method == "median":
        return statistics.median(samples)
    if method == "mean":
        return statistics.fmean(samples)
    return min(samples)


def slug(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or "case"


def write_run_log(path, command, cwd, env_overrides, returncode, output):
    lines = [
        f"UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"PWD: {cwd}",
        f"COMMAND: {shlex.join(command)}",
    ]
    if env_overrides:
        lines.append("ENV: " + " ".join(
            f"{key}={shlex.quote(value)}" for key, value in sorted(env_overrides.items())))
    lines.extend([f"EXIT_CODE: {returncode}", "", output])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_one(command, cwd, env_overrides, timeout, log_path):
    env = os.environ.copy()
    env.update(env_overrides)
    try:
        completed = subprocess.run(
            command, cwd=cwd, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, check=False)
        output, returncode = completed.stdout, completed.returncode
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or "") + f"\nTIMEOUT: {timeout}s\n"
        returncode = 124
    write_run_log(log_path, command, cwd, env_overrides, returncode, output)
    return returncode, output


def select_items(config, args):
    cases = config["cases"]
    if args.case:
        cases = [case for case in cases
                 if any(fnmatch.fnmatch(case["name"], pattern) for pattern in args.case)]
    implementations = config["implementations"]
    if args.implementation:
        selected = set(args.implementation)
        implementations = [item for item in implementations if item["name"] in selected]
    if not cases:
        raise ValueError("case过滤后为空")
    names = [item["name"] for item in implementations]
    if config["baseline"] not in names:
        raise ValueError("筛选implementation时必须包含baseline")
    if len(implementations) < 2:
        raise ValueError("至少需要baseline和一个对比实现")
    return cases, implementations


def run_benchmarks(config, config_path, output_prefix, args):
    cases, implementations = select_items(config, args)
    repeat = args.repeat or int(config.get("repeat", 1))
    method = args.aggregate or config.get("aggregate", "median")
    if repeat <= 0:
        raise ValueError("repeat必须大于0")
    log_dir = output_prefix.parent / f"{output_prefix.name}-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases:
        context = merge_context(config, case, config_path, output_prefix)
        case_result = {
            "case": case["name"],
            "dimensions": case.get("dimensions", {}),
            "implementations": {},
        }
        for implementation in implementations:
            name = implementation["name"]
            command = resolve_command(config, case, implementation, context)
            cwd = resolve_cwd(config, case, implementation, context)
            env_overrides = resolve_env(config, case, implementation, context)
            timeout = case.get("timeout", implementation.get(
                "timeout", config.get("timeout", None)))
            print(f"[{case['name']}][{name}] {shlex.join(command)}", flush=True)
            if args.dry_run:
                case_result["implementations"][name] = {
                    "status": "dry-run", "command": command, "metrics": {}}
                continue
            samples = {metric["name"]: [] for metric in config["metrics"]}
            status, error = "ok", ""
            for index in range(repeat):
                log_path = log_dir / (
                    f"{slug(case['name'])}__{slug(name)}__run-{index + 1}.log")
                returncode, output = run_one(
                    command, cwd, env_overrides, timeout, log_path)
                if returncode != 0:
                    status, error = "failed", f"退出码{returncode}，日志：{log_path}"
                    break
                try:
                    for metric in config["metrics"]:
                        value = parse_metric(output, metric, implementation)
                        if value is not None:
                            samples[metric["name"]].append(value)
                except ValueError as exception:
                    status, error = "failed", f"{exception}，日志：{log_path}"
                    break
            metrics = {}
            if status == "ok":
                for metric in config["metrics"]:
                    values = samples[metric["name"]]
                    if values:
                        metrics[metric["name"]] = {
                            "value": aggregate(values, method),
                            "samples": values,
                            "min": min(values),
                            "max": max(values),
                            "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
                        }
            case_result["implementations"][name] = {
                "status": status,
                "error": error,
                "command": command,
                "metrics": metrics,
            }
            if status != "ok" and not args.keep_going:
                raise RuntimeError(f"{case['name']}/{name}: {error}")
        results.append(case_result)
    return results, implementations, repeat, method


def comparison_rows(config, results, implementations):
    baseline = config["baseline"]
    candidates = [item["name"] for item in implementations if item["name"] != baseline]
    rows = []
    for result in results:
        base_result = result["implementations"].get(baseline, {})
        for candidate in candidates:
            candidate_result = result["implementations"].get(candidate, {})
            for metric in config["metrics"]:
                name = metric["name"]
                base_metric = base_result.get("metrics", {}).get(name)
                candidate_metric = candidate_result.get("metrics", {}).get(name)
                baseline_value = base_metric["value"] if base_metric else None
                candidate_value = candidate_metric["value"] if candidate_metric else None
                speedup = None
                improvement = None
                if baseline_value is not None and candidate_value is not None:
                    if metric.get("goal", "min") == "min" and candidate_value != 0:
                        speedup = baseline_value / candidate_value
                        if baseline_value != 0:
                            improvement = ((baseline_value - candidate_value) /
                                           baseline_value * 100.0)
                    elif metric.get("goal", "min") == "max" and baseline_value != 0:
                        speedup = candidate_value / baseline_value
                        improvement = ((candidate_value - baseline_value) /
                                       baseline_value * 100.0)
                rows.append({
                    "case": result["case"],
                    **result["dimensions"],
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": name,
                    "unit": metric.get("unit", ""),
                    "goal": metric.get("goal", "min"),
                    "baseline_value": baseline_value,
                    "candidate_value": candidate_value,
                    "speedup": speedup,
                    "improvement_pct": improvement,
                })
    return rows


def all_dimensions(config):
    dimensions = []
    for case in config["cases"]:
        for name in case.get("dimensions", {}):
            if name not in dimensions:
                dimensions.append(name)
    return dimensions


def display(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def md_escape(value):
    return display(value).replace("|", "\\|").replace("\n", " ")


def make_markdown(config, comparisons, dimensions, repeat, method):
    lines = [
        f"# {config['name']}", "",
        f"> baseline：`{config['baseline']}`；外层重复：{repeat}；聚合：{method}。"
        "Speedup大于1表示candidate更快。", "",
        "## 汇总", "",
        "| candidate | metric | 有效case | 胜/平/负 | 几何平均speedup | 平均优化率 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    groups = {}
    for row in comparisons:
        if row["speedup"] is not None and row["speedup"] > 0:
            groups.setdefault((row["candidate"], row["metric"]), []).append(row)
    for (candidate, metric), rows in groups.items():
        speedups = [row["speedup"] for row in rows]
        wins = sum(value > 1.000001 for value in speedups)
        ties = sum(abs(value - 1.0) <= 0.000001 for value in speedups)
        losses = len(speedups) - wins - ties
        geomean = math.exp(statistics.fmean(math.log(value) for value in speedups))
        improvement = statistics.fmean(row["improvement_pct"] for row in rows)
        lines.append(
            f"| {md_escape(candidate)} | {md_escape(metric)} | {len(rows)} | "
            f"{wins}/{ties}/{losses} | {geomean:.4f}x | {improvement:+.2f}% |")
    lines.extend(["", "## 各维度明细", ""])
    columns = ["case", *dimensions, "baseline", "candidate", "metric", "unit",
               "baseline_value", "candidate_value", "speedup", "improvement_pct"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in comparisons:
        rendered = dict(row)
        rendered["speedup"] = (f"{row['speedup']:.4f}x"
                               if row["speedup"] is not None else "-")
        rendered["improvement_pct"] = (
            f"{row['improvement_pct']:+.2f}%"
            if row["improvement_pct"] is not None else "-")
        lines.append("| " + " | ".join(
            md_escape(rendered.get(column, "-")) for column in columns) + " |")
    lines.append("")
    return "\n".join(lines)


def write_reports(config, config_path, output_prefix, results, implementations,
                  repeat, method):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    dimensions = all_dimensions(config)
    comparisons = comparison_rows(config, results, implementations)
    json_path = Path(str(output_prefix) + ".json")
    results_csv = output_prefix.with_name(output_prefix.name + "-results.csv")
    comparison_csv = output_prefix.with_name(output_prefix.name + "-comparison.csv")
    md_path = Path(str(output_prefix) + ".md")

    payload = {
        "name": config["name"],
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": str(config_path.resolve()),
        "baseline": config["baseline"],
        "repeat": repeat,
        "aggregate": method,
        "results": results,
        "comparisons": comparisons,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    result_columns = ["case", *dimensions, "implementation", "status", "metric",
                      "value", "min", "max", "stdev", "samples"]
    with results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_columns)
        writer.writeheader()
        for result in results:
            for implementation, item in result["implementations"].items():
                if not item["metrics"]:
                    writer.writerow({
                        "case": result["case"], **result["dimensions"],
                        "implementation": implementation, "status": item["status"]})
                for metric, values in item["metrics"].items():
                    writer.writerow({
                        "case": result["case"], **result["dimensions"],
                        "implementation": implementation, "status": item["status"],
                        "metric": metric, "value": values["value"],
                        "min": values["min"], "max": values["max"],
                        "stdev": values["stdev"],
                        "samples": json.dumps(values["samples"]),
                    })

    comparison_columns = ["case", *dimensions, "baseline", "candidate", "metric",
                          "unit", "goal", "baseline_value", "candidate_value",
                          "speedup", "improvement_pct"]
    with comparison_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_columns)
        writer.writeheader()
        writer.writerows(comparisons)

    markdown = make_markdown(config, comparisons, dimensions, repeat, method)
    md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"JSON: {json_path}")
    print(f"CSV明细: {results_csv}")
    print(f"CSV对比: {comparison_csv}")
    print(f"Markdown: {md_path}")


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    config = load_config(config_path)
    results, implementations, repeat, method = run_benchmarks(
        config, config_path, output_prefix, args)
    if not args.dry_run:
        write_reports(config, config_path, output_prefix, results,
                      implementations, repeat, method)
    return 0


if __name__ == "__main__":
    sys.exit(main())
