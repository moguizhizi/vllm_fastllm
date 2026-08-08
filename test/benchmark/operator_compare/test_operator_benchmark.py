#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("operator_benchmark.py")
SPEC = importlib.util.spec_from_file_location("operator_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OperatorBenchmarkTest(unittest.TestCase):
    def test_example_report(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "report"
            subprocess.run([
                sys.executable, str(SCRIPT),
                "--config", str(Path(__file__).with_name("example.json")),
                "--output-prefix", str(prefix),
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            payload = json.loads(Path(str(prefix) + ".json").read_text(encoding="utf-8"))
            latency = next(row for row in payload["comparisons"]
                           if row["case"] == "demo_m1" and
                           row["metric"] == "latency_ms")
            throughput = next(row for row in payload["comparisons"]
                              if row["case"] == "demo_m1" and
                              row["metric"] == "throughput")
            self.assertAlmostEqual(latency["speedup"], 2.0)
            self.assertAlmostEqual(latency["improvement_pct"], 50.0)
            self.assertAlmostEqual(throughput["speedup"], 2.0)
            self.assertAlmostEqual(throughput["improvement_pct"], 100.0)
            self.assertTrue(Path(str(prefix) + ".md").is_file())
            self.assertTrue(prefix.with_name("report-results.csv").is_file())
            self.assertTrue(prefix.with_name("report-comparison.csv").is_file())

    def test_metric_implementation_override(self):
        metric = {
            "name": "latency_ms",
            "pattern": "default=(?P<value>[0-9.]+)",
            "patterns": {"vllm": "vllm=(?P<value>[0-9.]+)"},
        }
        implementation = {"name": "vllm"}
        self.assertEqual(MODULE.parse_metric("vllm=1.25", metric, implementation), 1.25)


if __name__ == "__main__":
    unittest.main()
