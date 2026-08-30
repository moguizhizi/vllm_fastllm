#!/usr/bin/env python3
"""W8A8顺序执行的vLLM/FastLLM前向正确性检查入口。"""

import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "test" / "nvfp4"))

import forward_check_vllm as shared_forward  # noqa: E402


if __name__ == "__main__":
    if "--label" not in sys.argv:
        sys.argv.extend(("--label", "W8A8"))
    if "--target-trace-pattern" not in sys.argv:
        sys.argv.extend((
            "--target-trace-pattern",
            "[fastllm][w8a8] path=w8a8-cutlass",
        ))
    raise SystemExit(shared_forward.main())
