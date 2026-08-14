#!/usr/bin/env python3
"""覆盖vLLM CUTLASS FP8 scaled-mm官方形状和缩放组合。"""

import gc

import torch
from vllm import _custom_ops as ops


ALIGNED = [
    (1, 256, 128), (1, 16384, 1024), (1, 24576, 496),
    (16, 256, 496), (16, 16384, 128), (16, 24576, 4096),
    (32, 8192, 4096), (32, 16384, 4096), (33, 1024, 1024),
    (33, 8192, 128), (64, 2048, 496), (64, 16384, 1024),
    (100, 8192, 496), (128, 32768, 4096), (256, 4096, 4096),
    (512, 256, 1024), (512, 8192, 4096), (512, 16384, 128),
    (512, 24576, 128),
]
UNALIGNED = [
    (32, 3420, 1280), (32, 1280, 6840), (1, 3420, 1280),
    (64, 6840, 1280), (16, 100, 200), (33, 255, 513),
]


def expanded_scale(scale, rows, columns, side):
    if scale.numel() == 1:
        return scale
    if side == "a":
        return scale.view(rows, 1)
    return scale.view(1, columns)


def check_case(m, n, k, per_token, per_channel, use_bias,
               out_dtype=torch.bfloat16):
    fp8 = torch.float8_e4m3fn
    a = torch.empty((m, k), device="cuda", dtype=torch.bfloat16).normal_().to(fp8)
    b = torch.empty((n, k), device="cuda", dtype=torch.bfloat16).normal_().to(fp8).t()
    scale_a = torch.empty(
        (m, 1) if per_token else (1,), device="cuda",
        dtype=torch.float32).uniform_(0.001, 0.01)
    scale_b = torch.empty(
        (1, n) if per_channel else (1,), device="cuda",
        dtype=torch.float32).uniform_(0.001, 0.01)
    bias = (torch.empty((n,), device="cuda", dtype=out_dtype).uniform_(-0.1, 0.1)
            if use_bias else None)
    actual = ops.cutlass_scaled_mm(
        a, b, scale_a, scale_b, out_dtype, bias)
    expected = (
        a.float() * expanded_scale(scale_a, m, k, "a")) @ (
        b.float() * expanded_scale(scale_b, k, n, "b"))
    if bias is not None:
        expected += bias.float()
    torch.testing.assert_close(
        actual.float(), expected, rtol=5e-1, atol=1.5e-1)


def check_unaligned_case(m, n, k, per_token, per_channel, use_bias):
    """按vLLM高层Linear kernel的真实padding路径检查非对齐形状。"""
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
        CutlassFP8ScaledMMLinearKernel,
    )

    fp8 = torch.float8_e4m3fn
    a = torch.empty((m, k), device="cuda", dtype=torch.bfloat16).normal_().to(fp8)
    b = torch.empty((n, k), device="cuda", dtype=torch.bfloat16).normal_().to(fp8).t()
    scale_a = torch.empty(
        (m, 1) if per_token else (1,), device="cuda",
        dtype=torch.float32).uniform_(0.001, 0.01)
    scale_b = torch.empty(
        (1, n) if per_channel else (1,), device="cuda",
        dtype=torch.float32).uniform_(0.001, 0.01)
    bias = (torch.empty((n,), device="cuda", dtype=torch.bfloat16)
            .uniform_(-0.1, 0.1) if use_bias else None)
    expected = (
        a.float() * expanded_scale(scale_a, m, k, "a")) @ (
        b.float() * expanded_scale(scale_b, k, n, "b"))
    if bias is not None:
        expected += bias.float()

    # 与vLLM process_weights_after_loading保持一致：权重和per-channel
    # scale先补齐，apply_scaled_mm再负责激活K维及输出N维的padding/slice。
    pad_k = (16 - k % 16) % 16
    pad_n = (16 - n % 16) % 16
    if pad_k or pad_n:
        b = torch.nn.functional.pad(
            b.t().contiguous(), (0, pad_k, 0, pad_n)).t()
        if pad_n and scale_b.numel() > 1:
            scale_b = torch.nn.functional.pad(scale_b, (0, pad_n), value=1.0)
    kernel = object.__new__(CutlassFP8ScaledMMLinearKernel)
    kernel.logical_output_size = n
    actual = kernel.apply_scaled_mm(
        A=a, B=b, out_dtype=torch.bfloat16, As=scale_a, Bs=scale_b,
        bias=bias, output_shape=[m, n])
    torch.testing.assert_close(
        actual.float(), expected, rtol=5e-1, atol=1.5e-1)


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("本测试要求SM120 GPU")
    total = 0
    # 与vLLM官方test_cutlass_fp8_gemm一致，覆盖两种A scale、两种B scale和bias。
    for m, n, k in ALIGNED:
        for per_token in (True, False):
            for per_channel in (True, False):
                for use_bias in (False, True):
                    check_case(m, n, k, per_token, per_channel, use_bias)
                    total += 1
                    print(
                        f"PASS m={m} n={n} k={k} per_token={per_token} "
                        f"per_channel={per_channel} bias={use_bias}", flush=True)
                    torch.cuda.empty_cache()
                    gc.collect()
    for m, n, k in UNALIGNED:
        for per_token in (True, False):
            for per_channel in (True, False):
                for use_bias in (False, True):
                    check_unaligned_case(
                        m, n, k, per_token, per_channel, use_bias)
                    total += 1
                    print(
                        f"PASS padded m={m} n={n} k={k} "
                        f"per_token={per_token} per_channel={per_channel} "
                        f"bias={use_bias}", flush=True)
                    torch.cuda.empty_cache()
                    gc.collect()
    # 对应vLLM output-dtype case；FastLLM侧也独立覆盖FP16/BF16输出。
    for out_dtype in (torch.bfloat16, torch.float16):
        check_case(512, 512, 512, True, True, True, out_dtype)
        total += 1
        print(f"PASS output_dtype={out_dtype}", flush=True)
    print(f"Summary: PASS ({total} vLLM FP8 scaled-mm cases)")


if __name__ == "__main__":
    main()
