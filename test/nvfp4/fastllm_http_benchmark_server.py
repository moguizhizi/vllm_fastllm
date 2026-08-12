#!/usr/bin/env python3
"""为模型性能对比提供直接消费Token ID的FastLLM HTTP流式接口。"""

import argparse
import ctypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--atype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batch", type=int, required=True)
    return parser.parse_args()


def write_sse(handler, payload):
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    handler.wfile.write(f"data: {data}\n\n".encode("utf-8"))
    handler.wfile.flush()


def make_handler(model):
    from ftllm import llm

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            sys.stderr.write("[fastllm-http] " + fmt % args + "\n")

        def send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/v1/models":
                self.send_json(404, {"error": "not found"})
                return
            self.send_json(200, {"data": [{"id": "nvfp4-performance"}]})

        def do_POST(self):
            if self.path != "/v1/completions":
                self.send_json(404, {"error": "not found"})
                return
            try:
                trace_cpu = os.environ.get(
                    "FASTLLM_HTTP_CPU_TRACE", "").lower() in ("1", "true", "on")
                request_begin_ns = time.perf_counter_ns() if trace_cpu else 0
                launch_us = 0
                wait_fetch_us = 0
                statistics_us = 0
                sse_us = 0
                wait_fetch_calls = 0
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                input_tokens = request.get("prompt")
                output_tokens = int(request.get("max_tokens", 0))
                if (not isinstance(input_tokens, list) or
                        not all(isinstance(token, int) for token in input_tokens)):
                    raise ValueError("prompt必须是Token ID数组")
                if output_tokens <= 0:
                    raise ValueError("max_tokens必须大于0")

                token_buffer = (ctypes.c_int * len(input_tokens))(*input_tokens)
                empty_stops = (ctypes.c_int * 0)()
                # min_length与max_length相同，从logits中屏蔽EOS，保证双方都
                # 严格生成指定数量的Token。
                stage_begin_ns = time.perf_counter_ns() if trace_cpu else 0
                handle = llm.fastllm_lib.launch_response_llm_model(
                    model.model, len(input_tokens), token_buffer,
                    ctypes.c_int(output_tokens), ctypes.c_int(output_tokens),
                    ctypes.c_bool(False), ctypes.c_float(1.0), ctypes.c_int(1),
                    ctypes.c_float(1.0), ctypes.c_float(1.0), ctypes.c_bool(False),
                    ctypes.c_int(0), empty_stops)
                if trace_cpu:
                    launch_us = (time.perf_counter_ns() - stage_begin_ns) // 1000

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                generated = 0
                first = True
                statistics = {}
                token_batch = (ctypes.c_int * 1)()
                while True:
                    # 使用正式导出的阻塞式取Token入口。其内部通过条件变量等待，
                    # 等待期间会释放调度器共用的dictLocker，避免HTTP轮询线程每
                    # 0.5ms抢锁。每次只取一个Token，保持逐Token SSE和ITL语义。
                    stage_begin_ns = time.perf_counter_ns() if trace_cpu else 0
                    fetched = llm.fastllm_lib.fetch_response_tokens_batch_llm_model(
                        model.model, handle, token_batch, 1)
                    if trace_cpu:
                        wait_fetch_us += (
                            time.perf_counter_ns() - stage_begin_ns) // 1000
                        wait_fetch_calls += 1
                    # 超时只用于兼容旧调度器遗漏通知的情况；继续阻塞等待，
                    # 不恢复CanFetchResponse忙轮询。
                    if fetched == 0:
                        continue
                    if fetched < 0:
                        break
                    token = int(token_batch[0])
                    # 最后一个真实Token取出后ResponseContext仍然存在；下一次
                    # 取Token发现结束状态时才会删除，因此这里可安全保存统计。
                    stage_begin_ns = time.perf_counter_ns() if trace_cpu else 0
                    statistics = model.get_response_statistics(handle) or statistics
                    if trace_cpu:
                        statistics_us += (time.perf_counter_ns() - stage_begin_ns) // 1000
                    choice = {"index": 0, "token_ids": [token], "text": ""}
                    if first:
                        choice["prompt_token_ids"] = input_tokens
                        first = False
                    stage_begin_ns = time.perf_counter_ns() if trace_cpu else 0
                    write_sse(self, {"choices": [choice]})
                    if trace_cpu:
                        sse_us += (time.perf_counter_ns() - stage_begin_ns) // 1000
                    generated += 1
                cached = max(0, int(statistics.get("cached_input_tokens", 0)))
                missed = max(
                    0, int(statistics.get(
                        "missed_input_tokens", len(input_tokens) - cached)))
                stage_begin_ns = time.perf_counter_ns() if trace_cpu else 0
                write_sse(self, {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": len(input_tokens),
                        "completion_tokens": generated,
                        "total_tokens": len(input_tokens) + generated,
                        "prompt_tokens_details": {
                            "cached_tokens": cached,
                            "missed_tokens": missed,
                        },
                    },
                })
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                if trace_cpu:
                    sse_us += (time.perf_counter_ns() - stage_begin_ns) // 1000
                    total_us = (time.perf_counter_ns() - request_begin_ns) // 1000
                    print(
                        "[fastllm][http][cpu] "
                        f"prompt={len(input_tokens)} generated={generated} "
                        f"launch_us={launch_us} wait_fetch_calls={wait_fetch_calls} "
                        f"wait_fetch_us={wait_fetch_us} "
                        f"statistics_us={statistics_us} "
                        f"sse_us={sse_us} total_us={total_us}",
                        file=sys.stderr,
                        flush=True,
                    )
                self.close_connection = True
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception as error:
                self.send_json(400, {"error": str(error)})

    return Handler


def main():
    args = parse_args()
    from ftllm import llm

    llm.set_device_map(args.device)
    llm.set_device_map(args.device, True)
    model = llm.model(args.model, dtype=args.dtype)
    model.set_atype(args.atype)
    model.set_max_batch(args.max_batch)
    model.warmup()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model))
    server.daemon_threads = True
    print(
        f"FastLLM benchmark HTTP server ready: http://{args.host}:{args.port}",
        flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        model.release_memory()


if __name__ == "__main__":
    main()
