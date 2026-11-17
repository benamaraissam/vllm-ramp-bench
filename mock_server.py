#!/usr/bin/env python3
"""
Faux serveur OpenAI-compatible pour tester bench.py sans GPU.

Il simule un prefill proportionnel aux tokens non cachés (prefix cache par
préfixe de messages) et un débit de génération fixe.

    python mock_server.py --port 8021
    python bench.py --url http://127.0.0.1:8021/v1 --steps 2,4 --step-duration 20 \
        --think-min 1 --think-max 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CHARS_PER_TOKEN = 3.2
ARGS: argparse.Namespace
_seen: set[str] = set()
_lock = threading.Lock()


def _count(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def _prompt_stats(messages: list[dict]) -> tuple[int, int]:
    """Retourne (tokens du prompt, tokens déjà en cache)."""
    total = cached = 0
    h = hashlib.sha256()
    hit = True
    keys = []
    for m in messages:
        content = m.get("content") or ""
        n = _count(content) + 4
        h.update(m.get("role", "").encode() + b"\0" + content.encode() + b"\1")
        key = h.hexdigest()
        keys.append(key)
        with _lock:
            if hit and key in _seen:
                cached += n
            else:
                hit = False
        total += n
    with _lock:
        _seen.update(keys)
    return total, cached


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # silence
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        prompt_tokens, cached = _prompt_stats(body.get("messages", []))
        max_tokens = int(body.get("max_tokens") or 16)
        time.sleep((prompt_tokens - cached) / ARGS.prefill_tok_s)
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens,
                 "total_tokens": prompt_tokens + max_tokens,
                 "prompt_tokens_details": {"cached_tokens": cached}}
        if not body.get("stream"):
            payload = json.dumps({
                "id": "mock", "object": "chat.completion", "model": body.get("model"),
                "choices": [{"index": 0, "finish_reason": "length",
                             "message": {"role": "assistant", "content": "ok " * max_tokens}}],
                "usage": usage}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def send(obj) -> None:
            data = obj if isinstance(obj, str) else json.dumps(obj)
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()

        try:
            for i in range(max_tokens):
                # Les premiers tokens sortent comme du raisonnement, comme avec
                # un reasoning parser.
                key = "reasoning_content" if i < 3 else "content"
                send({"choices": [{"index": 0, "delta": {key: "tok "}}]})
                time.sleep(1.0 / ARGS.decode_tok_s)
            send({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
            send({"choices": [], "usage": usage})
            send("[DONE]")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> None:
    global ARGS
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8021)
    p.add_argument("--prefill-tok-s", type=float, default=200_000.0)
    p.add_argument("--decode-tok-s", type=float, default=400.0)
    ARGS = p.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    server.daemon_threads = True
    print(f"mock OpenAI server on http://127.0.0.1:{ARGS.port}/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()
