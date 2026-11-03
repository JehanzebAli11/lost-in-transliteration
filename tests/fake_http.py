"""A local HTTP server that speaks the OpenAI chat-completions and Anthropic messages
wire formats, answering with tests.fakes.FakeLLM. Lets tests run the *real* SDK clients."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tests.fakes import FakeLLM


def serve(llm: FakeLLM | None = None):
    llm = llm or FakeLLM()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.endswith("/models"):
                return self._send({"object": "list", "data": [
                    {"id": "fake/gemma-4-31b-it", "object": "model", "created": 0, "owned_by": "x"}]})
            self._send({}, 404)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/chat/completions"):
                sys_ = next((m["content"] for m in req["messages"] if m["role"] == "system"), None)
                user = [m["content"] for m in req["messages"] if m["role"] == "user"][-1]
                r = llm.chat(model=req["model"], system=sys_, user=user, temperature=req.get("temperature"),
                             max_tokens=req.get("max_tokens"), seed=req.get("seed"))
                return self._send({"id": "x", "object": "chat.completion", "created": 0, "model": req["model"],
                                   "choices": [{"index": 0, "finish_reason": "stop",
                                                "message": {"role": "assistant", "content": r.text}}],
                                   "usage": {"prompt_tokens": r.input_tokens, "completion_tokens": r.output_tokens,
                                             "total_tokens": r.input_tokens + r.output_tokens}})
            if self.path.endswith("/messages"):
                user = req["messages"][-1]["content"]
                user = user if isinstance(user, str) else user[0]["text"]
                r = llm.chat(model=req["model"], system=req.get("system"), user=user,
                             temperature=req.get("temperature"), max_tokens=req["max_tokens"])
                return self._send({"id": "msg_x", "type": "message", "role": "assistant", "model": req["model"],
                                   "content": [{"type": "text", "text": r.text}], "stop_reason": "end_turn",
                                   "stop_sequence": None,
                                   "usage": {"input_tokens": r.input_tokens, "output_tokens": r.output_tokens}})
            self._send({}, 404)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}", llm
