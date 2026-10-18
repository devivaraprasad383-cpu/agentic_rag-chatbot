"""Tiny fake of the RAG API so the benchmark harness can be smoke-tested in CI (no LLM/keys)."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

CHUNKS = ["Total revenue was $5.2 billion in FY2023.", "The CEO is Jane Doe.", "Risks include FX exposure."]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._json({"/stats/usage": {"total_prompt_tokens": 1000, "total_cached_tokens": 400,
                                     "avg_cost_per_query": 0.008},
                    "/stats/cache": {"response_cache": {"total_hits": 1}}}.get(self.path, {}))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/session":
            return self._json({"session_id": "s"})
        q = body["query"].lower()
        ev = []
        if "ignore all previous" in q or "weather" in q:
            ev = [{"type": "safety_block", "layer": "input_guard"}, {"type": "done"}]
        else:
            ev = [{"type": "node", "node": "retriever"}, {"type": "sources", "sources": CHUNKS},
                  {"type": "node", "node": "answer_generator"}, {"type": "token", "token": "Revenue was $5.2 billion."},
                  {"type": "evaluation", "data": {"faithfulness": 5, "answer_relevance": 4, "context_relevance": 4}},
                  {"type": "done"}]
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for e in ev: self.wfile.write(f"data: {json.dumps(e)}\n\n".encode())


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8765), H).serve_forever()
