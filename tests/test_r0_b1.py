"""CPU-only real-HTTP transport tests. Does not import vLLM or allocate CUDA."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import unittest
from tempfile import TemporaryDirectory

from scripts.r0_b1 import canonical, digest, load_requests, percentiles, send_one, summarize


WORK = {"id": "W2", "prompt_tokens": 3, "output_tokens": 4,
        "concurrency": 2, "unique_requests": 1, "repetitions": 1,
        "warmup_requests_per_repetition": 1,
        "ttft_slo_ms": 1000, "tpot_slo_ms": 1000}


class Handler(BaseHTTPRequestHandler):
    seen = []
    fail_mode = None

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/version":
            output = {"version": "0.32.0"}
        elif self.path == "/v1/models":
            output = {"data": [{"id": "model-test"}]}
        else:
            self.send_error(404)
            return
        data = json.dumps(output).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.seen.append(payload)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        ids = payload["prompt"] if Handler.fail_mode != "wrong_prompt" else [7, 8, 9]
        for j, token in enumerate([11, 12, 13]):
            choice = {"index": 0, "text": "a", "token_ids": [token],
                      "finish_reason": "stop" if j == 2 else None}
            if j == 0:
                choice["prompt_token_ids"] = ids
            chunk = {"choices": [choice], "usage": None}
            try:
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.flush()
            except BrokenPipeError:
                return
            time.sleep(.003)
        if Handler.fail_mode != "incomplete":
            chunk = {"choices": [], "usage": {"prompt_tokens": len(payload["prompt"]),
                                                "completion_tokens": 3}}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:" + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=1)

    def setUp(self):
        Handler.fail_mode = None
        Handler.seen = []
        ids = [1, 2, 3]
        self.row = {"request_id": "W2-0", "prompt": "example",
                    "prompt_sha256": hashlib.sha256(b"example").hexdigest(),
                    "prompt_tokens": 3, "output_tokens": 4,
                    "token_ids_sha256": digest(ids), "_ids": ids}

    def test_stream_and_real_token_timing(self):
        outcome = send_one(self.row, WORK, 0, "measured", time.monotonic_ns(),
                           self.base, "model-test", 5)
        self.assertEqual(outcome["status"], "success")
        self.assertEqual(outcome["output_tokens_observed"], 3)
        self.assertEqual(len(outcome["itl_ms"]), 2)
        self.assertGreater(outcome["tpot_ms"], 0)
        self.assertEqual(outcome["server_token_ids_sha256"], digest([1, 2, 3]))
        self.assertEqual(outcome["token_identity_evidence_stage"], "tokenize_preflight_only")
        self.assertEqual(Handler.seen[0]["return_token_ids"], True)
        self.assertEqual(Handler.seen[0]["stream_interval"], 1)

    def test_mismatch_preserved_as_failure(self):
        Handler.fail_mode = "wrong_prompt"
        outcome = send_one(self.row, WORK, 0, "measured", time.monotonic_ns(),
                           self.base, "model-test", 5)
        self.assertEqual(outcome["status"], "error")
        self.assertEqual(outcome["error_code"], "SERVER_PROMPT_TOKEN_IDS_MISMATCH")

    def test_incomplete_stream_not_success(self):
        Handler.fail_mode = "incomplete"
        outcome = send_one(self.row, WORK, 0, "measured", time.monotonic_ns(),
                           self.base, "model-test", 5)
        self.assertEqual(outcome["status"], "error")
        self.assertEqual(outcome["error_code"], "INCOMPLETE_STREAM")

    def test_summary_denominators_and_missing_itl(self):
        ok = send_one(self.row, WORK, 0, "measured", time.monotonic_ns(),
                      self.base, "model-test", 5)
        bad = dict(ok, request_id="W2-1", status="error", error_code="FAIL",
                   tpot_ms=None)
        stats = summarize([{"measured_wall_s": 2.0, "records": [ok, bad]}],
                          {**WORK, "unique_requests": 2})
        self.assertEqual(stats["observed_measured_n"], 2)
        self.assertEqual(stats["success_n"], 1)
        self.assertEqual(stats["error_n"], 1)
        self.assertEqual(stats["slo_pass_n"], 1)
        self.assertEqual(stats["slo_goodput_requests_per_second"], 0.5)
        self.assertEqual(percentiles([])["n"], 0)


if __name__ == "__main__":
    unittest.main()
