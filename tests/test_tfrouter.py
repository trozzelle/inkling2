import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ink2q import tfrouter


def sse(obj) -> str:
    return "data: " + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n"


def role_chunk() -> str:
    return sse({"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})


def text_chunk(text="a") -> str:
    return sse({"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})


def finish_chunk() -> str:
    return sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})


def usage_chunk(prompt, cached=None) -> str:
    u = {"prompt_tokens": prompt, "completion_tokens": 1, "total_tokens": prompt + 1}
    if cached is not None:
        u["prompt_tokens_details"] = {"cached_tokens": cached}
    return sse({"choices": [], "usage": u})


def test_request_record_times_the_first_text_chunk_not_the_role_chunk():
    ev = [(0.01, role_chunk()), (2.0, text_chunk()), (2.0, finish_chunk()), (2.0, usage_chunk(8000, 0)), (2.0, sse("[DONE]"))]
    r = tfrouter.request_record(ev, 0.0)
    assert r["basis"] == "text" and r["prefill_s"] == 2.0 and r["first_chunk"] == 0.01
    assert r["prefill_tok_s"] == pytest.approx(4000.0) and r["prompt_tokens"] == 8000 and r["cached_tokens"] == 0


def test_request_record_falls_back_to_the_finish_chunk_when_no_text_streams_and_reads_no_cached_field_as_none():
    ev = [(0.01, role_chunk()), (1.6, finish_chunk()), (1.6, usage_chunk(8000))]
    r = tfrouter.request_record(ev, 0.0)
    assert r["basis"] == "finish" and r["prefill_s"] == 1.6 and r["text_chunk"] is None and r["cached_tokens"] is None
    with pytest.raises(RuntimeError, match="neither"):
        tfrouter.request_record([(0.1, role_chunk()), (0.2, usage_chunk(5))], 0.0)


def rec(seq, tok_s, cached=0, pt=8191):
    return {"seq": seq, "prompt_tokens": pt, "prefill_tok_s": tok_s, "cached_tokens": cached}


def test_summarize_reports_first_median_and_uncached_median_per_prompt_and_flags_prompt_token_mismatches():
    prompts = [{"seq": 0, "case": "a", "ids_length": 8192, "rendered_length": 8191},
               {"seq": 1, "case": "b", "ids_length": 8192, "rendered_length": 8191}]
    recs = [rec(0, 1000), rec(1, 900, pt=8000), rec(0, 5000, cached=8190), rec(1, 910, pt=8000), rec(0, 1100),
            rec(1, 920, pt=8000)]
    s = tfrouter.summarize(prompts, recs)
    a, b = s["per_prompt"]
    assert a["first_tok_s"] == 1000 and a["median_tok_s"] == 1100 and a["median_uncached_tok_s"] == 1050
    assert a["uncached_requests"] == 2 and a["cached_tokens"] == [0, 8190, 0]
    assert b["first_tok_s"] == 900 and b["median_tok_s"] == 910 and b["prompt_tokens"] == [8000]
    assert s["prompt_token_mismatches"] == [{"seq": 1, "case": "b", "server_prompt_tokens": [8000],
                                             "rendered_length": 8191, "ids_length": 8192}]


LOG = """[tensorfold] listening
[tensorfold] done chatcmpl-1 prompt=20 cached=0 thinking=False effort=none tokens=3 sha=ab finish=stop rounds=2 accepted=0/0 ttft=0.9 decode=0.1 elapsed=1.0 queued=0.000 tok/s=3.0
[tensorfold] done chatcmpl-2 prompt=8191 cached=0 thinking=False effort=none tokens=1 sha=cd finish=length rounds=1 accepted=0/0 ttft=4.25 decode=0.0 elapsed=4.3 queued=0.000 tok/s=0.0
[tensorfold] done chatcmpl-3 prompt=8191 cached=8190 thinking=False effort=none tokens=1 sha=cd finish=length rounds=1
[tensorfold] done chatcmpl-4 prompt=7000 cached=0 thinking=False effort=none tokens=1 sha=ef finish=length rounds=1 ttft=3.0
"""


def test_done_lines_and_pairing_with_the_requests_in_order_sent():
    log = tfrouter.done_lines(LOG)
    assert [(d["prompt"], d["cached"], d["ttft"]) for d in log] == [(20, 0, 0.9), (8191, 0, 4.25), (8191, 8190, None),
                                                                    (7000, 0, 3.0)]
    sent = [{"prompt_tokens": 20}, {"prompt_tokens": 8191}, {"prompt_tokens": 7000}, {"prompt_tokens": 8191}]
    paired = tfrouter.pair_with_log(sent, log)
    assert paired[0] == log[0] and paired[1] == log[1] and paired[2] == log[3]
    assert paired[3] is None  # the later 8191 line precedes the 7000 one that was already matched


def serve(handler_body):
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers["content-length"])
            body = json.loads(self.rfile.read(n))
            seen.append((self.headers.get("authorization"), body))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for chunk in handler_body(body):
                self.wfile.write(chunk.encode())
                self.wfile.flush()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def test_bench_through_a_fake_router_warms_up_then_runs_round_robin_and_writes_the_report(tmp_path, capsys):
    def reply(body):
        long = len(body["messages"][0]["content"]) > 50
        yield role_chunk()
        time.sleep(0.05 if long else 0.0)
        yield text_chunk()
        yield finish_chunk()
        yield usage_chunk(400 if long else 12, 0)
        yield sse("[DONE]")

    srv, seen = serve(reply)
    try:
        key = tmp_path / "key"
        key.write_text("sekrit\n")
        msgs = tmp_path / "m.json"
        msgs.write_text(json.dumps([{"seq": i, "set": "prefill8k", "case": f"p{i}", "ids_length": 401, "rendered_length": 400,
                                     "messages": [{"role": "user", "content": "x" * 100}]} for i in range(2)]))
        out = tmp_path / "out" / "r.json"
        tfrouter.main(["bench", "--base-url", f"http://127.0.0.1:{srv.server_port}/v1", "--model", "m", "--messages", str(msgs),
                       "--api-key-file", str(key), "--requests", "3", "--out", str(out)])
    finally:
        srv.shutdown()
    d = json.loads(out.read_text())
    assert [b["max_tokens"] for _, b in seen] == [8] + [1] * 6  # the warm-up, then six 8k-style requests
    assert {a for a, _ in seen} == {"Bearer sekrit"}
    assert [r["seq"] for r in d["records"]] == [0, 1, 0, 1, 0, 1]  # round-robin passes
    assert d["warmup"]["prompt_tokens"] == 12 and d["prompt_token_mismatches"] == []
    p0 = d["per_prompt"][0]
    assert p0["requests"] == 3 and p0["first_tok_s"] == pytest.approx(400 / d["records"][0]["prefill_s"])
    assert 0 < p0["median_tok_s"] and all(r["basis"] == "text" for r in d["records"])
    assert "per_prompt" in capsys.readouterr().out.splitlines()[-1]


def test_bench_records_a_failed_request_keeps_the_finished_ones_and_exits_nonzero(tmp_path):
    calls = []

    def reply(body):
        calls.append(body)
        n = len(calls)
        if n == 3:  # the third request gets a stream with no text and no finish chunk
            yield role_chunk()
            yield usage_chunk(400, 0)
            return
        yield role_chunk()
        yield text_chunk()
        yield finish_chunk()
        yield usage_chunk(400 if len(body["messages"][0]["content"]) > 50 else 12, 0)
        yield sse("[DONE]")

    srv, seen = serve(reply)
    try:
        key = tmp_path / "key"
        key.write_text("k")
        msgs = tmp_path / "m.json"
        msgs.write_text(json.dumps([{"seq": i, "case": f"p{i}", "ids_length": 401, "rendered_length": 400,
                                     "messages": [{"role": "user", "content": "x" * 100}]} for i in range(2)]))
        out = tmp_path / "r.json"
        with pytest.raises(SystemExit) as e:
            tfrouter.main(["bench", "--base-url", f"http://127.0.0.1:{srv.server_port}/v1", "--model", "m",
                           "--messages", str(msgs), "--api-key-file", str(key), "--requests", "2", "--out", str(out)])
    finally:
        srv.shutdown()
    assert e.value.code == 1
    d = json.loads(out.read_text())
    assert len(d["records"]) == 4 and len(d["errors"]) == 1  # warm-up was call 1; call 3 is the second 8k-style request
    assert d["errors"][0]["seq"] == 1 and d["errors"][0]["request"] == 0 and "neither" in d["errors"][0]["error"]
    assert [p["requests"] for p in d["per_prompt"]] == [2, 1]  # failed request left out of the rates
    assert not (tmp_path / "r.json.tmp").exists()


def test_bench_survives_an_http_error_and_writes_the_report_before_exiting(tmp_path):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            self.send_response(500)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        key = tmp_path / "key"
        key.write_text("k")
        msgs = tmp_path / "m.json"
        msgs.write_text(json.dumps([{"seq": 0, "case": "p", "messages": [{"role": "user", "content": "x"}]}]))
        out = tmp_path / "r.json"
        with pytest.raises(SystemExit):
            tfrouter.main(["bench", "--base-url", f"http://127.0.0.1:{srv.server_port}/v1", "--model", "m",
                           "--messages", str(msgs), "--api-key-file", str(key), "--requests", "1", "--out", str(out)])
    finally:
        srv.shutdown()
    d = json.loads(out.read_text())
    assert len(d["errors"]) == 2 and "HTTPError" in d["errors"][0]["error"]  # the warm-up and the one request
    assert d["per_prompt"][0]["requests"] == 0 and d["per_prompt"][0]["first_tok_s"] is None
