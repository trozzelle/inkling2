"""Served decode-rate measurement for the drafted-decode gate: streamed chat completions at one stream against a
TensorFold server, N requests a prompt. A request's decode time runs from its first to its last content or
reasoning chunk, its decoded tokens are completion_tokens - 1 (the first token ends the time to first token), and the
pooled rate is total decoded tokens over total decode time across every request."""
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


def parse_stream(events, t0: float) -> dict:
    """One request's record from (time, SSE line) pairs and the request's start time."""
    first = last = None
    usage, finish = None, None
    for t, line in events:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue
        d = json.loads(data)
        if d.get("error"):
            raise RuntimeError(f"server error in stream: {d['error']}")
        if d.get("usage"):
            usage = d["usage"]
        for c in d.get("choices") or []:
            delta = c.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                first = t if first is None else first
                last = t
            finish = c.get("finish_reason") or finish
    if usage is None:
        raise RuntimeError("stream carried no usage")
    return {"ttft": None if first is None else first - t0,
            "decode_time": None if first is None else last - first,
            "completion_tokens": usage["completion_tokens"], "prompt_tokens": usage["prompt_tokens"],
            "finish_reason": finish}


def decoded(rec: dict) -> int:
    return rec["completion_tokens"] - 1


def usable(rec: dict) -> bool:
    return rec["decode_time"] is not None and rec["decode_time"] > 0 and decoded(rec) > 0


def rate(recs: list[dict]) -> float | None:
    """Total decoded tokens over total decode time of the usable requests."""
    use = [r for r in recs if usable(r)]
    return sum(decoded(r) for r in use) / sum(r["decode_time"] for r in use) if use else None


def summarize(prompts: list[dict], records: list[dict]) -> dict:
    """prompts: {seq, set, case, ids_length?, rendered_length?}; records: per-request records with a seq."""
    per, check = [], {"mismatches": [], "unexplained": []}
    for p in prompts:
        recs = [r for r in records if r["seq"] == p["seq"]]
        served = sorted({r["prompt_tokens"] for r in recs})
        row = {"seq": p["seq"], "set": p["set"], "case": p["case"], "requests": len(recs), "rate": rate(recs),
               "decoded_tokens": sum(decoded(r) for r in recs if usable(r)),
               "decode_time": sum(r["decode_time"] for r in recs if usable(r)), "prompt_tokens": served}
        per.append(row)
        want = p.get("ids_length")
        if want is None or served == [want]:
            continue
        bad = {"seq": p["seq"], "case": p["case"], "server_prompt_tokens": served, "ids_length": want,
               "rendered_length": p.get("rendered_length")}
        check["mismatches"].append(bad)
        if served != [p.get("rendered_length")]:
            check["unexplained"].append(bad)
    rates = [r["rate"] for r in per if r["rate"] is not None]
    bench = [r["rate"] for r in per if r["set"] == "bench"]
    use = [r for r in records if usable(r)]
    return {"pooled_rate": rate(records), "median_prompt_rate": statistics.median(rates) if rates else None,
            "bench_rate": bench[0] if bench else None, "decoded_tokens": sum(decoded(r) for r in use),
            "decode_time": sum(r["decode_time"] for r in use), "requests": len(records),
            "unusable_requests": len(records) - len(use), "per_prompt": per, "prompt_tokens_check": check}


def request_body(a, entry: dict) -> dict:
    body = {"model": a.model, "messages": entry["messages"], "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": a.max_tokens, "temperature": a.temperature}
    if a.seed is not None:
        body["seed"] = a.seed
    for k in ("tools", "reasoning_effort"):
        if entry.get(k) is not None:
            body[k] = entry[k]
    return body


def run_request(a, entry: dict) -> dict:
    req = urllib.request.Request(a.base_url.rstrip("/") + "/chat/completions", json.dumps(request_body(a, entry)).encode(),
                                 {"content-type": "application/json"})
    t0, events = time.perf_counter(), []
    with urllib.request.urlopen(req, timeout=a.timeout) as r:
        for line in r:
            events.append((time.perf_counter(), line.decode("utf-8")))
    return {"seq": entry["seq"], **parse_stream(events, t0)}


def bench_cmd(a) -> None:
    entries = json.loads(a.messages.read_text())
    records = []
    for entry in entries:
        for i in range(a.requests):
            rec = {**run_request(a, entry), "request": i}
            records.append(rec)
            print(json.dumps({"seq": rec["seq"], "request": i, "ttft": rec["ttft"], "decode_time": rec["decode_time"],
                              "completion_tokens": rec["completion_tokens"], "prompt_tokens": rec["prompt_tokens"]}),
                  flush=True)
    prompts = [{k: e.get(k) for k in ("seq", "set", "case", "ids_length", "rendered_length")} for e in entries]
    out = {"base_url": a.base_url, "model": a.model, "requests_per_prompt": a.requests, "max_tokens": a.max_tokens,
           "temperature": a.temperature, "seed": a.seed, **summarize(prompts, records), "records": records}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("pooled_rate", "median_prompt_rate", "bench_rate", "unusable_requests")}
                     | {"prompt_token_mismatches": len(out["prompt_tokens_check"]["mismatches"])}))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="streamed one-stream decode rate over a messages file")
    b.add_argument("--base-url", required=True, help="server URL ending in /v1")
    b.add_argument("--model", required=True)
    b.add_argument("--messages", type=Path, required=True, help="gate-messages.json from ink2q.tfchat messages")
    b.add_argument("--requests", type=int, required=True, help="requests a prompt")
    b.add_argument("--max-tokens", type=int, default=256)
    b.add_argument("--temperature", type=float, default=0.0)
    b.add_argument("--seed", type=int)
    b.add_argument("--timeout", type=float, default=600.0)
    b.add_argument("--out", type=Path, required=True)
    b.set_defaults(fn=bench_cmd)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
