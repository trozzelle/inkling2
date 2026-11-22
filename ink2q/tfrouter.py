"""Prefill rate through the router for the prompt-path gate: one short warm-up request (so the model load is not in a
time to first token), then each long prompt N times as a streamed request with max_tokens 1. A request's prefill rate is
usage.prompt_tokens over the time from sending it to its first streamed text chunk (a bare role chunk sent before the
prefill does not count; a request that streams no text falls back to its finish chunk). Per prompt the report gives the
first request's rate (the cold one: later requests may be served from the server's prompt cache, which usage reports as
cached_tokens) and the median over all of them, and optionally pairs every request with the server's own `done` log line
(prompt=, cached=, ttft=) to cross-check the time."""
import argparse
import json
import re
import statistics
from pathlib import Path

from ink2q import tfgate

WARMUP = {"seq": -1, "messages": [{"role": "user", "content": "Reply with one word."}]}
WARMUP_MAX_TOKENS = 8
DONE = re.compile(r"\[tensorfold\] done \S+ prompt=(\d+) cached=(\d+)")


def request_record(events, t0: float) -> dict:
    """One request's prefill record from its (time, SSE line) pairs and start time."""
    base = tfgate.parse_stream(events, t0)
    first_any = finish_at = None
    cached = None
    for t, line in events:
        line = line.strip()
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            continue
        d = json.loads(line[5:].strip())
        first_any = t if first_any is None else first_any
        if d.get("usage"):
            cached = ((d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"))
        if finish_at is None and any(c.get("finish_reason") for c in d.get("choices") or []):
            finish_at = t
    if base["ttft"] is not None:
        basis, secs = "text", base["ttft"]
    elif finish_at is not None:
        basis, secs = "finish", finish_at - t0
    else:
        raise RuntimeError("stream carried neither a text chunk nor a finish chunk")
    return {"prompt_tokens": base["prompt_tokens"], "completion_tokens": base["completion_tokens"],
            "first_chunk": first_any - t0, "text_chunk": base["ttft"], "finish": None if finish_at is None else finish_at - t0,
            "cached_tokens": cached, "basis": basis, "prefill_s": secs, "prefill_tok_s": base["prompt_tokens"] / secs}


def done_lines(text: str) -> list[dict]:
    """The server log's `done` lines in order: prompt tokens, cached tokens and the server's ttft in seconds (None when
    the line has none)."""
    out = []
    for line in text.splitlines():
        m = DONE.search(line)
        if m is None:
            continue
        ttft = re.search(r"\bttft=([\d.]+)", line)
        out.append({"prompt": int(m.group(1)), "cached": int(m.group(2)), "ttft": float(ttft.group(1)) if ttft else None})
    return out


def pair_with_log(sent: list[dict], log: list[dict]) -> list[dict | None]:
    """For each sent request (in the order sent, with prompt_tokens), the next log line that has its prompt count, or
    None when the log has none left."""
    out, at = [], 0
    for rec in sent:
        hit = next((i for i in range(at, len(log)) if log[i]["prompt"] == rec["prompt_tokens"]), None)
        out.append(None if hit is None else log[hit])
        at = at if hit is None else hit + 1
    return out


def summarize(prompts: list[dict], records: list[dict]) -> dict:
    """prompts: {seq, case, ids_length?, rendered_length?}; records: per-request records with seq, in the order sent."""
    per, mismatches = [], []
    for p in prompts:
        recs = [r for r in records if r["seq"] == p["seq"]]
        served = sorted({r["prompt_tokens"] for r in recs})
        rates = [r["prefill_tok_s"] for r in recs]
        fresh = [r["prefill_tok_s"] for r in recs if not r["cached_tokens"]]
        per.append({"seq": p["seq"], "case": p["case"], "requests": len(recs), "prompt_tokens": served,
                    "first_tok_s": rates[0] if rates else None,
                    "median_tok_s": statistics.median(rates) if rates else None,
                    "uncached_requests": len(fresh), "median_uncached_tok_s": statistics.median(fresh) if fresh else None,
                    "cached_tokens": [r["cached_tokens"] for r in recs]})
        want = p.get("rendered_length")
        if want is not None and served != [want]:
            mismatches.append({"seq": p["seq"], "case": p["case"], "server_prompt_tokens": served,
                               "rendered_length": want, "ids_length": p.get("ids_length")})
    return {"per_prompt": per, "prompt_token_mismatches": mismatches}


def attempt(ns, entry: dict) -> dict:
    """One streamed request as a record, or {"error": text} when it fails (HTTP error, timeout, malformed stream)."""
    try:
        t0, ev = tfgate.stream_events(ns, entry)
        return request_record(ev, t0)
    except Exception as e:  # a failed request is a result, not a reason to lose the finished ones
        return {"error": f"{type(e).__name__}: {e}"}


def report(a, entries: list[dict], warm: dict, records: list[dict]) -> dict:
    """The report so far: failed requests are listed under `errors` and left out of the rates."""
    good = [r for r in records if "error" not in r]
    prompts = [{k: e.get(k) for k in ("seq", "case", "ids_length", "rendered_length")} for e in entries]
    out = {"base_url": a.base_url, "model": a.model, "requests_per_prompt": a.requests, "order": a.order,
           "max_tokens": a.max_tokens, "warmup": warm, **summarize(prompts, good),
           "errors": [{"seq": r["seq"], "request": r["request"], "error": r["error"]} for r in records if "error" in r]
           + ([{"seq": -1, "request": 0, "error": warm["error"]}] if "error" in warm else []),
           "records": records}
    if a.server_log and a.server_log.exists():
        sent = [r for r in [warm] + records if "error" not in r]
        log = done_lines(a.server_log.read_text(errors="replace"))
        for rec, hit in zip(sent, pair_with_log(sent, log)):
            rec["server"] = hit
            if hit and hit["ttft"]:
                rec["server_ttft_ratio"] = rec["prefill_s"] / hit["ttft"]
    return out


def write_report(a, out: dict) -> None:
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    tmp.replace(a.out)


def bench_cmd(a) -> None:
    entries = json.loads(a.messages.read_text())
    ns = argparse.Namespace(base_url=a.base_url, model=a.model, max_tokens=a.max_tokens, temperature=0.0, seed=None,
                            timeout=a.timeout, api_key_file=a.api_key_file)
    warm = attempt(argparse.Namespace(**{**vars(ns), "max_tokens": WARMUP_MAX_TOKENS}), WARMUP)
    print(json.dumps({"warmup": {k: warm[k] for k in ("prompt_tokens", "prefill_s", "error") if k in warm}}), flush=True)
    order = [(e, i) for i in range(a.requests) for e in entries] if a.order == "round-robin" \
        else [(e, i) for e in entries for i in range(a.requests)]
    records = []
    try:
        for entry, i in order:
            rec = {"seq": entry["seq"], "request": i, **attempt(ns, entry)}
            records.append(rec)
            print(json.dumps({k: rec[k] for k in ("seq", "request", "prompt_tokens", "cached_tokens", "basis", "prefill_s",
                                                    "prefill_tok_s", "error") if k in rec}), flush=True)
            write_report(a, report(a, entries, warm, records))  # finished measurements survive whatever comes next
    finally:
        out = report(a, entries, warm, records)
        write_report(a, out)
    print(json.dumps({"per_prompt": [{k: p[k] for k in ("case", "first_tok_s", "median_tok_s", "median_uncached_tok_s")}
                                     for p in out["per_prompt"]],
                      "prompt_token_mismatches": len(out["prompt_token_mismatches"]), "errors": len(out["errors"])}))
    if out["errors"]:
        raise SystemExit(1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="prefill tok/s of long prompts through the router")
    b.add_argument("--base-url", required=True, help="router URL ending in /v1")
    b.add_argument("--model", required=True)
    b.add_argument("--messages", type=Path, required=True, help="prefill-messages.json from ink2q.tfchat documents")
    b.add_argument("--api-key-file", type=Path, required=True, help="a file holding the bearer key")
    b.add_argument("--requests", type=int, default=3, help="requests a prompt")
    b.add_argument("--max-tokens", type=int, default=1)
    b.add_argument("--order", choices=("round-robin", "sequential"), default="round-robin",
                   help="round-robin: each pass sends every prompt once (a single-slot prompt cache never holds the "
                        "prompt being sent); sequential: one prompt's requests together")
    b.add_argument("--timeout", type=float, default=900.0)
    b.add_argument("--server-log", type=Path, help="the engine's log, to pair each request with its `done` line")
    b.add_argument("--out", type=Path, required=True)
    b.set_defaults(fn=bench_cmd)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
