"""Prefill rate through the router for the prompt-path gate: one short warm-up request (so the model load is not in a
time to first token), then each long prompt N times as a streamed request with max_tokens 1. A request's prefill rate is
usage.prompt_tokens over the time from sending it to its first streamed text chunk (a bare role chunk sent before the
prefill does not count; a request that streams no text falls back to its finish chunk). Per prompt the report gives the
first request's rate (the cold one: later requests may be served from the server's prompt cache, which usage reports as
cached_tokens) and the median over all of them, and optionally pairs every request with the server's own `done` log line
(prompt=, cached=, ttft=) to cross-check the time."""
import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

from ink2q import tfgate

WARMUP = {"seq": -1, "messages": [{"role": "user", "content": "Reply with one word."}]}
WARMUP_MAX_TOKENS = 8
DONE = re.compile(r"\[tensorfold\] done \S+ prompt=(\d+) cached=(\d+)")

FIRST_QUESTION = "Summarise the main points in one sentence."
FOLLOW_QUESTIONS = (
    "Summarise the previous answer in one sentence.",
    "What is the main uncertainty in that?",
    "Give one concrete example.",
)
EDIT_QUESTIONS = (
    "List the three most important facts from the document.",
    "State the document's main conclusion in one sentence.",
)
REASK_QUESTIONS = (
    "What are the key assumptions or limitations in this document?",
    "Identify the main open question raised by this document.",
)
DEFAULT_SCRIPT = "follow,regenerate,edit,follow,reask"
VALID_OPS = ("follow", "regenerate", "edit", "reask")


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
        fresh = [r["prefill_tok_s"] for r in recs if r["cached_tokens"] == 0]  # None: the server did not say
        per.append({"seq": p["seq"], "case": p["case"], "requests": len(recs), "prompt_tokens": served,
                    "first_tok_s": rates[0] if recs and recs[0]["request"] == 0 else None,  # None: the cold one failed
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


def log_since(path: Path, offset: int) -> str:
    """The log's text from byte `offset` on (all of it when the log has since been cut shorter than that)."""
    with path.open("rb") as f:
        if f.seek(0, 2) < offset:
            offset = 0
        f.seek(offset)
        return f.read().decode(errors="replace")


def report(a, entries: list[dict], warm: dict, records: list[dict], log_from: int = 0) -> dict:
    """The report so far: failed requests are listed under `errors` and left out of the rates. The server log is read
    from byte `log_from` on, so lines from before this run are never paired."""
    good = [r for r in records if "error" not in r]
    prompts = [{k: e.get(k) for k in ("seq", "case", "ids_length", "rendered_length")} for e in entries]
    out = {"base_url": a.base_url, "model": a.model, "requests_per_prompt": a.requests, "order": a.order,
           "max_tokens": a.max_tokens, "warmup": warm, **summarize(prompts, good),
           "errors": [{"seq": r["seq"], "request": r["request"], "error": r["error"]} for r in records if "error" in r]
           + ([{"seq": -1, "request": 0, "error": warm["error"]}] if "error" in warm else []),
           "records": records}
    if a.server_log and a.server_log.exists():
        sent = [r for r in [warm] + records if "error" not in r]
        log = done_lines(log_since(a.server_log, log_from))
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
    log_from = a.server_log.stat().st_size if a.server_log and a.server_log.exists() else 0
    warm = attempt(argparse.Namespace(**{**vars(ns), "max_tokens": WARMUP_MAX_TOKENS}), WARMUP)
    print(json.dumps({"warmup": {k: warm[k] for k in ("prompt_tokens", "prefill_s", "error") if k in warm}}), flush=True)
    order = [(e, i) for i in range(a.requests) for e in entries] if a.order == "round-robin" \
        else [(e, i) for e in entries for i in range(a.requests)]
    records = []
    if "error" in warm:  # the router is down, refusing the key or hung: no long request would fare better
        order = []
    try:
        for entry, i in order:
            rec = {"seq": entry["seq"], "request": i, **attempt(ns, entry)}
            records.append(rec)
            print(json.dumps({k: rec[k] for k in ("seq", "request", "prompt_tokens", "cached_tokens", "basis", "prefill_s",
                                                    "prefill_tok_s", "error") if k in rec}), flush=True)
            write_report(a, report(a, entries, warm, records, log_from))  # finished measurements survive whatever comes next
    finally:
        out = report(a, entries, warm, records, log_from)
        write_report(a, out)
    print(json.dumps({"per_prompt": [{k: p[k] for k in ("case", "first_tok_s", "median_tok_s", "median_uncached_tok_s")}
                                     for p in out["per_prompt"]],
                      "prompt_token_mismatches": len(out["prompt_token_mismatches"]), "errors": len(out["errors"])}))
    if out["errors"]:
        raise SystemExit(1)


def _iter_chunks(events):
    """Yield (time, parsed dict) from (time, dict) or (time, SSE line) pairs."""
    for t, item in events:
        if isinstance(item, str):
            line = item.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            d = json.loads(data)
        else:
            d = item
        yield t, d


def turn_record(events, t0: float) -> dict:
    """One turn's timing, token counts and reply from (time, dict | SSE line) pairs and start time."""
    first = last = finish_at = None
    usage = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for t, d in _iter_chunks(events):
        if d.get("error"):
            raise RuntimeError(f"server error in stream: {d['error']}")
        if d.get("usage"):
            usage = d["usage"]
        for c in d.get("choices") or []:
            delta = c.get("delta") or {}
            text = delta.get("content") or ""
            reason = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if text:
                content_parts.append(text)
            if reason:
                reasoning_parts.append(reason)
            if text or reason:
                first = t if first is None else first
                last = t
            if finish_at is None and c.get("finish_reason"):
                finish_at = t
    if usage is None:
        raise RuntimeError("stream carried no usage")
    if first is not None:
        ttft_s = first - t0
    elif finish_at is not None:
        ttft_s = finish_at - t0
    else:
        raise RuntimeError("stream carried neither a text chunk nor a finish chunk")
    completion_tokens = usage["completion_tokens"]
    decode_time = None if first is None else last - first
    decode_tok_s = (
        (completion_tokens - 1) / decode_time
        if decode_time is not None and decode_time > 0 and completion_tokens > 1
        else None
    )
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    cached_tokens = 0 if cached is None else cached
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    reply = content if content else reasoning
    rec = {
        "ttft_s": ttft_s,
        "prompt_tokens": usage["prompt_tokens"],
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "decode_tok_s": decode_tok_s,
        "reply": reply,
        "reply_sha256": hashlib.sha256(reply.encode("utf-8")).hexdigest(),
    }
    if content and reasoning:
        rec["reasoning_sha256"] = hashlib.sha256(reasoning.encode("utf-8")).hexdigest()
    return rec


def first_messages(messages: list[dict], question: str) -> list[dict]:
    """Copy messages with question appended to the last user message."""
    out = [dict(m) for m in messages]
    idx = next((i for i in range(len(out) - 1, -1, -1) if out[i].get("role") == "user"), None)
    if idx is None:
        raise ValueError("messages has no user message")
    body = out[idx].get("content") or ""
    out[idx]["content"] = f"{body}\n\n{question}" if body else question
    return out


def next_messages(op: str, messages: list[dict], reply: str, question: str) -> list[dict]:
    """Next request's messages for one conversation operation without mutating `messages`."""
    out = [dict(m) for m in messages]
    if op == "follow":
        return out + [{"role": "assistant", "content": reply}, {"role": "user", "content": question}]
    if op == "regenerate":
        return out
    if op == "edit":
        idx = next((i for i in range(len(out) - 1, -1, -1) if out[i].get("role") == "user"), None)
        if idx is None:
            raise ValueError("messages has no user message")
        out[idx]["content"] = question
        return out
    raise ValueError(f"unknown operation: {op!r}")


def first_difference(a: dict, b: dict) -> tuple[str, int] | None:
    """(label, turn_index) of the first turn whose reply_sha256 differs, or None when every turn matches."""
    ca = a.get("conversations") or []
    cb = b.get("conversations") or []
    for i in range(max(len(ca), len(cb))):
        if i >= len(ca):
            return (cb[i].get("label", f"#{i}"), 0)
        if i >= len(cb):
            return (ca[i].get("label", f"#{i}"), 0)
        la, lb = ca[i].get("label", f"#{i}"), cb[i].get("label", f"#{i}")
        if la != lb:
            return (la, 0)
        ta = ca[i].get("turns") or []
        tb = cb[i].get("turns") or []
        for j in range(max(len(ta), len(tb))):
            if j >= len(ta) or j >= len(tb):
                return (la, j)
            if "error" in ta[j] or "error" in tb[j]:
                return (la, j)
            if ta[j].get("reply_sha256") is None or ta[j].get("reply_sha256") != tb[j].get("reply_sha256"):
                return (la, j)
            if ta[j].get("reasoning_sha256") != tb[j].get("reasoning_sha256"):
                return (la, j)
    return None


def _attempt_turn(ns, entry: dict) -> dict:
    try:
        t0, ev = tfgate.stream_events(ns, entry)
        return turn_record(ev, t0)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def turns_cmd(a) -> None:
    ops = [s.strip() for s in a.script.split(",") if s.strip()]
    bad = [op for op in ops if op not in VALID_OPS]
    if bad:
        raise SystemExit(f"unknown script operation(s): {', '.join(bad)}")
    entries = json.loads(a.messages.read_text())
    ns = argparse.Namespace(base_url=a.base_url, model=a.model, max_tokens=a.max_tokens, temperature=0.0, seed=None,
                            timeout=a.timeout, api_key_file=a.api_key_file)
    conversations: list[dict] = []
    errors: list[dict] = []

    def snapshot() -> dict:
        return {"base_url": a.base_url, "model": a.model, "script": ops, "max_tokens": a.max_tokens,
                "conversations": conversations, "errors": errors}

    try:
        for idx, entry in enumerate(entries):
            label = entry.get("label") or entry.get("case") or f"seq-{entry.get('seq', idx)}"
            base_messages = entry["messages"]
            conv = {"label": label, "turns": []}
            conversations.append(conv)
            cur = first_messages(base_messages, FIRST_QUESTION)
            follow_i = edit_i = reask_i = 0
            last_reply = ""
            for step_i, op in enumerate(["first"] + ops):
                if op == "follow":
                    q = FOLLOW_QUESTIONS[follow_i % len(FOLLOW_QUESTIONS)]
                    follow_i += 1
                    cur = next_messages("follow", cur, last_reply, q)
                elif op == "regenerate":
                    cur = next_messages("regenerate", cur, last_reply, "")
                elif op == "edit":
                    q = EDIT_QUESTIONS[edit_i % len(EDIT_QUESTIONS)]
                    edit_i += 1
                    cur = next_messages("edit", cur, last_reply, q)
                elif op == "reask":
                    q = REASK_QUESTIONS[reask_i % len(REASK_QUESTIONS)]
                    reask_i += 1
                    cur = first_messages(base_messages, q)
                rec = {"op": op, **_attempt_turn(ns, {**entry, "messages": cur})}
                conv["turns"].append(rec)
                print(json.dumps({"label": label, "turn": step_i, **{k: rec[k] for k in (
                    "op", "ttft_s", "prompt_tokens", "cached_tokens", "completion_tokens", "decode_tok_s", "error")
                    if k in rec}}), flush=True)
                if "error" in rec:
                    errors.append({"label": label, "turn": step_i, "op": op, "error": rec["error"]})
                    write_report(a, snapshot())
                    break
                last_reply = rec["reply"]
                write_report(a, snapshot())
    finally:
        out = snapshot()
        write_report(a, out)
    print(json.dumps({"conversations": len(conversations), "errors": len(errors), "out": str(a.out)}))
    if errors:
        raise SystemExit(1)


def same_cmd(a) -> None:
    da = json.loads(a.a.read_text())
    db = json.loads(a.b.read_text())
    diff = first_difference(da, db)
    if diff is None:
        print(json.dumps({"equal": True, "conversations": len(da.get("conversations") or [])}))
        return
    label, turn = diff
    print(json.dumps({"equal": False, "label": label, "turn": turn}))
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

    t = sub.add_parser("turns", help="multi-turn conversation script for the prompt cache gate")
    t.add_argument("--base-url", required=True, help="server or router URL ending in /v1")
    t.add_argument("--model", required=True)
    t.add_argument("--messages", type=Path, required=True, help="prefill-messages.json from ink2q.tfchat documents")
    t.add_argument("--api-key-file", type=Path, help="a file holding the bearer key, when behind a router")
    t.add_argument("--script", default=DEFAULT_SCRIPT,
                   help="comma-separated operations after turn 1 (follow, regenerate, edit, reask)")
    t.add_argument("--max-tokens", type=int, default=128)
    t.add_argument("--timeout", type=float, default=900.0)
    t.add_argument("--out", type=Path, required=True)
    t.set_defaults(fn=turns_cmd)

    s = sub.add_parser("same", help="exit 0 when every turn's reply_sha256 matches between two turns outputs")
    s.add_argument("--a", type=Path, required=True)
    s.add_argument("--b", type=Path, required=True)
    s.set_defaults(fn=same_cmd)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
