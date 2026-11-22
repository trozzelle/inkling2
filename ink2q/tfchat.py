"""Inverse of the Inkling chat template for the drafted-decode gate: turn a rendered token stream back into an OpenAI
messages list (plus tools and reasoning effort), then check that rendering the messages through the source chat
template reproduces the ids. The template emits no generation prompt, so a stream cut at its model-message opener
parses to the messages before it and the trailing opener is reported separately (`opener`)."""
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# reasoning_effort names in chat_template.jinja's effort_map, by the number the template renders
EFFORT_NAME = {0.0: "none", 0.1: "minimal", 0.2: "low", 0.7: "medium", 0.9: "high", 0.99: "xhigh"}
DEFAULT_EFFORT = 0.9
_EFFORT_TEXT = re.compile(r"Thinking effort level: (\S+)")
TOOL_DECLARE = "tool_declare"


@dataclass(frozen=True)
class Specials:
    user: int
    model: int
    system: int
    tool: int
    text: int
    image: int
    thinking: int
    xml: int
    invoke_json: int
    end_message: int
    end_sampling: int

    @classmethod
    def from_vocab(cls, vocab: dict[str, int]) -> "Specials":
        return cls(**{f: vocab[f"<|{n}|>"] for f, n in (
            ("user", "message_user"), ("model", "message_model"), ("system", "message_system"),
            ("tool", "message_tool"), ("text", "content_text"), ("image", "content_image"),
            ("thinking", "content_thinking"), ("xml", "content_xml"), ("invoke_json", "content_invoke_tool_json"),
            ("end_message", "end_message"), ("end_sampling", "content_model_end_sampling"))})


@dataclass
class Parsed:
    messages: list[dict]
    tools: list[dict] | None
    reasoning_effort: str | float | None  # None when the template's default effort applies
    opener: bool  # the stream ends with a bare model-message opener the template does not render


def _segments(ids: list[int], sp: Specials, decode: Callable[[list[int]], str]):
    """Split a stream into (role, name, content_kind, text) segments, end_sampling markers as ("end_sampling",)
    and a final bare header as ("opener", role)."""
    roles = {sp.user: "user", sp.model: "assistant", sp.system: "system", sp.tool: "tool"}
    kinds = {sp.text: "text", sp.thinking: "thinking", sp.xml: "xml", sp.image: "image", sp.invoke_json: "invoke"}
    out, i, n = [], 0, len(ids)
    while i < n:
        t = ids[i]
        if t == sp.end_sampling:
            out.append(("end_sampling",))
            i += 1
            continue
        if t not in roles:
            raise ValueError(f"token {t} at {i} is not a message header")
        role, j = roles[t], i + 1
        while j < n and ids[j] not in kinds and ids[j] != sp.end_message and ids[j] not in roles:
            j += 1
        if j == n and j == i + 1:
            out.append(("opener", role))
            break
        if j >= n or ids[j] not in kinds:
            raise ValueError(f"message at {i} has no content marker")
        name, kind = decode(ids[i + 1:j]) if j > i + 1 else "", kinds[ids[j]]
        k = j + 1
        while k < n and ids[k] != sp.end_message:
            if ids[k] in roles or ids[k] == sp.end_sampling:
                raise ValueError(f"message at {i} is not closed by end_message")
            k += 1
        if k >= n:
            raise ValueError(f"message at {i} is not closed by end_message")
        out.append((role, name, kind, decode(ids[j + 1:k])))
        i = k + 1
    return out


def _effort_name(value: float):
    for num, name in EFFORT_NAME.items():
        if abs(num - value) < 1e-9:
            return name
    return value


def parse_messages(ids, sp: Specials, decode: Callable[[list[int]], str]) -> Parsed:
    segs = _segments([int(t) for t in ids], sp, decode)
    opener = bool(segs) and segs[-1][0] == "opener"
    if opener:
        if segs[-1][1] != "assistant":
            raise ValueError(f"stream ends with a bare {segs[-1][1]} header")
        segs = segs[:-1]
    tools = None
    if segs and segs[0][0] == "system" and segs[0][1] == TOOL_DECLARE and segs[0][2] == "xml":
        specs = json.loads(segs[0][3])
        tools = [{"type": s.get("type") or "function",
                  "function": {"name": s["name"], "description": s.get("description") or "",
                               "parameters": s.get("parameters") or {}}} for s in specs]
        segs = segs[1:]
    # the template renders the effort marker as a system message just before the first non-system message
    first = next((j for j, s in enumerate(segs) if s[0] != "system"), len(segs))
    m = _EFFORT_TEXT.fullmatch(segs[first - 1][3]) if first and segs[first - 1][2] == "text" else None
    if m is None:
        raise ValueError("no thinking-effort marker before the first non-system message")
    effort = float(m.group(1))
    segs = segs[:first - 1] + segs[first:]
    messages, cur = [], None
    for s in segs:
        if s[0] == "end_sampling":
            if cur is None:
                raise ValueError("end_sampling outside an assistant message")
            messages.append(cur)
            cur = None
            continue
        role, name, kind, text = s
        if name or kind not in ("text", "thinking"):
            raise ValueError(f"unsupported {role} segment (name {name!r}, content {kind})")
        if cur is not None and role != "assistant":
            raise ValueError("assistant message not closed by end_sampling")
        if role == "assistant":
            cur = cur or {"role": "assistant", "content": ""}
            if kind == "thinking":
                cur["reasoning_content"] = text
            else:
                cur["content"] = text
        elif kind != "text":
            raise ValueError(f"{role} message with {kind} content")
        else:
            messages.append({"role": role, "content": text})
    if cur is not None:
        raise ValueError("assistant message not closed by end_sampling")
    return Parsed(messages, tools, None if abs(effort - DEFAULT_EFFORT) < 1e-9 else _effort_name(effort), opener)


def render_ids(tok, messages, tools=None, reasoning_effort=None) -> list[int]:
    """The ids the source chat template gives for a messages list, tokenized as ink2q.tfprompts.bench_ids does."""
    kw = {} if reasoning_effort is None else {"reasoning_effort": reasoning_effort}
    text = tok.apply_chat_template(messages, tools=tools or None, add_generation_prompt=True, tokenize=False, **kw)
    return [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]


def compare_ids(want: list[int], got: list[int]) -> dict:
    """Exact, or the first differing position and how many positions differ (a length difference counts)."""
    if want == got:
        return {"exact": True}
    first = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
    differing = sum(a != b for a, b in zip(want, got)) + abs(len(want) - len(got))
    return {"exact": False, "first_diff": first, "differing": differing, "want_length": len(want),
            "got_length": len(got)}


def document_messages(src, sp: Specials, decode: Callable[[list[int]], str], strip: set[int], k: int) -> Parsed | None:
    """Messages for the first k tokens of a document stream. A rendered chat stream (it opens with a system or user
    header) is cut inside its last message, which must be a user message: its text is the tokens so far, and its
    system messages and effort marker are kept. Any other stream is raw text: the `strip` tokens (begin-of-text
    separators and other specials, which would render as literal markup) are dropped and the rest is one user
    message under the template's default effort. None when the cut falls inside a character."""
    head = [int(t) for t in src[:k]]
    if head and head[0] in (sp.system, sp.user):
        p = parse_messages(head + [sp.end_message, sp.model], sp, decode)
        if not p.messages or p.messages[-1]["role"] != "user":
            raise ValueError("the cut message is not a user message")
    else:
        p = Parsed([{"role": "user", "content": decode([t for t in head if t not in strip])}], None, None, True)
    return None if "\ufffd" in p.messages[-1]["content"] else p


def fit_document(src, sp: Specials, decode, strip: set[int], render: Callable[[Parsed], list[int]], target: int):
    """(k, parsed, rendered ids) for the cut of src whose rendered ids plus the opener are exactly `target` long.
    Retokenizing changes the length by a few tokens, so the cut walks by the miss until it hits; when it cycles
    without hitting, the longest fit below the target (the caller sees the length)."""
    k, seen, best = min(target, len(src)), set(), None
    while k >= 1 and k not in seen:
        seen.add(k)
        p = document_messages(src, sp, decode, strip, k)
        if p is None:
            k -= 1
            continue
        got = render(p)
        n = len(got) + 1
        if n == target:
            return k, p, got
        if n < target and (best is None or n > best[0]):
            best = (n, k, p, got)
        k = min(k + target - n, len(src))
    if best is None:
        raise ValueError(f"no cut of the document renders to {target} ids or fewer")
    return best[1:]


def documents_cmd(a) -> None:
    from transformers import AutoTokenizer
    from ink2q.tokens import load_manifest, load_tokens
    tok = AutoTokenizer.from_pretrained(str(a.source))
    sp = Specials.from_vocab(tok.get_added_vocab())
    strip = set(tok.get_added_vocab().values())

    def decode(toks):
        return tok.decode(toks, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    seqs = man = None
    ids_dir = a.out_dir / "prefill-ids"
    ids_dir.mkdir(parents=True, exist_ok=True)
    out, report, paths = [], [], []
    for seq, spec in enumerate(a.doc):
        label, _, src_spec = spec.partition("=")
        kind, _, ref = src_spec.partition(":")
        if kind == "case":
            if seqs is None:
                seqs, man = load_tokens(a.tokens), load_manifest(a.manifest)
            src = [int(t) for t in seqs[next(i for i, m in enumerate(man) if m["case"] == ref)]]
        elif kind == "ids":
            src = [int(t) for t in json.loads(Path(ref).read_text())]
        else:
            raise SystemExit(f"--doc {spec!r}: expected LABEL=case:NAME or LABEL=ids:PATH")
        k, p, got = fit_document(src, sp, decode, strip, lambda q: render_ids(tok, q.messages, q.tools, q.reasoning_effort),
                                 a.target)
        ids = got + [sp.model]  # the opener the template never renders, as the drafted gate's prompts end
        back = parse_messages(ids, sp, decode)  # the ids must parse back to the messages that rendered them
        again = render_ids(tok, back.messages, back.tools, back.reasoning_effort)
        check = compare_ids(got, again)
        check["messages_roundtrip"] = back.messages == p.messages and back.tools == p.tools \
            and back.reasoning_effort == p.reasoning_effort and back.opener
        row = {"seq": seq, "label": label, "source": src_spec, "source_tokens": len(src), "cut_tokens": k,
               "ids_length": len(ids), "rendered_length": len(got), "opener_not_rendered": True,
               "reasoning_effort": p.reasoning_effort, **check}
        report.append(row)
        entry = {"seq": seq, "set": "prefill8k", "case": label, "messages": p.messages}
        if p.tools:
            entry["tools"] = p.tools
        if p.reasoning_effort is not None:
            entry["reasoning_effort"] = p.reasoning_effort
        entry["ids_length"], entry["rendered_length"] = len(ids), len(got)
        out.append(entry)
        path = (ids_dir / f"seq-{seq:05d}.json").resolve()
        path.write_text(json.dumps(ids))
        paths.append(str(path))
    (ids_dir / "index.txt").write_text("\n".join(paths) + "\n")
    (a.out_dir / "prefill-messages.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    (a.out_dir / "prefill-report.json").write_text(json.dumps(report, indent=1) + "\n")
    for r in report:
        print(json.dumps(r), flush=True)
    good = [r for r in report if r["exact"] and r["messages_roundtrip"] and r["ids_length"] == a.target]
    print(json.dumps({"prompts": len(report), "exact_at_target": len(good), "target": a.target, "out": str(a.out_dir)}))
    if len(good) != len(report):
        raise SystemExit(1)


def messages_cmd(a) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(a.source))
    sp = Specials.from_vocab(tok.get_added_vocab())

    def decode(toks):
        return tok.decode(toks, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    meta = json.loads((a.prompts / "prompts.json").read_text())
    out, report = [], []
    for m in meta:
        if m["set"] not in ("bench", "chat"):  # agentic ids are session-derived and never read here
            continue
        ids = json.loads((a.prompts / "ids" / f"seq-{m['seq']:05d}.json").read_text())
        row = {"seq": m["seq"], "set": m["set"], "case": m["case"], "ids_length": len(ids)}
        try:
            p = parse_messages(ids, sp, decode)
        except ValueError as e:
            report.append({**row, "exact": False, "error": str(e)})
            continue
        want = ids[:-1] if p.opener else ids  # the opener is the generation prompt the template never renders
        got = render_ids(tok, p.messages, p.tools, p.reasoning_effort)
        report.append({**row, "opener_not_rendered": p.opener, **compare_ids(want, got)})
        entry = {"seq": m["seq"], "set": m["set"], "case": m["case"], "messages": p.messages}
        if p.tools:
            entry["tools"] = p.tools
        if p.reasoning_effort is not None:
            entry["reasoning_effort"] = p.reasoning_effort
        entry["ids_length"] = len(ids)
        entry["rendered_length"] = len(got)
        out.append(entry)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    for r in report:
        print(json.dumps(r), flush=True)
    print(json.dumps({"prompts": len(report), "exact": sum(1 for r in report if r["exact"]), "out": str(a.out)}))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("messages", help="ids of the bench and chat prompts -> verified OpenAI messages")
    s.add_argument("--prompts", type=Path, required=True, help="tf-inkling prompts dir: prompts.json and ids/")
    s.add_argument("--source", type=Path, required=True, help="source model dir with the tokenizer and chat template")
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(fn=messages_cmd)
    d = sub.add_parser("documents", help="long documents (held-out sequences, raw id lists) -> chat messages whose ids "
                                         "plus the opener are exactly --target long")
    d.add_argument("--source", type=Path, required=True, help="source model dir with the tokenizer and chat template")
    d.add_argument("--doc", action="append", required=True, metavar="LABEL=case:NAME|ids:PATH",
                   help="a held-out case (needs --tokens, --manifest) or a JSON id list; repeat, in seq order")
    d.add_argument("--tokens", type=Path, help="held-out safetensors, for case: documents")
    d.add_argument("--manifest", type=Path, help="its manifest, for case: documents")
    d.add_argument("--target", type=int, default=8192, help="ids length with the opener")
    d.add_argument("--out-dir", type=Path, required=True,
                   help="gets prefill-messages.json, prefill-report.json and prefill-ids/ (seq-NNNNN.json, index.txt)")
    d.set_defaults(fn=documents_cmd)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
