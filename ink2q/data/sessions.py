"""Agent session archives -> OpenAI-format conversations (the last branch of each session tree), scrubbed."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .conversation import Conversation
from .scrub import scrub, scrub_arguments

_SID = re.compile(r"^\d{4}-\d\d-\d\dT[\d-]+Z_[0-9a-f-]{36}$")


def session_group(path) -> str:
    """omp subagent files (`<sid>/<Task>.jsonl`) and Claude Code subagent files
    (`<session-uuid>/subagents/agent-*.jsonl`) share their parent session's group; any other file is its own."""
    path = Path(path)
    if path.parent.name == "subagents" and path.parent.parent.name:
        return path.parent.parent.name
    return path.parent.name if _SID.match(path.parent.name) else path.stem


def _lines(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _branch(recs: list[dict], key: str, parent: str, keep) -> list[dict]:
    """Records on the path from the root to the last record that `keep` accepts."""
    by_id = {r[key]: r for r in recs if key in r}
    last = next((r for r in reversed(recs) if keep(r)), None)
    path, cur, seen = [], last, set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        path.append(cur)
        cur = by_id.get(cur.get(parent))
    return path[::-1]


def _text(blocks) -> str:
    if isinstance(blocks, str):
        return blocks
    return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def _call(cid: str, name: str, args) -> dict:
    return {"type": "function", "id": cid, "function": {"name": name, "arguments": scrub_arguments(args)}}


def read_omp(path, host: str) -> Conversation | None:
    path = Path(path)
    recs = _lines(path)
    msgs: list[dict] = []
    for r in recs:
        if r.get("type") == "session_init" and r.get("systemPrompt"):
            sp = r["systemPrompt"]
            msgs.append({"role": "system", "content": scrub("\n".join(sp) if isinstance(sp, list) else sp)})
            break
    for r in _branch(recs, "id", "parentId", lambda r: r.get("type") == "message"):
        if r.get("type") != "message":
            continue
        m = r["message"]
        role = m.get("role")
        if role == "user":
            msgs.append({"role": "user", "content": scrub(_text(m.get("content", [])))})
        elif role == "toolResult":
            msgs.append({"role": "tool", "tool_call_id": m.get("toolCallId", ""),
                         "content": scrub(_text(m.get("content", [])))})
        elif role == "assistant":
            out: dict = {"role": "assistant", "content": ""}
            texts, calls, reasoning = [], [], []
            for b in m.get("content", []):
                t = b.get("type")
                if t == "text":
                    texts.append(b.get("text", ""))
                elif t == "thinking" and b.get("thinkingSignature") == "reasoning_content":
                    reasoning.append(b.get("thinking", ""))
                elif t == "toolCall":
                    calls.append(_call(b.get("id", ""), b["name"], b.get("arguments", {})))
            out["content"] = scrub("\n".join(texts))
            if reasoning:
                out["reasoning_content"] = scrub("\n".join(reasoning))
            if calls:
                out["tool_calls"] = calls
            msgs.append(out)
    conv = Conversation(id=f"{path.parent.name}/{path.stem}", source="omp", host=host, messages=msgs,
                        group=session_group(path))
    return conv if conv.has_assistant() else None


def read_claude(path, host: str) -> Conversation | None:
    path = Path(path)
    chain = _branch(_lines(path), "uuid", "parentUuid", lambda r: r.get("type") in ("user", "assistant"))
    branch = [r for r in chain if r.get("type") in ("user", "assistant") and not r.get("isMeta")]
    msgs: list[dict] = []
    cur_id = None
    parts: list[str] = []
    for r in branch:
        m = r["message"]
        if r["type"] == "assistant":
            if not msgs or msgs[-1]["role"] != "assistant" or m.get("id") != cur_id:
                msgs.append({"role": "assistant", "content": ""})
                cur_id, parts = m.get("id"), []
            a = msgs[-1]
            for b in m.get("content", []):
                if b.get("type") == "text":
                    if b.get("text"):
                        parts.append(scrub(b["text"]))
                        a["content"] = "\n".join(parts)
                elif b.get("type") == "tool_use":
                    a.setdefault("tool_calls", []).append(_call(b["id"], b["name"], b.get("input", {})))
            continue
        cur_id = None
        content = m.get("content", "")
        if isinstance(content, list) and any(b.get("type") == "tool_result" for b in content):
            for b in content:
                if b.get("type") == "tool_result":
                    msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id", ""),
                                 "content": scrub(_text(b.get("content", "")))})
        else:
            msgs.append({"role": "user", "content": scrub(_text(content))})
    conv = Conversation(id=f"claude/{path.stem}", source="claude", host=host, messages=msgs,
                        group=session_group(path))
    return conv if conv.has_assistant() else None
