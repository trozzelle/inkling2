"""Scrubbed agent sessions -> Inkling 2 chat renders: eval/calib split by session group, message conversion and
windowing. Template call as Baekpica's build_calib_inkling2.Renderer.chat (reasoning effort "none")."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from ink2q.data.conversation import Conversation

SYSTEM_MAX_TOKENS = 1024
EVAL_PERCENT = 15
_JOIN = "\u2060"            # word joiner: makes a literal control-token string inert without losing its text


def split(group: str) -> str:
    """'eval' for about 15% of session groups (by sha256), 'calib' otherwise; subagent files share their group."""
    return "eval" if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100 < EVAL_PERCENT else "calib"


class Defang:
    """Inserts a word joiner after '<' in every literal added-token string (all of the form <|...|>)."""

    def __init__(self, tokenizer_json: Path):
        added = json.loads(Path(tokenizer_json).read_text())["added_tokens"]
        specials = sorted({a["content"] for a in added}, key=len, reverse=True)
        self.re = re.compile("|".join(re.escape(s) for s in specials))

    def text(self, s: str) -> str:
        return self.re.sub(lambda m: "<" + _JOIN + m.group(0)[1:], s) if s and "<|" in s else s

    def value(self, v):
        if isinstance(v, str):
            return self.text(v)
        if isinstance(v, list):
            return [self.value(x) for x in v]
        if isinstance(v, dict):
            return {k: self.value(x) for k, x in v.items()}
        return v


def to_inkling(conv: Conversation, defang: Defang | None = None) -> tuple[list[dict], list[dict]]:
    """calq OpenAI-style messages -> Inkling template messages (no reasoning; parsed tool arguments; named tool
    results) and permissive tool declarations for the called functions."""
    d = defang.value if defang else (lambda v: v)
    out, names, by_id = [], [], {}
    for m in conv.messages:
        role = m.get("role")
        if role in ("system", "user"):
            if m.get("content", "").strip() or role == "user":
                out.append({"role": role, "content": d(m.get("content", ""))})
        elif role == "assistant":
            n = {"role": "assistant", "content": d(m.get("content", ""))}
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    args = {"input": args}
                if not isinstance(args, dict):
                    args = {"input": args}
                call = {"type": "function", "function": {"name": fn["name"], "arguments": d(args)}}
                if tc.get("id"):
                    call["id"] = tc["id"]
                    by_id[tc["id"]] = fn["name"]
                calls.append(call)
                if fn["name"] not in names:
                    names.append(fn["name"])
            if calls:
                n["tool_calls"] = calls
            if n["content"] or calls:
                out.append(n)
        elif role == "tool":
            n = {"role": "tool", "content": d(m.get("content", ""))}
            name = by_id.get(m.get("tool_call_id", ""))
            if name:
                n["name"] = name
            out.append(n)
    tools = [{"type": "function", "function": {"name": nm, "parameters": {"type": "object"}}} for nm in names]
    return out, tools


def _render(tok, msgs, tools) -> str:
    return tok.apply_chat_template(copy.deepcopy(msgs), tools=tools or None, tokenize=False, reasoning_effort="none")


def _ids(tok, text) -> np.ndarray:
    return np.array(tok(text, add_special_tokens=False)["input_ids"], np.int64)


def _window_tools(tools, msgs):
    used = {tc["function"]["name"] for m in msgs for tc in m.get("tool_calls", [])}
    return [t for t in tools if t["function"]["name"] in used]


def windows(messages, tools, tok, max_len: int) -> list[np.ndarray]:
    """Windows of one conversation: lead (system if <= SYSTEM_MAX_TOKENS, plus the first user message) followed by
    as many consecutive later messages as fit in max_len tokens; each window starts where the last one ended."""
    first_user = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
    if first_user is None:
        return []
    lead = []
    if messages[0]["role"] == "system" and len(_ids(tok, _render(tok, messages[:1], None))) <= SYSTEM_MAX_TOKENS:
        lead.append(messages[0])
    lead.append(messages[first_user])
    rest = messages[first_user + 1:]
    base = len(_ids(tok, _render(tok, lead, None)))
    if base >= max_len:
        return []
    cost = [len(_ids(tok, _render(tok, lead + [m], _window_tools(tools, [m])))) - base for m in rest]
    out, i = [], 0
    while i < len(rest):
        j, used = i, base
        while j < len(rest) and used + cost[j] <= max_len:
            used += cost[j]
            j += 1
        if j == i:                       # one message larger than a window: skip it
            i += 1
            continue
        while j > i:
            win = lead + rest[i:j]
            ids = _ids(tok, _render(tok, win, _window_tools(tools, win)))
            if len(ids) <= max_len:
                break
            j -= 1
        if j > i and any(m["role"] == "assistant" for m in rest[i:j]):
            out.append(ids)
        i = max(j, i + 1)
    return out
