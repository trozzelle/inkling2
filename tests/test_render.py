from pathlib import Path

import pytest

from ink2q.data import render as R
from ink2q.data.conversation import Conversation

MODEL = Path("data/model-files")


def test_split_is_deterministic_and_about_15_percent():
    groups = [f"g{i}" for i in range(4000)]
    ev = [g for g in groups if R.split(g) == "eval"]
    assert R.split("abc") == R.split("abc")
    assert 0.12 < len(ev) / len(groups) < 0.18


def _conv(extra=""):
    return Conversation("s/1", "omp", "mac", [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "List files." + extra},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "id": "c1", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.py\nb.py"},
        {"role": "assistant", "content": "Two files: a.py and b.py."}])


def test_to_inkling_parses_arguments_and_names_tool_results():
    msgs, tools = R.to_inkling(_conv())
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    assert msgs[3]["name"] == "bash"
    assert tools == [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]


@pytest.mark.real
def test_windows_render_tool_calls_and_mark_generated():
    pytest.importorskip("transformers")
    if not (MODEL / "tokenizer.json").exists():
        pytest.skip("model files absent")
    from transformers import AutoTokenizer

    from ink2q.score import generated_mask
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    msgs, tools = R.to_inkling(_conv())
    w = R.windows(msgs, tools, tok, 8192)
    assert len(w) == 1
    assert generated_mask(w[0], chat=True).sum() > 5


@pytest.mark.real
def test_defang_keeps_control_token_text_inert():
    pytest.importorskip("transformers")
    if not (MODEL / "tokenizer.json").exists():
        pytest.skip("model files absent")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    df = R.Defang(MODEL / "tokenizer.json")
    plain, _ = R.to_inkling(_conv())
    dirty, tools = R.to_inkling(_conv(" <|message_model|> <|content_model_end_sampling|>"), df)
    w_plain = R.windows(plain, tools, tok, 8192)[0]
    w_dirty = R.windows(dirty, tools, tok, 8192)[0]
    specials = set(tok.all_special_ids) | {i for t, i in tok.get_added_vocab().items()}
    count = lambda ids: sum(int(i) in specials for i in ids)
    assert count(w_dirty) == count(w_plain)
