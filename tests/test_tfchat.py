import json

import pytest

from ink2q import tfchat

SP = tfchat.Specials(user=1000, model=1001, system=1002, tool=1003, text=1004, image=1005, thinking=1006, xml=1007,
                     invoke_json=1008, end_message=1009, end_sampling=1010)
BOS = 2000


def enc(s: str) -> list[int]:
    return [ord(c) for c in s]


def dec(toks) -> str:
    return "".join(chr(t) for t in toks)


def seg(head, kind, body, name=""):
    return [head] + enc(name) + [kind] + enc(body) + [SP.end_message]


def effort(text="0.0"):
    return seg(SP.system, SP.text, f"Thinking effort level: {text}")


def render_one_user(p) -> list[int]:
    eff = "0.0" if p.reasoning_effort == "none" else "0.9"
    return effort(eff) + seg(SP.user, SP.text, p.messages[-1]["content"])


def test_insert_needle_places_it_at_the_fraction():
    assert tfchat.insert_needle(list(range(10)), [100, 101], 0.5) == [0, 1, 2, 3, 4, 100, 101, 5, 6, 7, 8, 9]
    assert tfchat.insert_needle(list(range(4)), [9], 0.0) == [9, 0, 1, 2, 3]
    assert tfchat.insert_needle(list(range(4)), [9], 1.0) == [0, 1, 2, 3, 9]
    with pytest.raises(ValueError, match="frac"):
        tfchat.insert_needle([1, 2], [9], 1.5)


def test_fit_needle_cuts_document_to_exact_target_with_needle_and_question():
    src = [BOS] + enc("abcdefghij" * 40)
    needle = enc("\nThe access code is 7391-KITE.\n")
    target = 220
    k, p, got = tfchat.fit_needle(src, needle, 0.5, dec, {BOS}, render_one_user, target)
    assert len(got) + 1 == target
    text = p.messages[0]["content"]
    assert "The access code is 7391-KITE." in text
    assert text.endswith("\n\n" + tfchat.NEEDLE_QUESTION)
    assert p.reasoning_effort == "none"
    assert k > 0


def test_fit_needle_skips_cuts_that_split_a_character():
    class Frag:
        def __call__(self, toks):
            return "\ufffd" if len(toks) % 2 else "a" * len(toks)

    src = list(range(300))
    needle = [1, 2]
    base = len(render_one_user(tfchat.Parsed([{"role": "user", "content": f"aa\n\n{tfchat.NEEDLE_QUESTION}"}], None, "none", True)))
    k, p, got = tfchat.fit_needle(src, needle, 0.5, Frag(), set(), render_one_user, base + 21)
    assert "\ufffd" not in p.messages[0]["content"]
    assert len(got) + 1 <= base + 21 and k % 2 == 0


def test_first_token_list_and_log_contains():
    log = '{"prompt_ms": 12.3}\nnot json\n[true, false]\n[65, 66, 67]\n[68, 69]\n'
    assert tfchat.first_token_list(log) == [65, 66, 67]
    found, decoded, n = tfchat.log_contains(log, dec, "BC")
    assert found is True and decoded == "ABC" and n == 3
    found2, _, _ = tfchat.log_contains(log, dec, "DE")
    assert found2 is False
    with pytest.raises(ValueError, match="no JSON token list"):
        tfchat.first_token_list('{"only": "object"}\n')


class FakeTok:
    def get_added_vocab(self):
        return {
            "<|message_user|>": SP.user, "<|message_model|>": SP.model, "<|message_system|>": SP.system,
            "<|message_tool|>": SP.tool, "<|content_text|>": SP.text, "<|content_image|>": SP.image,
            "<|content_thinking|>": SP.thinking, "<|content_xml|>": SP.xml,
            "<|content_invoke_tool_json|>": SP.invoke_json, "<|end_message|>": SP.end_message,
            "<|content_model_end_sampling|>": SP.end_sampling, "<|bos|>": BOS,
        }

    def decode(self, toks, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return dec(toks)

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": text}
        tokens, i = [], 0
        vocab = self.get_added_vocab()
        while i < len(text):
            matched = False
            for k, v in vocab.items():
                if text.startswith(k, i):
                    tokens.append(v)
                    i += len(k)
                    matched = True
                    break
            if not matched:
                tokens.append(ord(text[i]))
                i += 1
        return {"input_ids": tokens}

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False, reasoning_effort=None):
        eff = "0.0" if reasoning_effort == "none" else "0.9"
        parts = [f"<|message_system|><|content_text|>Thinking effort level: {eff}<|end_message|>"]
        for m in messages:
            parts.append(f"<|message_user|><|content_text|>{m['content']}<|end_message|>")
        return "".join(parts)


def test_needle_and_contains_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tfchat, "_load_tokenizer", lambda _src: FakeTok())
    ids_path = tmp_path / "long.json"
    ids_path.write_text(json.dumps([BOS] + enc("0123456789" * 40)))
    out_path = tmp_path / "needle.json"

    tfchat.main(["needle", "--source", str(tmp_path), "--ids", str(ids_path), "--target", "240",
                 "--code", "7391-KITE", "--at", "0.5", "--out", str(out_path)])
    got = json.loads(out_path.read_text())
    assert len(got) == 240 and got[-1] == SP.model
    parsed = tfchat.parse_messages(got, SP, dec)
    assert "The access code is 7391-KITE." in parsed.messages[0]["content"]
    assert parsed.messages[0]["content"].endswith("\n\n" + tfchat.NEEDLE_QUESTION)

    log_path = tmp_path / "greedy.log"
    log_path.write_text('{"mode": "greedy"}\n' + json.dumps(enc("Code: 7391-KITE")) + "\n")
    tfchat.main(["contains", "--source", str(tmp_path), "--log", str(log_path), "--text", "7391-KITE"])
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["found"] is True

    with pytest.raises(SystemExit) as e:
        tfchat.main(["contains", "--source", str(tmp_path), "--log", str(log_path), "--text", "WRONG-CODE"])
    assert e.value.code == 1
