import json

import pytest

from ink2q import tfchat, tfgate

# ---- tfgate: SSE record parsing and the pooling maths


def sse(obj) -> str:
    return "data: " + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n"


def chunk(**delta) -> dict:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


def stream(times_and_deltas, completion_tokens, prompt_tokens=10):
    ev = [(t, sse(chunk(**d))) for t, d in times_and_deltas]
    t_end = times_and_deltas[-1][0]
    ev.append((t_end, sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})))
    ev.append((t_end, sse({"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                                                   "total_tokens": prompt_tokens + completion_tokens}})))
    ev.append((t_end, sse("[DONE]")))
    return ev


def test_parse_stream_times_first_to_last_content_or_reasoning_chunk():
    ev = [(0.1, sse({"choices": [{"delta": {"role": "assistant"}}]})),  # no text: not the first chunk
          (0.5, sse(chunk(reasoning_content="a"))), (1.0, sse(chunk(content="b"))), (2.5, sse(chunk(content="c")))]
    ev += stream([(2.5, {"content": ""})], 4, 7)[1:]  # empty delta, finish, usage, done
    rec = tfgate.parse_stream(ev, t0=0.0)
    assert rec == {"ttft": 0.5, "decode_time": 2.0, "completion_tokens": 4, "prompt_tokens": 7, "finish_reason": "length"}


def test_parse_stream_requires_usage_and_surfaces_errors():
    with pytest.raises(RuntimeError, match="no usage"):
        tfgate.parse_stream([(1.0, sse(chunk(content="a")))], 0.0)
    with pytest.raises(RuntimeError, match="boom"):
        tfgate.parse_stream([(1.0, sse({"error": "boom"}))], 0.0)


def rec(seq, ct, dt, pt=10):
    return {"seq": seq, "ttft": 0.1, "decode_time": dt, "completion_tokens": ct, "prompt_tokens": pt, "finish_reason": "length"}


def test_pooled_rate_is_total_tokens_over_total_time_not_a_mean_of_rates():
    # 100 decoded tokens in 1s (100/s) and 10 decoded tokens in 10s (1/s): pooled 110/11 = 10, mean of rates 50.5
    recs = [rec(0, 101, 1.0), rec(1, 11, 10.0)]
    assert tfgate.rate(recs) == pytest.approx(10.0)


def test_summarize_pools_medians_and_reports_the_bench_prompt():
    prompts = [{"seq": 0, "set": "bench", "case": "b"}, {"seq": 1, "set": "chat", "case": "x"},
               {"seq": 2, "set": "chat", "case": "y"}]
    recs = [rec(0, 101, 1.0), rec(0, 101, 1.0), rec(1, 51, 1.0), rec(2, 11, 10.0)]
    s = tfgate.summarize(prompts, recs)
    assert s["pooled_rate"] == pytest.approx((100 + 100 + 50 + 10) / (1 + 1 + 1 + 10))
    assert s["bench_rate"] == pytest.approx(100.0)
    assert s["median_prompt_rate"] == pytest.approx(50.0)  # per-prompt rates 100, 50, 1
    assert [p["rate"] for p in s["per_prompt"]] == pytest.approx([100.0, 50.0, 1.0])
    assert s["requests"] == 4 and s["unusable_requests"] == 0


def test_unusable_requests_are_counted_and_left_out_of_the_pool():
    prompts = [{"seq": 0, "set": "bench", "case": "b"}]
    recs = [rec(0, 101, 1.0), rec(0, 1, 0.0), {**rec(0, 5, 1.0), "decode_time": None, "ttft": None}]
    s = tfgate.summarize(prompts, recs)
    assert s["pooled_rate"] == pytest.approx(100.0)
    assert s["unusable_requests"] == 2


def test_prompt_token_check_reports_mismatches_and_flags_the_unexplained():
    prompts = [{"seq": 0, "set": "bench", "case": "b", "ids_length": 116, "rendered_length": 116},
               {"seq": 1, "set": "chat", "case": "x", "ids_length": 50, "rendered_length": 49},
               {"seq": 2, "set": "chat", "case": "y", "ids_length": 60, "rendered_length": 59}]
    recs = [rec(0, 9, 1.0, pt=116), rec(1, 9, 1.0, pt=49), rec(2, 9, 1.0, pt=70)]
    chk = tfgate.summarize(prompts, recs)["prompt_tokens_check"]
    assert [m["seq"] for m in chk["mismatches"]] == [1, 2]
    assert [m["seq"] for m in chk["unexplained"]] == [2]
    assert chk["mismatches"][0] == {"seq": 1, "case": "x", "server_prompt_tokens": [49], "ids_length": 50,
                                    "rendered_length": 49}


def test_request_body_streams_with_usage_and_carries_tools_and_effort():
    a = tfgate.argparse.Namespace(model="m", max_tokens=256, temperature=0.0, seed=None)
    entry = {"seq": 1, "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}],
             "reasoning_effort": "none"}
    body = tfgate.request_body(a, entry)
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["temperature"] == 0.0 and body["tools"] == [{"type": "function"}] and body["reasoning_effort"] == "none"
    assert "seed" not in body
    a.seed = 3
    assert tfgate.request_body(a, {"messages": []})["seed"] == 3


# ---- tfchat: the parser on a synthetic rendered stream (one token per character, specials above 1000)

SP = tfchat.Specials(user=1000, model=1001, system=1002, tool=1003, text=1004, image=1005, thinking=1006, xml=1007,
                     invoke_json=1008, end_message=1009, end_sampling=1010)


def enc(s: str) -> list[int]:
    return [ord(c) for c in s]


def dec(toks) -> str:
    return "".join(chr(t) for t in toks)


def seg(head, kind, body, name=""):
    return [head] + enc(name) + [kind] + enc(body) + [SP.end_message]


def effort(text="0.9"):
    return seg(SP.system, SP.text, f"Thinking effort level: {text}")


def test_parse_user_prompt_with_default_effort_and_opener():
    ids = effort() + seg(SP.user, SP.text, "héllo\nthere") + [SP.model]
    p = tfchat.parse_messages(ids, SP, dec)
    assert p.messages == [{"role": "user", "content": "héllo\nthere"}]
    assert p.tools is None and p.reasoning_effort is None and p.opener is True


def test_parse_system_messages_precede_the_effort_marker_and_nondefault_effort_is_named():
    ids = seg(SP.system, SP.text, "be brief") + effort("0") + seg(SP.user, SP.text, "hi")
    p = tfchat.parse_messages(ids, SP, dec)
    assert p.messages == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    assert p.reasoning_effort == "none" and p.opener is False
    assert tfchat.parse_messages(effort("0.7") + seg(SP.user, SP.text, "x"), SP, dec).reasoning_effort == "medium"
    assert tfchat.parse_messages(effort("0.5") + seg(SP.user, SP.text, "x"), SP, dec).reasoning_effort == 0.5


def test_parse_tool_declaration_into_openai_tools():
    spec = [{"description": "d", "name": "f", "parameters": {"type": "object"}, "type": "function"}]
    ids = seg(SP.system, SP.xml, json.dumps(spec), name="tool_declare") + effort() \
        + seg(SP.user, SP.text, "hi") + [SP.model]
    p = tfchat.parse_messages(ids, SP, dec)
    assert p.tools == [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]
    assert p.messages == [{"role": "user", "content": "hi"}]


def test_parse_assistant_turn_with_thinking_and_end_sampling():
    ids = effort() + seg(SP.user, SP.text, "q") + seg(SP.model, SP.thinking, "hmm") + seg(SP.model, SP.text, "a") \
        + [SP.end_sampling] + seg(SP.user, SP.text, "q2") + [SP.model]
    p = tfchat.parse_messages(ids, SP, dec)
    assert p.messages == [{"role": "user", "content": "q"},
                          {"role": "assistant", "content": "a", "reasoning_content": "hmm"},
                          {"role": "user", "content": "q2"}]


def test_parse_rejects_streams_the_template_cannot_have_rendered():
    with pytest.raises(ValueError, match="effort"):
        tfchat.parse_messages(seg(SP.user, SP.text, "hi"), SP, dec)
    with pytest.raises(ValueError, match="not closed"):
        tfchat.parse_messages(effort() + seg(SP.user, SP.text, "q") + seg(SP.model, SP.text, "a"), SP, dec)
    with pytest.raises(ValueError, match="unsupported"):
        tfchat.parse_messages(effort() + seg(SP.user, SP.image, ""), SP, dec)
    with pytest.raises(ValueError, match="not closed by end_message"):
        tfchat.parse_messages(effort() + [SP.user, SP.text] + enc("x"), SP, dec)


def test_compare_ids_reports_exact_or_the_first_difference():
    assert tfchat.compare_ids([1, 2, 3], [1, 2, 3]) == {"exact": True}
    assert tfchat.compare_ids([1, 2, 3, 4], [1, 9, 3, 8, 7]) == {
        "exact": False, "first_diff": 1, "differing": 3, "want_length": 4, "got_length": 5}
    assert tfchat.compare_ids([1, 2], [1, 2, 3])["first_diff"] == 2
