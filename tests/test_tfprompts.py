from ink2q import tfprompts
from ink2q.score import MSG_MODEL


def test_chat_prefix_stops_at_the_first_assistant_opener():
    assert tfprompts.chat_prefix([1, 2, MSG_MODEL, 3, MSG_MODEL, 4]) == [1, 2, MSG_MODEL]
    assert tfprompts.chat_prefix([1, 2, 3]) is None


def test_agentic_prefix_takes_the_last_opener_below_the_cap():
    ids = [7] * 10 + [MSG_MODEL] + [8] * 10 + [MSG_MODEL] + [9] * 10
    assert tfprompts.agentic_prefix(ids, 30) == ids[:22]
    assert tfprompts.agentic_prefix(ids, 15) == ids[:11]
    assert tfprompts.agentic_prefix(ids, 5) is None
