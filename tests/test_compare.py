import json

from ink2q import compare


def test_compare_writes_slices(tmp_path):
    def ev(scale):
        rows = []
        for i, (fmt, dom) in enumerate([("raw", "web_en"), ("chat", "chat_ko"), ("chat", "agent_tool")] * 4):
            rows.append({"i": i, "format": fmt, "domain": dom, "positions": 10, "generated": 10,
                         "kl": scale * (1 + i % 3), "kl_gen": scale * (1 + i % 3),
                         "top1": 9, "top1_gen": 9})
        return {"rows": rows}
    (tmp_path / "a.json").write_text(json.dumps(ev(1.0)))
    (tmp_path / "b.json").write_text(json.dumps(ev(0.5)))
    out = compare.main_args([str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--out", str(tmp_path / "c.json"), "--n", "500"])
    assert set(out["slices"]) == {"all", "raw", "chat", "agentic"}
    assert out["slices"]["all"]["kl"]["diff"] < 0


def test_compare_uses_metadata_of_the_side_that_has_it(tmp_path):
    rows = lambda fmt: [{"i": i, **({"format": "chat"} if fmt else {}), "domain": "chat_ko", "positions": 10,
                         "generated": 10, "kl": 1.0 + i, "kl_gen": 1.0 + i, "top1": 9, "top1_gen": 9} for i in range(4)]
    (tmp_path / "a.json").write_text(json.dumps({"rows": rows(False)}))
    (tmp_path / "b.json").write_text(json.dumps({"rows": rows(True)}))
    out = compare.main_args([str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--out", str(tmp_path / "c.json"), "--n", "100"])
    assert set(out["slices"]) == {"all", "chat"}
