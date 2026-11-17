from pathlib import Path

from ink2q.config import SOURCE_REV, Paths


def test_paths_default(monkeypatch):
    for k in ("INK2Q_ROOT", "INK2Q_SOURCE", "INK2Q_DATA"):
        monkeypatch.delenv(k, raising=False)
    p = Paths.from_env()
    assert p.root == Path("$INK2Q_ROOT")
    assert p.source == p.root / "source"
    assert p.data == p.root / "data"


def test_paths_override(monkeypatch, tmp_path):
    monkeypatch.setenv("INK2Q_ROOT", str(tmp_path))
    monkeypatch.setenv("INK2Q_SOURCE", str(tmp_path / "src"))
    monkeypatch.delenv("INK2Q_DATA", raising=False)
    p = Paths.from_env()
    assert p.source == tmp_path / "src"
    assert p.data == tmp_path / "data"


def test_source_rev_pinned():
    assert SOURCE_REV == "f69fa26848416b8f69a0576ef29278b0d3e92d2d"
