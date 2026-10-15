"""Jev integration with a fake client (no network, no key needed)."""
from types import SimpleNamespace

import pytest

from model_finder import config, jev, query


def choice(c, probs, conf=0.9):
    return SimpleNamespace(type="choice", choice=c, probabilities=probs, confidence=conf)


def noul(v):
    return SimpleNamespace(type="noul", noul=v)


class FakeClient:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        if self.fail:
            raise RuntimeError("boom")
        if "task" in questions:  # parse call
            assert set(questions["task"].criteria) >= {"text-to-speech", "unclear"}
            return SimpleNamespace(answers={
                "task": choice("text-to-speech", {"text-to-speech": 0.7, "audio-to-audio": 0.25, "unclear": 0.05}),
                "device": choice("phone", {"phone": 0.95}),
                "language": choice("unspecified", {"unspecified": 0.9}),
                "license_open": noul(0.1), "newest": noul(0.2), "code": noul(0.01), "voice_clone": noul(0.8),
            })
        # fit call: candidate 1 is the better fit
        return SimpleNamespace(answers={k: noul(0.9 if k == "c1" else 0.1) for k in questions})


@pytest.fixture
def fake(monkeypatch):
    fc = FakeClient()
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(config, "PARSER", "auto")
    monkeypatch.setattr(jev, "client", lambda: fc)
    return fc


def test_parse_maps_answers(fake):
    out = jev.parse("clone my voice on my iPhone")
    assert out["tasks"] == ["text-to-speech", "audio-to-audio"]
    assert out["device"] == "phone" and out["languages"] == [] and out["license_open"] is None
    assert "voice cloning" in out["keywords"]
    assert len(fake.calls) == 1  # everything in one request


def test_make_plan_uses_jev_and_code_for_numbers(fake):
    p = query.make_plan("voice model for my phone with 6GB RAM")
    assert p.parser.startswith("jev") and p.device == "phone"
    assert p.memory_gb == 6.0 and p.ram_cap_gb == 3.0  # numbers come from code, not Jev


def test_fit_scores(fake):
    recs = [{"id": "a/x", "task": "text-to-speech"}, {"id": "b/y", "task": "text-to-speech", "params": 82_000_000}]
    assert jev.fit_scores("q", "text to speech", recs) == {"a/x": 0.1, "b/y": 0.9}
    _, qs = fake.calls[-1]
    assert "b/y" in qs["c1"].instructions


def test_falls_back_when_jev_fails(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(config, "PARSER", "auto")
    monkeypatch.setattr(jev, "client", lambda: FakeClient(fail=True))
    monkeypatch.setattr(query.llm, "available", lambda: False)
    p = query.make_plan("voice model on my phone")
    assert p.parser == "rules" and p.tasks == ["text-to-speech"] and p.device == "phone"


def test_not_used_without_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert not jev.available()


def test_env_file_loader(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# comment\nexport MF_TEST_A='one'\nMF_TEST_B=two\nMF_TEST_C=keep\n")
    monkeypatch.delenv("MF_TEST_A", raising=False)
    monkeypatch.delenv("MF_TEST_B", raising=False)
    monkeypatch.setenv("MF_TEST_C", "already")
    config._load_env_file(f)
    import os
    assert os.environ["MF_TEST_A"] == "one" and os.environ["MF_TEST_B"] == "two" and os.environ["MF_TEST_C"] == "already"


def test_quick_answer():
    res = [{"id": "hexgrad/Kokoro-82M", "best_variant": "q4", "best_variant_repo": "onnx-community/Kokoro-82M-ONNX",
            "ram_gb_min": 0.06}, {"id": "Qwen/Qwen3-TTS", "ram_gb_min": 1.2}]
    a = query.quick_answer(res)
    assert a.startswith("Try hexgrad/Kokoro-82M first (use the q4 build onnx-community/Kokoro-82M-ONNX)")
    assert "Alternative: Qwen/Qwen3-TTS" in a
