import dataclasses

import pytest

from kdaa.mel_shim import MelRegimeError, load_classism_melio, load_meligation, open_backend, regime_mismatches


def test_pinned_meligation_extracts_with_classism_values(repos):
    mel = load_meligation(repos["meligation"]).Regime()
    cls = load_classism_melio(repos["classism"]).REGIME
    assert regime_mismatches(mel, cls) == []


def test_backend_stamps_the_hash_classism_accepts(repos):
    b = open_backend(repos["meligation"], repos["classism"])
    cls_hash = load_classism_melio(repos["classism"]).REGIME.param_hash()
    assert b.param_hash == cls_hash
    if b.own_hash != cls_hash:  # pinned meligation still hashes its own field names (PLAN.md B1)
        assert b.hash_source == "kdaa-shim"
        assert b.meta()["meligation_param_hash"] == b.own_hash


def test_a_real_value_difference_is_named(repos):
    mel = dataclasses.replace(load_meligation(repos["meligation"]).Regime(), hop_length=128, post_ms=45.0)
    bad = regime_mismatches(mel, load_classism_melio(repos["classism"]).REGIME)
    assert any(b.startswith("hop_length") for b in bad)
    assert any(b.startswith("pre_ms+post_ms") for b in bad)


def test_drifted_meligation_is_refused(repos, tmp_path):
    src = (repos["meligation"] / "code").read_text(encoding="utf-8")
    assert "hop_length: int = 96" in src
    (tmp_path / "code").write_text(src.replace("hop_length: int = 96", "hop_length: int = 120"), encoding="utf-8")
    with pytest.raises(MelRegimeError, match="hop_length"):
        open_backend(tmp_path, repos["classism"])
