import json
import subprocess
import sys

import numpy as np
from conftest import REPOS, classism_hash, make_work

from kdaa.pack import csv_clip_ids, pack_session

SYMS = ["ㄱ", "ㅏ", "<sp>", "ㄴ", "<bs>"]


def classism_check(work, tmp_path, sessions):
    """classism's own dataset check (`python -m src.dataset`) on kdaa's shards, configured as kdaa configures it."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("data:\n  sessions: " + json.dumps(sessions) + "\n  on_missing_mel: drop\n  require_param_hash: true\n",
                   encoding="utf-8")
    out = tmp_path / "check.json"
    r = subprocess.run([sys.executable, "-m", "src.dataset", "--data", str(work), "--config", str(cfg), "--json", str(out)],
                       cwd=REPOS["classism"], capture_output=True, text=True, encoding="utf-8")
    return r, (json.loads(out.read_text(encoding="utf-8")) if out.exists() else None)


def test_pack_writes_one_shard_per_session_in_csv_order(tmp_path):
    work = make_work(tmp_path / "w", {"P01_A": SYMS}, missing={"P01_A_00000002"})
    r = pack_session(work, "P01_A")
    assert (r.rows, r.packed, r.missing) == (5, 4, 1)
    with np.load(r.path, allow_pickle=False) as z:
        assert z["logmel"].shape == (4, 64, 48) and z["logmel"].dtype == np.float32
        assert z["clip_id"].tolist() == [c for c in csv_clip_ids(work / "labels" / "P01_A.csv") if c != "P01_A_00000002"]
        assert str(z["param_hash"]) == classism_hash()
        np.testing.assert_array_equal(z["logmel"][0], np.load(work / "mel" / "P01_A_00000000.npy"))


def test_pack_can_drop_the_per_clip_files(tmp_path):
    work = make_work(tmp_path / "w", {"P01_A": SYMS})
    pack_session(work, "P01_A", remove_clip_npy=True)
    assert not list((work / "mel").glob("*.npy"))


def test_classism_reads_kdaa_shards_for_the_listed_sessions_only(tmp_path):
    work = make_work(tmp_path / "w", {"P01_A": SYMS, "P01_B": SYMS, "P02_C": SYMS}, missing={"P01_B_00000001"})
    for sid in ("P01_A", "P01_B", "P02_C"):
        pack_session(work, sid, remove_clip_npy=True)  # only the shards are left to read
    r, doc = classism_check(work, tmp_path, ["P01_A", "P01_B"])
    assert r.returncode == 1, r.stdout + r.stderr  # 1 = coverage short (5 clips per session), not an error
    assert doc["sessions"] == ["P01_A", "P01_B"] == doc["shard_sessions"]
    assert (doc["n_samples"], doc["dropped_no_mel"]) == (9, 1)
    assert doc["param_hash"] == classism_hash() and doc["coverage_ok"] is False


def test_classism_refuses_shards_from_another_regime(tmp_path):
    work = make_work(tmp_path / "w", {"P01_A": SYMS}, param_hash="5711c12b")
    pack_session(work, "P01_A")
    r, _ = classism_check(work, tmp_path, ["P01_A"])
    assert r.returncode == 3
    assert "5711c12b" in r.stderr and "NFR-2" in r.stderr
