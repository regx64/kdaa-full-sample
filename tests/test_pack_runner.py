import json
import subprocess
import sys

import numpy as np
import pytest
from conftest import REPOS, ROOT, classism_hash, make_work

from kdaa.pack import ShardError, ShardStore, csv_clip_ids, pack_session

SYMS = ["ㄱ", "ㅏ", "<sp>", "ㄴ", "<bs>"]


def runner(work, sessions, *extra):
    return subprocess.run([sys.executable, "-m", "kdaa.classism_runner", "check", "--classism", str(REPOS["classism"]),
                           "--data", str(work), "--sessions", ",".join(sessions), *extra],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8")


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
    assert len(ShardStore(work / "mel_packed", ["P01_A"])) == 5


def test_store_refuses_a_shard_from_another_regime(tmp_path):
    work = make_work(tmp_path / "w", {"P01_A": SYMS}, param_hash="5711c12b")
    pack_session(work, "P01_A")
    with pytest.raises(ShardError, match="5711c12b"):
        ShardStore(work / "mel_packed", ["P01_A"], expected_hash=classism_hash())


def test_runner_check_reads_shards_and_keeps_only_listed_sessions(tmp_path):
    work = make_work(tmp_path / "w", {"P01_A": SYMS, "P01_B": SYMS, "P02_C": SYMS}, missing={"P01_B_00000001"})
    for sid in ("P01_A", "P01_B", "P02_C"):
        pack_session(work, sid)
    out = tmp_path / "check.json"
    r = runner(work, ["P01_A", "P01_B"], "--out", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["sessions"] == ["P01_A", "P01_B"]
    assert (doc["n_samples"], doc["n_dropped_no_mel"]) == (9, 1)
    assert doc["param_hash"] == classism_hash() and set(doc["shard_hashes"]) == {"P01_A", "P01_B"}
    assert doc["coverage_ok"] is False  # 5 clips per session is far below spec §7


def test_runner_exits_3_on_param_hash_mismatch(tmp_path):
    work = make_work(tmp_path / "w", {"P01_A": SYMS}, param_hash="5711c12b")
    pack_session(work, "P01_A")
    r = runner(work, ["P01_A"])
    assert r.returncode == 3
    assert "NFR-2" in r.stderr
