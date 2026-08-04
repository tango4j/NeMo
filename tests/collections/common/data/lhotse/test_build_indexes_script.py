import os

from scripts.dataloading.build_indexes import JSONL, IndexJob, _build_one, _is_indexed


def test_is_indexed_requires_valid_nonempty_uint64_sidecar(tmp_path):
    source = tmp_path / "manifest.jsonl"
    source.write_text("{}\n")
    job = IndexJob(str(source), JSONL)
    sidecar = tmp_path / "manifest.jsonl.idx"

    assert not _is_indexed(job)

    sidecar.mkdir()
    assert not _is_indexed(job)
    sidecar.rmdir()

    sidecar.write_bytes(b"")
    assert not _is_indexed(job)

    sidecar.write_bytes(b"123456789")
    assert not _is_indexed(job)


def test_is_indexed_rejects_sidecar_older_than_local_source(tmp_path):
    source = tmp_path / "manifest.jsonl"
    source.write_text("{}\n")
    sidecar = tmp_path / "manifest.jsonl.idx"
    sidecar.write_bytes(source.stat().st_size.to_bytes(8, "little"))
    job = IndexJob(str(source), JSONL)

    os.utime(source, ns=(1_000_000_000, 1_000_000_000))
    os.utime(sidecar, ns=(2_000_000_000, 2_000_000_000))
    assert _is_indexed(job)

    os.utime(source, ns=(3_000_000_000, 3_000_000_000))
    assert not _is_indexed(job)


def test_is_indexed_rejects_newer_sidecar_with_wrong_source_size_sentinel(tmp_path):
    source = tmp_path / "manifest.jsonl"
    source.write_text("")
    sidecar = tmp_path / "manifest.jsonl.idx"
    sidecar.write_bytes((1_543_482).to_bytes(8, "little"))
    job = IndexJob(str(source), JSONL)

    os.utime(source, ns=(1_000_000_000, 1_000_000_000))
    os.utime(sidecar, ns=(2_000_000_000, 2_000_000_000))
    assert not _is_indexed(job)

    _build_one(job)
    assert _is_indexed(job)
