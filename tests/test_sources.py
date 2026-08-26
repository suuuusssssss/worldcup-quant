"""Download cache tests with the network stubbed out."""
import io
import hashlib

import pytest

from wcq.data import sources


class FakeResponse(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


@pytest.fixture
def stub_network(monkeypatch):
    calls = []
    payload = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        payload["n"] += 1
        return FakeResponse(b"date,home_team\n2020-01-01,X\n")

    monkeypatch.setattr(sources.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_fetch_downloads_once_then_caches(tmp_path, stub_network):
    a = sources.fetch("international", cache=tmp_path)
    b = sources.fetch("international", cache=tmp_path)
    assert a == b and a.exists()
    assert len(stub_network) == 1, "second call must not hit the network"


def test_force_redownloads(tmp_path, stub_network):
    sources.fetch("international", cache=tmp_path)
    sources.fetch("international", cache=tmp_path, force=True)
    assert len(stub_network) == 2


def test_digest_sidecar_is_written_and_verifies(tmp_path, stub_network):
    p = sources.fetch("international", cache=tmp_path)
    sidecar = tmp_path / (p.name + ".sha256")
    assert sidecar.exists()
    assert sidecar.read_text().strip() == hashlib.sha256(p.read_bytes()).hexdigest()
    assert sources.verify("international", cache=tmp_path)


def test_verify_detects_a_changed_file(tmp_path, stub_network):
    """The point of the digest: a silently-updated upstream CSV would make an
    old backtest number irreproducible without any error being raised."""
    p = sources.fetch("international", cache=tmp_path)
    p.write_bytes(b"tampered")
    assert not sources.verify("international", cache=tmp_path)


def test_verify_false_when_nothing_cached(tmp_path):
    assert not sources.verify("international", cache=tmp_path)


def test_no_partial_files_are_left_behind(tmp_path, stub_network):
    sources.fetch("international", cache=tmp_path)
    assert not list(tmp_path.glob("*.part"))


def test_unknown_dataset_is_a_key_error(tmp_path):
    with pytest.raises(KeyError):
        sources.fetch("does-not-exist", cache=tmp_path)


def test_every_registered_source_is_well_formed():
    for name, s in sources.SOURCES.items():
        assert s.name == name
        assert s.url.startswith("https://")
        assert s.filename and s.description and s.licence
