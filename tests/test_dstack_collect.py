# SPDX-License-Identifier: MIT
"""The dstack collector (contrib/dstack-b200/collect.py).

Pure parts are tested directly; the end-to-end build reuses the real 8x H100 reference bundle,
faking only the dstack transport and the NVIDIA SDK, so the resulting bundle is checked by the
genuine verifier. The collector's safety rule is the important one: if the quote's report_data
is not our manifest hash byte for byte, it refuses to write a bundle and names what it saw.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from voltage_verify import pcs
from voltage_verify.bundle import Bundle
from voltage_verify.canonical import commitments
from voltage_verify.verify import verify_bundle

REPO = Path(__file__).parent.parent
EXAMPLE = REPO / "examples" / "hello-workload" / "bundle-8xh100-2026-09-12.json"
MANIFEST = REPO / "examples" / "hello-workload" / "manifest-8xh100.json"
CHALLENGE = "ef7f54c160627ee536a02b5e73896ac4b1ac2cdefdb7cfaaf2366a7d33cc8731"
FIXTURES = Path(__file__).parent / "fixtures"


def _load_collect():
    spec = importlib.util.spec_from_file_location("dstack_collect", REPO / "contrib" / "dstack-b200" / "collect.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collect = _load_collect()


def test_parse_quote_hex_accepts_prefix_and_whitespace():
    assert collect.parse_quote_hex("0400") == b"\x04\x00"
    assert collect.parse_quote_hex("0x0400") == b"\x04\x00"
    assert collect.parse_quote_hex("04 00\n81") == b"\x04\x00\x81"
    with pytest.raises(collect.CollectError, match="no 'quote'"):
        collect.parse_quote_hex(None)
    with pytest.raises(collect.CollectError, match="not hex"):
        collect.parse_quote_hex("zz")


def test_http_body_requires_200():
    assert collect.http_body(b"HTTP/1.0 200 OK\r\nX: 1\r\n\r\n{}", "/Info") == b"{}"
    with pytest.raises(collect.CollectError, match="malformed"):
        collect.http_body(b"no headers no body", "/Info")
    with pytest.raises(collect.CollectError, match="404"):
        collect.http_body(b"HTTP/1.0 404 Not Found\r\n\r\n", "/GetQuote")


def test_classify_report_data_raw_vs_transforms():
    c512 = bytes(range(64))
    c256 = bytes(range(32))
    sent = c512.hex()
    assert collect.classify_report_data(c512, c512, c256, sent) == "raw"
    classify = collect.classify_report_data
    assert classify(hashlib.sha512(c512).digest(), c512, c256, sent) == "sha512(report_data bytes)"
    assert classify(hashlib.sha512(sent.encode()).digest(), c512, c256, sent) == "sha512(report_data hex-ascii)"
    assert classify(b"\x00" * 64, c512, c256, sent) == "all zero (empty/ignored report_data)"
    assert collect.classify_report_data(b"\xab" * 64, c512, c256, sent) == "unknown"


def test_environment_facts_keeps_a_dstack_block_without_the_bulk():
    info = {"app_id": "abc", "instance_id": "def", "app_cert": "-----BEGIN...", "event_log": [{"imr": 3}]}
    quote_response = {"quote": "0400deadbeef", "compose_hash": "sha256:1234"}
    facts = collect.environment_facts(info, quote_response)
    assert facts["dstack"] == {
        "app_id": "abc",
        "instance_id": "def",
        "event_log": [{"imr": 3}],
        "compose_hash": "sha256:1234",
    }
    assert "app_cert" not in facts["dstack"] and "quote" not in facts["dstack"]


APP_ID = "0cc30085543511090db2952809f104dd30ec7d86"


def _fake_transport(monkeypatch, quote_hex: str, info: dict | None = None):
    info = info or {"app_id": APP_ID, "instance_id": "80fbfce9", "event_log": [{"imr": 3}]}

    def fake_get(_socket_path: str, path: str) -> dict:
        if path == "/Info":
            return info
        if path.startswith("/GetQuote?report_data="):
            return {"quote": quote_hex, "event_log": info.get("event_log", [])}
        raise AssertionError(path)

    monkeypatch.setattr(collect, "dstack_get_json", fake_get)


def test_collect_builds_a_bundle_that_verifies(monkeypatch, tmp_path):
    example = Bundle.load(EXAMPLE)
    _fake_transport(monkeypatch, example.quote.hex())
    # The NVIDIA SDK and NRAS JWKS are faked with the reference bundle's real signed material.
    monkeypatch.setattr("voltage_verify.attest.nras_attest", lambda nonce, mode: example.nras)
    monkeypatch.setattr("voltage_verify.attest.fetch_json", lambda url: example.jwks)
    monkeypatch.setattr(pcs, "fetch_collateral", lambda *a, **k: pcs.Collateral.from_json(example.collateral["intel"]))

    out = tmp_path / "bundle.json"
    built = collect.collect(MANIFEST, "/var/run/dstack.sock", out, mode="multi-gpu-nvle")
    assert built.nvidia_mode == "multi-gpu-nvle"
    assert built.environment["dstack"]["app_id"] == APP_ID

    report = verify_bundle(Bundle.load(out), online=False, expected_challenge=CHALLENGE, require_devices=8)
    # Every binding check passes. The only tolerated failure is nvidia.freshness: the reused
    # reference token is dated September 2026, so replaying it today reads as stale (a real run
    # builds the bundle seconds after NRAS, when it passes).
    assert set(report.failed()) <= {"nvidia.freshness"}, report.failed()
    names = {c.name: c.label for c in report.checks}
    assert names["nvidia.mode"] == "PASS"
    assert names["nvidia.scope"] == "WARN"  # NVLink fabric named as not attested
    assert names["tdx.measured_compose"] == "WARN"  # dstack composition trusted, not proven


def test_collect_refuses_when_report_data_is_not_our_hash(monkeypatch, tmp_path):
    # A quote whose report_data is a different value (the H200 fixture): the collector must refuse.
    other = (FIXTURES / "quote_h200_2026-09-04.bin").read_bytes()
    _fake_transport(monkeypatch, other.hex())
    monkeypatch.setattr("voltage_verify.attest.nras_attest", lambda nonce, mode: Bundle.load(EXAMPLE).nras)
    with pytest.raises(collect.CollectError, match="does NOT carry our manifest hash"):
        collect.collect(MANIFEST, "/var/run/dstack.sock", tmp_path / "b.json", mode="multi-gpu-nvle")


def test_collect_reports_a_hashed_report_data_path(monkeypatch, tmp_path):
    # Simulate a dstack that SHA-512s the report_data we send: the message must name that transform.
    com = commitments(json.loads(MANIFEST.read_text(encoding="utf-8")))
    example = Bundle.load(EXAMPLE)
    raw = bytearray(example.quote)
    raw[568:632] = hashlib.sha512(com.sha512).digest()  # overwrite report_data in the TD report
    _fake_transport(monkeypatch, bytes(raw).hex())
    monkeypatch.setattr("voltage_verify.attest.nras_attest", lambda nonce, mode: example.nras)
    with pytest.raises(collect.CollectError, match="sha512\\(report_data bytes\\)"):
        collect.collect(MANIFEST, "/var/run/dstack.sock", tmp_path / "b.json", mode="multi-gpu-nvle")
