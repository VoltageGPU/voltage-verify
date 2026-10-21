# SPDX-License-Identifier: MIT
"""`voltage-verify quote` on a quote that was NOT produced on VoltageGPU.

The fixture is Google's production Sapphire Rapids TDX quote from go-tdx-guest (see
tests/fixtures/third-party/NOTICE.txt). Offline, the command must parse it and verify the
signature chain up to the pinned Intel root; the collateral step is reported as a warning.
"""
import json
from pathlib import Path

from voltage_verify import pcs, tdx
from voltage_verify.cli import main

THIRD_PARTY = Path(__file__).parent / "fixtures" / "third-party" / "tdx_prod_quote_SPR_E4.dat"


def test_quote_from_another_provider_verifies_offline(capsys):
    code = main(["quote", str(THIRD_PARTY), "--offline", "--json"])
    out = json.loads(capsys.readouterr().out)
    results = {c["name"]: c["result"] for c in out["checks"]}
    assert code == 0, results
    assert results["tdx.structure"] == "PASS"
    # go-tdx-guest appends "extra bytes(only for testing purpose)" after the signature
    assert results["tdx.trailing"] == "WARN"
    assert results["tdx.signatures"] == "PASS"
    assert results["tdx.collateral"] == "WARN"
    assert len(out["facts"]["tdx"]["mr_td"]) == 96
    assert len(out["facts"]["tdx"]["report_data"]) == 128
    assert "fmspc" in out["facts"]["tdx"]


def test_quote_accepts_hex_input(tmp_path, capsys):
    hexfile = tmp_path / "quote.hex"
    hexfile.write_text(THIRD_PARTY.read_bytes().hex() + "\n")
    code = main(["quote", str(hexfile), "--offline", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    direct = json.loads(_run(["quote", str(THIRD_PARTY), "--offline", "--json"], capsys))
    assert out["facts"]["tdx"]["mr_td"] == direct["facts"]["tdx"]["mr_td"]


def test_quote_rejects_wrong_report_data(capsys):
    code = main(["quote", str(THIRD_PARTY), "--offline", "--json", "--report-data", "00" * 64])
    out = json.loads(capsys.readouterr().out)
    results = {c["name"]: c["result"] for c in out["checks"]}
    assert code == 1
    assert results["tdx.report_data"] == "FAIL"
    assert results["tdx.signatures"] == "PASS"


def test_quote_rejects_a_tampered_quote(tmp_path, capsys):
    raw = bytearray(THIRD_PARTY.read_bytes())
    raw[200] ^= 0x01  # inside the TD report body
    bad = tmp_path / "bad.dat"
    bad.write_bytes(bytes(raw))
    code = main(["quote", str(bad), "--offline", "--json"])
    out = json.loads(capsys.readouterr().out)
    results = {c["name"]: c["result"] for c in out["checks"]}
    assert code == 1
    assert results["tdx.signatures"] == "FAIL"


def _with_debug_bit(tmp_path):
    raw = bytearray(THIRD_PARTY.read_bytes())
    raw[168] |= 0x01  # TD attributes bit 0, DEBUG: 48-byte header + 120 bytes into the TD report
    path = tmp_path / "debug.dat"
    path.write_bytes(bytes(raw))
    return path


def _stub_signatures(monkeypatch):
    # A debug TD gets a quote that Intel's Quoting Enclave signs like any other. We cannot
    # produce one, so the signature step is stubbed: tdx.debug has to reject it on its own.
    monkeypatch.setattr(tdx, "verify_signatures", lambda quote, **_: None)


def test_quote_shows_td_attributes_and_passes_debug(capsys):
    code = main(["quote", str(THIRD_PARTY), "--offline", "--json"])
    out = json.loads(capsys.readouterr().out)
    debug = next(c for c in out["checks"] if c["name"] == "tdx.debug")
    assert code == 0
    assert debug["result"] == "PASS"
    assert "0000004000000000" in debug["detail"]  # PKS (bit 30) only


def test_quote_from_a_debug_td_is_rejected(tmp_path, capsys):
    # Setting the bit also breaks the quote signature; the verdict must still name tdx.debug.
    code = main(["quote", str(_with_debug_bit(tmp_path)), "--offline"])
    out = capsys.readouterr().out
    assert code == 1
    assert "[FAIL] tdx.debug" in out
    verdict = out.strip().splitlines()[-1]
    assert verdict.startswith("RESULT: NOT VERIFIED") and "tdx.debug" in verdict


def test_signed_quote_from_a_debug_td_fails_on_the_debug_bit_alone(tmp_path, capsys, monkeypatch):
    _stub_signatures(monkeypatch)
    code = main(["quote", str(_with_debug_bit(tmp_path)), "--offline", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert [c["name"] for c in out["checks"] if c["result"] == "FAIL"] == ["tdx.debug"]


def test_allow_debug_turns_the_failure_into_a_visible_warning(tmp_path, capsys, monkeypatch):
    _stub_signatures(monkeypatch)
    code = main(["quote", str(_with_debug_bit(tmp_path)), "--offline", "--allow-debug"])
    out = capsys.readouterr().out
    assert code == 0
    assert "[WARN] tdx.debug" in out and "--allow-debug" in out
    assert out.strip().splitlines()[-1] == "RESULT: VERIFIED"


def test_debug_check_also_runs_online(tmp_path, capsys, monkeypatch):
    _stub_signatures(monkeypatch)

    def no_network(*_args, **_kwargs):
        raise pcs.CollateralError("no network in tests")

    monkeypatch.setattr(pcs, "fetch_collateral", no_network)
    code = main(["quote", str(_with_debug_bit(tmp_path)), "--json"])
    results = {c["name"]: c["result"] for c in json.loads(capsys.readouterr().out)["checks"]}
    assert code == 1
    assert results["tdx.debug"] == "FAIL"
    assert results["tdx.tcb"] == "FAIL"  # the online branch was taken


def test_allow_debug_does_not_excuse_a_broken_signature(tmp_path, capsys):
    code = main(["quote", str(_with_debug_bit(tmp_path)), "--offline", "--json", "--allow-debug"])
    out = json.loads(capsys.readouterr().out)
    results = {c["name"]: c["result"] for c in out["checks"]}
    assert code == 1
    assert results["tdx.debug"] == "WARN"
    assert results["tdx.signatures"] == "FAIL"


def _run(argv, capsys):
    main(argv)
    return capsys.readouterr().out
