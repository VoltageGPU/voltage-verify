# SPDX-License-Identifier: MIT
"""`voltage-verify quote` on a quote that was NOT produced on VoltageGPU.

The fixture is Google's production Sapphire Rapids TDX quote from go-tdx-guest (see
tests/fixtures/third-party/NOTICE.txt). Offline, the command must parse it and verify the
signature chain up to the pinned Intel root; the collateral step is reported as a warning.
"""
import json
from pathlib import Path

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


def _run(argv, capsys):
    main(argv)
    return capsys.readouterr().out
