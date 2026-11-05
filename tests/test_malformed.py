# SPDX-License-Identifier: MIT
"""A malformed input is a named failure with a valid --json report, never a traceback.

Up to 0.2.3 an unreadable bundle printed an error on stderr and nothing on stdout (no JSON for
--json), and a bundle that parsed but carried a field of the wrong type (a manifest that is a
list, a collateral that is a string, a manifest float, a date without a UTC offset) stopped the
verifier with a Python traceback. Found in the audit of 5 November 2026; the cases below come
from replacing each field of the reference bundle with values of the wrong type.

An unusable bundle is ``bundle.malformed`` (exit code 2). A field that a check already owns
fails that check (``manifest.format``, ``tdx.structure``, ``tdx.tcb``, exit code 1).
"""

from __future__ import annotations

import base64
import copy
import json
import struct
from pathlib import Path

import pytest

from voltage_verify import verify as verify_mod
from voltage_verify.cli import main

EXAMPLE = Path(__file__).parent.parent / "examples" / "hello-workload" / "bundle-8xh100-2026-09-12.json"
THIRD_PARTY = Path(__file__).parent / "fixtures" / "third-party" / "tdx_prod_quote_SPR_E4.dat"
BASE = json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _run(capsys, argv: list[str]) -> tuple[int, dict]:
    code = main([*argv, "--json"])
    out = capsys.readouterr().out
    return code, json.loads(out)  # must be valid JSON, whatever the input


def _verify(tmp_path, capsys, text: str) -> tuple[int, dict]:
    path = tmp_path / "bundle.json"
    path.write_bytes(text.encode("utf-8"))
    return _run(capsys, ["verify", str(path), "--offline"])


def _failed(out: dict) -> list[str]:
    return [c["name"] for c in out["checks"] if c["result"] == "FAIL"]


# --- unusable bundles: bundle.malformed, exit 2 -------------------------------------------------


def _set(path: tuple, value) -> str:
    data = copy.deepcopy(BASE)
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return json.dumps(data)


UNUSABLE = {
    "not JSON": "{not json",
    "empty file": "",
    "a JSON list": "[]",
    "deeply nested": "[" * 100_000 + "]" * 100_000,
    "unknown format": "{}",
    "manifest is a list": _set(("manifest",), []),
    "manifest is a string": _set(("manifest",), "x"),
    "collateral is a string": _set(("collateral",), "x"),
    "nvidia is a list": _set(("nvidia",), []),
    "created_at without UTC offset": _set(("created_at",), "2026-09-11T23:22:24"),
}


@pytest.mark.parametrize("text", UNUSABLE.values(), ids=UNUSABLE)
def test_unusable_bundle_is_bundle_malformed_with_valid_json(tmp_path, capsys, text):
    code, out = _verify(tmp_path, capsys, text)
    assert code == 2
    assert out["ok"] is False and out["verdict"] == "NOT VERIFIED (bundle.malformed)"
    assert [(c["name"], c["result"]) for c in out["checks"]] == [("bundle.malformed", "FAIL")]


def test_unusable_bundle_in_text_mode(tmp_path, capsys):
    path = tmp_path / "bundle.json"
    path.write_text("[]", encoding="utf-8")
    assert main(["verify", str(path)]) == 2
    captured = capsys.readouterr()
    assert "[FAIL] bundle.malformed" in captured.out
    assert captured.out.strip().splitlines()[-1] == "RESULT: NOT VERIFIED (bundle.malformed)"
    assert "error:" in captured.err


def test_unexpected_error_while_verifying_is_bundle_malformed(tmp_path, capsys, monkeypatch):
    def boom(*_args, **_kwargs):
        raise TypeError("something this version did not foresee")

    monkeypatch.setattr("voltage_verify.cli.verify_bundle", boom)
    code, out = _run(capsys, ["verify", str(EXAMPLE), "--offline"])
    assert code == 2 and _failed(out) == ["bundle.malformed"]
    assert "TypeError" in out["checks"][0]["detail"]


# --- a field a check owns: that check fails, exit 1 ---------------------------------------------


def _quote_with_auth_length(length: int) -> str:
    raw = bytearray(base64.b64decode(BASE["tdx"]["quote_b64"]))
    struct.pack_into("<H", raw, 632 + 4 + 134 + 384 + 64, length)  # QE authentication data length
    return base64.b64encode(bytes(raw)).decode()


FIELD = {
    "image is a string": ((("manifest", "workload", "image"), "x"), "manifest.format"),
    "artifacts is a string": ((("manifest", "workload", "artifacts"), "x"), "manifest.format"),
    "artifacts hold strings": ((("manifest", "workload", "artifacts"), ["x"]), "manifest.format"),
    "artifacts is null": ((("manifest", "workload", "artifacts"), None), "manifest.format"),
    "a float in the manifest": ((("manifest", "workload", "extra"), {"f": 1.5}), "manifest.format"),
    "intel collateral is a string": ((("collateral", "intel"), "x"), "tdx.tcb"),
    "intel collateral lacks tcb_info": ((("collateral", "intel"), {"a": 1}), "tdx.tcb"),
    "tcb_info is null": ((("collateral", "intel", "tcb_info"), None), "tdx.tcb"),
    "issuer chain is a number": ((("collateral", "intel", "tcb_info", "issuer_chain_pem"), 1), "tdx.tcb"),
    "PCK CRL is not PEM": ((("collateral", "intel", "pck_crl_pem"), "x"), "tdx.revocation"),
    "PCK CRL is a list": ((("collateral", "intel", "pck_crl_pem"), [1]), "tdx.tcb"),
    "root CRL is not base64": ((("collateral", "intel", "root_crl_der_b64"), "x"), "tdx.revocation"),
    "quote overruns its own length fields": ((("tdx", "quote_b64"), _quote_with_auth_length(0xFFFF)), "tdx.structure"),
}


@pytest.mark.parametrize(("change", "check"), FIELD.values(), ids=FIELD)
def test_malformed_field_fails_the_check_that_owns_it(tmp_path, capsys, change, check):
    code, out = _verify(tmp_path, capsys, _set(*change))
    assert code == 1 and check in _failed(out), _failed(out)
    assert out["verdict"].startswith("NOT VERIFIED (")


# --- quote ------------------------------------------------------------------------------------


@pytest.mark.parametrize("content", [b"abc\n", b"0" * 7, b"\x04\x00\x02\x00\x81" + b"\x00" * 3])
def test_malformed_quote_file_is_a_named_failure(tmp_path, capsys, content):
    path = tmp_path / "quote.bin"
    path.write_bytes(content)
    code, out = _run(capsys, ["quote", str(path), "--offline"])
    assert code == 1 and _failed(out) == ["tdx.structure"]


def test_quote_that_overruns_its_length_fields_is_a_named_failure(tmp_path, capsys):
    path = tmp_path / "quote.bin"
    path.write_bytes(base64.b64decode(_quote_with_auth_length(0xFFFF)))
    code, out = _run(capsys, ["quote", str(path), "--offline"])
    assert code == 1 and _failed(out) == ["tdx.structure"]


def test_the_safety_net_is_not_hiding_a_valid_bundle(capsys):
    code, out = _run(capsys, ["verify", str(EXAMPLE), "--offline", "--max-collateral-age", "3650"])
    assert code == 3 and out["ok"] and "bundle.malformed" not in json.dumps(out["checks"])
    assert verify_mod.BUNDLE_TCB_OFFLINE in out["limits"]
