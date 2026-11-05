# SPDX-License-Identifier: MIT
"""Offline checks of Intel collateral handling, using a snapshot fetched on 12 September 2026."""

import copy
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from voltage_verify import pcs, tdx
from voltage_verify.bundle import Bundle
from voltage_verify.verify import verify_bundle

AT = datetime(2026, 9, 12, tzinfo=timezone.utc)
EXAMPLE = Path(__file__).parent.parent / "examples" / "hello-workload" / "bundle-8xh100-2026-09-12.json"


def test_collateral_documents_verify(collateral_json):
    col = pcs.Collateral.from_json(collateral_json)
    pcs.verify_document(col.tcb_info, at=AT)
    pcs.verify_document(col.qe_identity, at=AT)
    assert col.tcb_info.inner["fmspc"].lower() == "90c06f000000"
    assert col.qe_identity.inner["id"] == "TD_QE"


def test_tampered_collateral_fails(collateral_json):
    col = pcs.Collateral.from_json(collateral_json)
    col.tcb_info.body = col.tcb_info.body.replace('"tcbStatus":"UpToDate"', '"tcbStatus":"UpToDat3"', 1)
    with pytest.raises(pcs.CollateralError, match="signature"):
        pcs.verify_document(col.tcb_info, at=AT)


def test_evaluate_real_quote(quote_bytes, collateral_json):
    q = tdx.parse_quote(quote_bytes)
    pck = tdx.pck_info(q.pck_leaf)
    col = pcs.Collateral.from_json(collateral_json)
    ev = pcs.evaluate(q, pck, col, at=AT)
    assert ev.status in pcs.STATUS_ORDER
    assert ev.qe_status in pcs.STATUS_ORDER
    assert ev.tdx_module_id == "TDX_01"
    assert ev.tdx_module_status in pcs.STATUS_ORDER


def test_evaluate_rejects_wrong_fmspc(quote_bytes, collateral_json):
    q = tdx.parse_quote(quote_bytes)
    pck = tdx.pck_info(q.pck_leaf)
    col = pcs.Collateral.from_json(collateral_json)
    body = col.tcb_info.body
    assert '"fmspc":"90c06f000000"' in body
    col.tcb_info.body = body.replace('"fmspc":"90c06f000000"', '"fmspc":"00c06f000000"', 1)
    with pytest.raises(pcs.CollateralError, match="FMSPC"):
        pcs.evaluate(q, pck, col, at=AT)


def test_revocation_not_checked_without_crls(quote_bytes, collateral_json):
    q = tdx.parse_quote(quote_bytes)
    col = pcs.Collateral.from_json(collateral_json)
    notes = pcs.check_revocation(q, col, at=AT)
    assert any("NOT checked" in n for n in notes)


# --- the document Intel signed, not a second copy after the signature -------------------------
# A body {"tcbInfo":<signed>,"signature":"<hex>","tcbInfo":<forged>} verified (the signature
# covers the first object) while json.loads kept the last one: offline, a bundle could carry
# any TCB status and any issue date. Found while fixing the audit of 5 November 2026.


def _forged_copy(doc: pcs.SignedDocument, **changes) -> str:
    inner = {**json.loads(doc.signed_bytes()), **changes}
    return doc.body[:-1] + f',"{doc.inner_key}":' + json.dumps(inner, separators=(",", ":")) + "}"


@pytest.mark.parametrize("kind", ["tcb_info", "qe_identity"])
def test_second_inner_object_after_the_signature_is_refused(collateral_json, kind):
    doc = getattr(pcs.Collateral.from_json(collateral_json), kind)
    doc.body = _forged_copy(doc, issueDate="2099-01-01T00:00:00Z", nextUpdate="2099-02-01T00:00:00Z")
    with pytest.raises(pcs.CollateralError, match="layout"):
        pcs.verify_document(doc, at=AT)
    with pytest.raises(pcs.CollateralError, match="layout"):
        doc.inner  # noqa: B018


@pytest.mark.parametrize(
    "tail",
    [',"signature":"00"}', ',"signature":"' + "0" * 128 + '"} ', ',"signature":"' + "0" * 128 + '","x":1}', ""],
)
def test_anything_but_the_signature_after_the_signed_part_is_refused(collateral_json, tail):
    doc = pcs.Collateral.from_json(collateral_json).qe_identity
    doc.body = doc.body[: doc.body.rfind(',"signature":')] + tail
    with pytest.raises(pcs.CollateralError, match="layout"):
        pcs.verify_document(doc, at=AT)


def test_inner_document_is_parsed_from_the_signed_bytes(collateral_json):
    doc = pcs.Collateral.from_json(collateral_json).tcb_info
    assert doc.inner == json.loads(doc.signed_bytes())
    assert doc.payload == {"tcbInfo": doc.inner, "signature": json.loads(doc.body)["signature"]}


def test_bundle_carrying_a_forged_tcb_status_does_not_verify_offline():
    bundle = Bundle.load(EXAMPLE)
    col = pcs.Collateral.from_json(copy.deepcopy(bundle.collateral["intel"]))
    levels = [{**level, "tcbStatus": "UpToDate", "advisoryIDs": []} for level in col.tcb_info.inner["tcbLevels"]]
    col.tcb_info.body = _forged_copy(col.tcb_info, tcbLevels=levels)
    forged = dataclasses.replace(bundle, collateral={"intel": col.to_json()})
    report = verify_bundle(forged, online=False, at=AT)
    assert not report.ok and "tdx.tcb" in report.failed()
    assert "layout" in " ".join(c.detail for c in report.checks if c.name == "tdx.tcb")
