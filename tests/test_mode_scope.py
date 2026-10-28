# SPDX-License-Identifier: MIT
"""Attestation mode and scope reporting.

The verifier must name the NVIDIA attestation mode, refuse one it does not know, and say
outright what a per-GPU attestation does NOT cover: the NVSwitch fabric on a Protected PCIe
node, the encrypted NVLink fabric on a Blackwell NVLE node, and the provider-measured
composition (dstack / RTMR3) when the TD is not the tenant's alone.

The NVLE wiring is exercised by relabelling the real 8x H100 Hopper reference bundle: only the
mode string and the environment change, so the signatures still verify and the test stays
deterministic. It is NOT a real B200 bundle and asserts nothing about Blackwell hwmodels.
"""

import dataclasses
from pathlib import Path

from voltage_verify.bundle import Bundle
from voltage_verify.verify import KNOWN_NVIDIA_MODES, nvidia_mode_scope, verify_bundle

EXAMPLE = Path(__file__).parent.parent / "examples" / "hello-workload" / "bundle-8xh100-2026-09-12.json"
CHALLENGE = "ef7f54c160627ee536a02b5e73896ac4b1ac2cdefdb7cfaaf2366a7d33cc8731"


def _check(report, name):
    return next((c for c in report.checks if c.name == name), None)


def test_mode_scope_is_pure_and_honest():
    assert nvidia_mode_scope("single-gpu") == (True, None)
    known, ppcie = nvidia_mode_scope("multi-gpu-ppcie")
    assert known and "NVSwitch fabric" in ppcie and "NOT attested" in ppcie
    known, nvle = nvidia_mode_scope("multi-gpu-nvle")
    assert known and "NVLink fabric" in nvle and "NOT attested" in nvle
    assert nvidia_mode_scope("multi-gpu-nvle")[0] is True
    unknown, scope = nvidia_mode_scope("something-else")
    assert unknown is False and scope is None
    assert set(KNOWN_NVIDIA_MODES) == {"single-gpu", "multi-gpu-ppcie", "multi-gpu-nvle"}


def test_known_mode_passes_and_names_the_unverified_fabric():
    report = verify_bundle(Bundle.load(EXAMPLE), online=False, expected_challenge=CHALLENGE)
    assert report.ok, report.failed()
    mode = _check(report, "nvidia.mode")
    assert mode is not None and mode.label == "PASS" and "multi-gpu-ppcie" in mode.detail
    scope = _check(report, "nvidia.scope")
    assert scope is not None and scope.label == "WARN" and "NVSwitch fabric" in scope.detail


def test_nvle_mode_verifies_and_names_the_nvlink_fabric():
    nvle = dataclasses.replace(Bundle.load(EXAMPLE), nvidia_mode="multi-gpu-nvle")
    report = verify_bundle(nvle, online=False, expected_challenge=CHALLENGE)
    assert report.ok, report.failed()  # mode is known, signatures unchanged
    assert _check(report, "nvidia.mode").label == "PASS"
    scope = _check(report, "nvidia.scope")
    assert scope.label == "WARN" and "NVLink fabric" in scope.detail and "NOT attested" in scope.detail


def test_unknown_mode_fails_closed():
    bogus = dataclasses.replace(Bundle.load(EXAMPLE), nvidia_mode="multi-gpu-teleport")
    report = verify_bundle(bogus, online=False, expected_challenge=CHALLENGE)
    assert not report.ok and "nvidia.mode" in report.failed()
    assert _check(report, "nvidia.mode").label == "FAIL"


def test_dstack_measured_composition_is_flagged_but_not_fatal():
    env = {"kernel": "6.9.0-dstack", "dstack": {"app_id": "0cc30085543511090db2952809f104dd30ec7d86"}}
    on_dstack = dataclasses.replace(Bundle.load(EXAMPLE), nvidia_mode="multi-gpu-nvle", environment=env)
    report = verify_bundle(on_dstack, online=False, expected_challenge=CHALLENGE)
    compose = _check(report, "tdx.measured_compose")
    assert compose is not None and compose.label == "WARN"
    assert "0cc30085543511090db2952809f104dd30ec7d86" in compose.detail
    assert "trusted, not proven" in compose.detail
    assert report.ok  # a WARN never fails the bundle


def test_our_own_vm_bundle_has_no_measured_composition_warning():
    # No dstack in the environment (our Confidential VMs are the tenant's alone): the check is absent.
    report = verify_bundle(Bundle.load(EXAMPLE), online=False, expected_challenge=CHALLENGE)
    assert _check(report, "tdx.measured_compose") is None
