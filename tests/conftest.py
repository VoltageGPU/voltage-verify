# SPDX-License-Identifier: MIT
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def quote_bytes() -> bytes:
    return (FIXTURES / "quote_h200_2026-09-04.bin").read_bytes()


@pytest.fixture(scope="session")
def report_data() -> bytes:
    return (FIXTURES / "report_data_h200_2026-09-04.bin").read_bytes()


@pytest.fixture(scope="session")
def nras_h200() -> list:
    return json.loads((FIXTURES / "nras_h200_2026-09-04.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def nras_ppcie() -> list:
    return json.loads((FIXTURES / "nras_ppcie_8xh100_2026-09-10.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def jwks() -> dict:
    return json.loads((FIXTURES / "nras_jwks_2026-09-12.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def jwks_oct22() -> dict:
    """Two keys copied from NRAS's live JWKS on 22 October 2026, one under each intermediate."""
    return json.loads((FIXTURES / "nras_jwks_2026-10-22_two_intermediates.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def collateral_json() -> dict:
    return json.loads((FIXTURES / "intel_collateral_90c06f000000_2026-09-12.json").read_text(encoding="utf-8"))
