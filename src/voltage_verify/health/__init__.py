# SPDX-License-Identifier: MIT
"""Hardware health check for NVIDIA GPUs: inventory, error counters, sensors under a variable
load, verified compute and a VRAM pattern test. Local, read-only, offline: nothing is sent
anywhere. The report it writes can be committed in a manifest (``manifest --health``), so that
an attestation covers it too.

Nothing is imported here on purpose: ``manifest`` and ``verify`` only need ``health.report``.
"""
