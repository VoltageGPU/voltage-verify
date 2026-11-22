# SPDX-License-Identifier: MIT
"""Reading nvidia-smi and the kernel log: only through argument lists with a timeout, and
nothing they print is trusted before it is validated."""

from __future__ import annotations

import os
import shutil
import sys

from health_fakes import FakeHost

from voltage_verify.health import klog, smi
from voltage_verify.health.host import Completed, SystemHost, find_tool

# --- nvidia-smi --------------------------------------------------------------------------------


def test_inventory_of_the_rtx_3060_ti_fixture():
    gpus = smi.Smi(FakeHost("rtx3060ti-1x")).query(smi.INVENTORY_FIELDS)
    assert gpus is not None and len(gpus) == 1
    gpu = gpus[0]
    assert gpu["index"] == "0"
    assert gpu["name"] == "NVIDIA GeForce RTX 3060 Ti"
    assert gpu["memory_total"] == "8192"
    assert gpu["driver"] == "610.88"
    assert gpu["vbios"] == "94.04.6b.00.4a"
    assert gpu["pci_bus_id"] == "00000000:29:00.0"
    assert gpu["ecc_mode"] == "[N/A]"


def test_eight_gpus_and_only_fields_the_driver_knows_are_asked_for():
    host = FakeHost("h100-8x")
    reader = smi.Smi(host)
    assert len(reader.query(smi.INVENTORY_FIELDS) or []) == 8
    samples = reader.query(smi.SAMPLE_FIELDS)
    assert samples is not None and samples[0]["temperature"] == "33" and samples[0]["hw_thermal_slowdown"] == "Not Active"
    asked = [a for call in host.calls for a in call if a.startswith("--query-gpu=")]
    # The driver of the fixture names the slowdown flags clocks_event_reasons.*: no other alias is sent.
    assert all("clocks_throttle_reasons" not in a for a in asked)
    assert sum(1 for call in host.calls if call == ["nvidia-smi", "--help-query-gpu"]) == 1


def test_an_older_driver_gets_the_alias_it_knows():
    class OldDriver(FakeHost):
        def run(self, argv, timeout):
            if list(argv) == ["nvidia-smi", "--help-query-gpu"]:
                return Completed(0, '"index"\n"temperature.gpu"\n"clocks_throttle_reasons.hw_slowdown"\n', "")
            return super().run(argv, timeout)

    host = OldDriver("h100-8x")
    names = smi.Smi(host).names(smi.SAMPLE_FIELDS)
    assert names == [
        ("index", "index"),
        ("temperature", "temperature.gpu"),
        ("hw_slowdown", "clocks_throttle_reasons.hw_slowdown"),
    ]


def test_no_nvidia_smi_means_no_gpu():
    reader = smi.Smi(FakeHost(None))
    assert reader.query(smi.INVENTORY_FIELDS) is None
    assert reader.text("topo", "-m") is None


def test_malformed_lines_are_dropped_and_values_are_sanitised():
    class Noisy(FakeHost):
        def run(self, argv, timeout):
            if any(a.startswith("--query-gpu=") for a in argv):
                fields = argv[1].removeprefix("--query-gpu=").split(",")
                good = ", ".join(["0", "NVIDIA \x1b[31mH100\x07 " + "x" * 500, *["1"] * (len(fields) - 2)])
                return Completed(0, f"{good}\nnot, enough, columns\nabc, {', '.join(['1'] * (len(fields) - 1))}\n", "")
            return super().run(argv, timeout)

    gpus = smi.Smi(Noisy("h100-8x")).query(smi.INVENTORY_FIELDS)
    assert gpus is not None and len(gpus) == 1  # too few columns, and an index that is not a number
    assert "\x1b" not in gpus[0]["name"] and "\x07" not in gpus[0]["name"]
    assert len(gpus[0]["name"]) <= smi.MAX_VALUE_CHARS


def test_numbers_are_read_without_floats():
    assert smi.as_int("81559") == 81559
    assert smi.as_int("[N/A]") is None and smi.as_int("-1") is None and smi.as_int("12abc") is None
    assert smi.as_milli("695.30") == 695300
    assert smi.as_milli("200") == 200000
    assert smi.as_milli("0.5") == 500
    assert smi.as_milli("38.456") == 38456
    assert smi.as_milli("N/A") is None and smi.as_milli("1e3") is None


def test_topology_matrix_of_an_nvlink_node():
    topo = smi.parse_topology(FakeHost("h100-8x").run(["nvidia-smi", "topo", "-m"], 30).stdout)
    assert topo == {"gpus": 8, "pairs": 28, "kinds": {"NV18": 28}}


def test_topology_matrix_of_a_pcie_node_and_of_one_gpu():
    pcie = smi.parse_topology(FakeHost("pcie-8x").run(["nvidia-smi", "topo", "-m"], 30).stdout)
    assert pcie == {"gpus": 8, "pairs": 28, "kinds": {"NODE": 8, "PIX": 4, "SYS": 16}}
    one = smi.parse_topology(FakeHost("rtx3060ti-1x").run(["nvidia-smi", "topo", "-m"], 30).stdout)
    assert one == {"gpus": 1, "pairs": 0, "kinds": {}}
    assert smi.parse_topology("garbage") == {"gpus": 0, "pairs": 0, "kinds": {}}


def test_nvlink_status_counts_active_and_inactive_links():
    # Constructed in the format nvidia-smi nvlink -s prints (no capture from an NVSwitch node yet).
    text = (
        "GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-0)\n\t Link 0: 26.562 GB/s\n\t Link 1: 26.562 GB/s\n\t Link 2: <inactive>\n"
        "GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-1)\n\t Link 0: 26.562 GB/s\n"
    )
    assert smi.parse_nvlink(text) == {"active": 3, "inactive": 1}
    # Real output of a GPU without NVLink (RTX 3060 Ti, driver 610.88, 22 November 2026).
    assert smi.parse_nvlink("GPU 0: NVIDIA GeForce RTX 3060 Ti (UUID: GPU-0)\nDevice does not have or support Nvlink\n") == {
        "active": 0,
        "inactive": 0,
    }


# --- kernel log --------------------------------------------------------------------------------


def test_kernel_log_from_dmesg_then_from_the_journal():
    log = klog.read(FakeHost("h100-8x-sick"))
    assert log is not None and log.source == "dmesg"
    assert [(e.xid, e.pci) for e in log.events] == [(13, "PCI:0000:03:00"), (79, "PCI:0000:08:00")]
    assert log.events[0].uptime_ms == 7345 and log.events[1].uptime_ms > 5_000_000

    journal = klog.read(FakeHost("h100-8x-sick", kernel_log="journalctl"))
    assert journal is not None and journal.source == "journalctl"
    assert [e.xid for e in journal.events] == [13, 79]


def test_kernel_log_not_readable_is_none_and_sudo_is_never_tried():
    host = FakeHost("h100-8x", kernel_log="none")
    assert klog.read(host) is None
    assert all(call[0] in ("dmesg", "journalctl") for call in host.calls)
    assert klog.read(FakeHost("rtx3060ti-1x")) is None  # no dmesg.txt in that fixture: Windows, or a container


def test_a_log_without_boot_stamps_is_not_used():
    class Unstamped(FakeHost):
        def run(self, argv, timeout):
            if argv[0] == "dmesg":
                return Completed(0, "NVRM: Xid (PCI:0000:08:00): 79, GPU has fallen off the bus.\n", "")
            return None

    assert klog.read(Unstamped("h100-8x")) is None


# --- the real host -----------------------------------------------------------------------------


def test_system_host_runs_an_argument_list_without_a_shell():
    host = SystemHost()
    done = host.run([sys.executable, "-c", "import sys; print(sys.argv[1])", "a; echo injected && $(whoami)"], 30)
    assert done is not None and done.returncode == 0
    assert done.stdout.strip() == "a; echo injected && $(whoami)"


def test_system_host_reports_a_missing_tool_and_a_timeout_as_none():
    host = SystemHost()
    assert host.run(["voltage-verify-no-such-tool"], 5) is None
    assert host.run([sys.executable, "-c", "import time; time.sleep(30)"], 0.5) is None


def test_system_host_never_runs_a_tool_from_the_current_directory(tmp_path, monkeypatch):
    # Windows looks in the current directory before PATH, and a PATH may hold "." or an empty
    # entry: a file named nvidia-smi dropped where the user stands must not be run.
    planted = tmp_path / ("nvidia-smi.exe" if os.name == "nt" else "nvidia-smi")
    shutil.copy(sys.executable, planted)  # any real executable will do
    planted.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", "relative/bin"]))
    assert find_tool("nvidia-smi") is None
    assert SystemHost().run(["nvidia-smi", "-L"], 5) is None
    monkeypatch.setenv("PATH", str(tmp_path))  # named outright in PATH, it is found
    assert find_tool("nvidia-smi") == str(planted)


def test_system_host_caps_what_it_keeps_of_a_tool_output():
    done = SystemHost(max_output=1000).run([sys.executable, "-c", "print('x' * 100000)"], 30)
    assert done is not None and len(done.stdout) == 1000


def test_system_host_clock_and_uptime():
    host = SystemHost()
    assert host.clock() > 1_700_000_000
    up = host.uptime()
    assert up is None or up > 0
