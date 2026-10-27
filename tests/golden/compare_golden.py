#!/usr/bin/env python3
"""Compare two golden test result directories made by run_golden.sh.

Usage:
    compare_golden.py <before_dir> <after_dir>

Exit status is 0 if there are no failures, else 1. Warnings do not change the exit status.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

RESULTS_FILE = "results.txt"
XGPUINFO_FILE = "xgpuinfo.txt"
EXPORTS_FILE = "exports.txt"

STATUS_PASS = "PASS"
BUILD_KEY = "build"
BUILD_FAIL = "FAIL"

# Cases with integer-valued input. The output must be bit-identical.
EXACT_CASES = ("zero", "int_small", "accumulate", "clear", "int_large")
# Cases with non-integer input. The harness checks them against the CPU with a tolerance.
TOLERANCE_CASES = ("float",)
ALL_CASES = (*EXACT_CASES, *TOLERANCE_CASES, "timing")

INFO_PREFIX = "info."
# The version string changes with every commit, so it is not compared.
IGNORED_INFO_KEYS = ("info.version",)
XGPUINFO_VERSION_PREFIX = "xGPU library version:"

# The minimum time per call is compared, not the mean: the minimum is the speed of the GPU when nothing else
# disturbs it. The mean follows any other load or clock throttling during the run.
TIMING_MIN_KEY = "timing.min_ms"
TIMING_MAX_KEY = "timing.max_ms"
# Warn if the minimum kernel time is more than this fraction slower.
TIMING_WARN_FRACTION = 0.05
# Warn if max/min of one run is more than this. The run was disturbed and its timing is not reliable.
TIMING_NOISE_RATIO = 1.1

EXIT_OK = 0
EXIT_FAIL = 1
EXPECTED_ARGS = 3


@dataclass
class Report:
    """Collected failures and warnings for one configuration.

    Attributes:
        failures: Messages that make the comparison fail.
        warnings: Messages for information only.
    """

    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def read_results(path: Path) -> dict[str, str]:
    """Read a key=value results file.

    Args:
        path: Path to the results file.

    Returns:
        Mapping of key to value. Empty if the file does not exist.
    """
    results: dict[str, str] = {}
    if not path.is_file():
        return results
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            results[key.strip()] = value.strip()
    return results


def read_lines(path: Path) -> list[str]:
    """Read a text file as a list of lines.

    Args:
        path: Path to the file.

    Returns:
        The lines of the file. Empty if the file does not exist.
    """
    return path.read_text().splitlines() if path.is_file() else []


def compare_cases(before: dict[str, str], after: dict[str, str], report: Report) -> None:
    """Compare case status and output hashes.

    Args:
        before: Results before the change.
        after: Results after the change.
        report: Report to add messages to.
    """
    for case in ALL_CASES:
        key = f"case.{case}.status"
        if after.get(key) != STATUS_PASS:
            report.failures.append(f"{case}: status after is {after.get(key, 'missing')}")
        if before.get(key) != STATUS_PASS:
            report.warnings.append(f"{case}: status before is {before.get(key, 'missing')}")

    for case in EXACT_CASES:
        key = f"case.{case}.hash"
        if before.get(key) != after.get(key):
            report.failures.append(f"{case}: output hash changed ({before.get(key)} -> {after.get(key)})")

    for case in TOLERANCE_CASES:
        key = f"case.{case}.hash"
        if before.get(key) != after.get(key):
            norm_key = f"case.{case}.max_norm_err"
            report.warnings.append(
                f"{case}: output not bit-identical, max_norm_err vs CPU "
                f"{before.get(norm_key)} -> {after.get(norm_key)} (within tolerance if status is PASS)"
            )


def compare_info(before: dict[str, str], after: dict[str, str], report: Report) -> None:
    """Compare the xgpuInfo() values written by the harness.

    Args:
        before: Results before the change.
        after: Results after the change.
        report: Report to add messages to.
    """
    keys = {k for k in (*before, *after) if k.startswith(INFO_PREFIX) and k not in IGNORED_INFO_KEYS}
    for key in sorted(keys):
        if before.get(key) != after.get(key):
            report.failures.append(f"{key}: {before.get(key)} -> {after.get(key)}")


def timing_spread(results: dict[str, str]) -> float | None:
    """Get the ratio of the slowest to the fastest call in one run.

    Args:
        results: Results of one run.

    Returns:
        The max/min ratio, or None if a value is missing or not positive.
    """
    if TIMING_MIN_KEY not in results or TIMING_MAX_KEY not in results:
        return None
    t_min = float(results[TIMING_MIN_KEY])
    t_max = float(results[TIMING_MAX_KEY])
    return t_max / t_min if t_min > 0 else None


def compare_timing(before: dict[str, str], after: dict[str, str], report: Report) -> None:
    """Compare the minimum time per xgpuCudaXengine() call.

    Also warn if the time of one call varied much within a run (max/min above TIMING_NOISE_RATIO). Then
    the GPU was probably busy or throttled, and the run must be repeated on an idle GPU.

    Args:
        before: Results before the change.
        after: Results after the change.
        report: Report to add messages to.
    """
    if TIMING_MIN_KEY not in before or TIMING_MIN_KEY not in after:
        report.warnings.append("timing: value missing, not compared")
        return
    t_before = float(before[TIMING_MIN_KEY])
    t_after = float(after[TIMING_MIN_KEY])
    change = (t_after - t_before) / t_before if t_before > 0 else 0.0
    message = f"timing: min {t_before:.4f} ms -> {t_after:.4f} ms ({change:+.1%})"
    if change > TIMING_WARN_FRACTION:
        message += " SLOWER"
    report.warnings.append(message)

    for label, results in (("before", before), ("after", after)):
        spread = timing_spread(results)
        if spread is not None and spread > TIMING_NOISE_RATIO:
            report.warnings.append(
                f"timing: NOISY {label} run (max/min = {spread:.2f}), timing not reliable, repeat on an idle GPU"
            )


def compare_xgpuinfo(before_dir: Path, after_dir: Path, report: Report) -> None:
    """Compare xgpuinfo output, ignoring the version line.

    Args:
        before_dir: Configuration directory before the change.
        after_dir: Configuration directory after the change.
        report: Report to add messages to.
    """

    def strip(lines: list[str]) -> list[str]:
        return [line for line in lines if not line.startswith(XGPUINFO_VERSION_PREFIX)]

    before = strip(read_lines(before_dir / XGPUINFO_FILE))
    after = strip(read_lines(after_dir / XGPUINFO_FILE))
    if before != after:
        report.failures.append("xgpuinfo output changed")


def compare_exports(before_dir: Path, after_dir: Path, report: Report) -> None:
    """Compare the exported xgpu* symbols of the library.

    Args:
        before_dir: Configuration directory before the change.
        after_dir: Configuration directory after the change.
        report: Report to add messages to.
    """
    before = set(read_lines(before_dir / EXPORTS_FILE))
    after = set(read_lines(after_dir / EXPORTS_FILE))
    for name in sorted(before - after):
        report.failures.append(f"exported symbol removed: {name}")
    for name in sorted(after - before):
        report.warnings.append(f"exported symbol added: {name}")


def compare_config(before_dir: Path, after_dir: Path) -> Report:
    """Compare one tiles configuration.

    Args:
        before_dir: Configuration directory before the change.
        after_dir: Configuration directory after the change.

    Returns:
        The report for this configuration.
    """
    report = Report()
    if not after_dir.is_dir():
        report.failures.append("configuration missing in after results")
        return report
    before = read_results(before_dir / RESULTS_FILE)
    after = read_results(after_dir / RESULTS_FILE)
    if after.get(BUILD_KEY) == BUILD_FAIL:
        report.failures.append("build failed after")
        return report
    if before.get(BUILD_KEY) == BUILD_FAIL:
        report.failures.append("build failed before, no reference")
        return report
    compare_cases(before, after, report)
    compare_info(before, after, report)
    compare_timing(before, after, report)
    compare_xgpuinfo(before_dir, after_dir, report)
    compare_exports(before_dir, after_dir, report)
    return report


def main(argv: list[str]) -> int:
    """Compare all configurations in two result directories.

    Args:
        argv: Command line arguments.

    Returns:
        Process exit status.
    """
    if len(argv) != EXPECTED_ARGS:
        print(__doc__, file=sys.stderr)
        return EXIT_FAIL
    before_root = Path(argv[1])
    after_root = Path(argv[2])
    configs = sorted(p.name for p in before_root.iterdir() if p.is_dir())
    if not configs:
        print(f"No configurations in {before_root}", file=sys.stderr)
        return EXIT_FAIL

    total_failures = 0
    for name in configs:
        report = compare_config(before_root / name, after_root / name)
        verdict = "FAIL" if report.failures else "PASS"
        print(f"=== {name}: {verdict}")
        for message in report.failures:
            print(f"  FAIL: {message}")
        for message in report.warnings:
            print(f"  info: {message}")
        total_failures += len(report.failures)

    extra = sorted(p.name for p in after_root.iterdir() if p.is_dir() and p.name not in configs)
    for name in extra:
        print(f"=== {name}: not in before results, not compared")

    print(f"Total failures: {total_failures}")
    return EXIT_OK if total_failures == 0 else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main(sys.argv))
