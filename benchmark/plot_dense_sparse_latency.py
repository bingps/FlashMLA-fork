import argparse
import csv
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


@dataclass
class Point:
    kernel: str
    sq: int
    batch_size: int
    time_us: float


def _run_test_script(script_path: str) -> str:
    cmd = [sys.executable, script_path]
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=pathlib.Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        print(combined_output)
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")

    return combined_output


def _parse_dense_output(output: str) -> List[Point]:
    points: List[Point] = []
    current: Tuple[int, int] | None = None  # (batch_size, sq)

    run_re = re.compile(r"Running on TestParam\(b=(\d+),\s*s_q=(\d+),")
    time_re = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s+us,")

    for line in output.splitlines():
        run_match = run_re.search(line)
        if run_match:
            current = (int(run_match.group(1)), int(run_match.group(2)))
            continue

        if current is None:
            continue

        time_match = time_re.search(line)
        if time_match:
            batch_size, sq = current
            points.append(Point("dense", sq, batch_size, float(time_match.group(1))))
            current = None

    return points


def _parse_sparse_output(output: str) -> List[Point]:
    points: List[Point] = []
    current: Tuple[int, int] | None = None  # (batch_size, sq)

    run_re = re.compile(
        r"Running on TestParam\(s_q=(\d+),.*decode=ExtraTestParamForDecode\(b=(\d+),"
    )
    time_re = re.compile(r"Time \(per\):\s*([0-9]+(?:\.[0-9]+)?)\s+us")

    for line in output.splitlines():
        run_match = run_re.search(line)
        if run_match:
            current = (int(run_match.group(2)), int(run_match.group(1)))
            continue

        if current is None:
            continue

        time_match = time_re.search(line)
        if time_match:
            batch_size, sq = current
            points.append(Point("sparse", sq, batch_size, float(time_match.group(1))))
            current = None

    return points


def _filter_and_validate(points: List[Point]) -> List[Point]:
    filtered = [p for p in points if p.sq in (1, 2)]
    grouped: Dict[Tuple[str, int], List[Point]] = {}
    for p in filtered:
        grouped.setdefault((p.kernel, p.sq), []).append(p)

    expected_groups = [("dense", 1), ("dense", 2), ("sparse", 1), ("sparse", 2)]
    missing = [g for g in expected_groups if g not in grouped]
    if missing:
        raise RuntimeError(f"Missing curve data for groups: {missing}")

    for group in expected_groups:
        grouped[group].sort(key=lambda p: p.batch_size)

    return [p for group in expected_groups for p in grouped[group]]


def _save_csv(points: List[Point], out_csv: pathlib.Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "sq", "batch_size", "time_us"])
        for p in points:
            writer.writerow([p.kernel, p.sq, p.batch_size, f"{p.time_us:.6f}"])


def _plot(points: List[Point], out_png: pathlib.Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 5.5))

    by_group: Dict[Tuple[str, int], List[Point]] = {}
    for p in points:
        by_group.setdefault((p.kernel, p.sq), []).append(p)

    style = {
        ("dense", 1): {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        ("dense", 2): {"color": "#1f77b4", "linestyle": ":", "marker": "s"},
        ("sparse", 1): {"color": "#d62728", "linestyle": "-", "marker": "o"},
        ("sparse", 2): {"color": "#d62728", "linestyle": ":", "marker": "s"},
    }

    for (kernel, sq), group_points in sorted(by_group.items()):
        group_points = sorted(group_points, key=lambda p: p.batch_size)
        xs = [p.batch_size for p in group_points]
        ys = [p.time_us for p in group_points]
        label = f"{kernel}, sq={sq}"
        plt.plot(xs, ys, label=label, linewidth=2, markersize=5, **style[(kernel, sq)])

    plt.title("Dense vs Sparse Decode Latency")
    plt.xlabel("Batch size")
    plt.ylabel("Time (us)")
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.75)
    plt.legend(handlelength=3.2, frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dense/sparse tests and plot latency curves")
    parser.add_argument(
        "--out-png",
        type=str,
        default="benchmark/dense_sparse_latency_us.png",
        help="Path to the output PNG figure",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="benchmark/dense_sparse_latency_us.csv",
        help="Path to the parsed CSV data",
    )
    args = parser.parse_args()

    dense_output = _run_test_script("tests/test_flash_mla_dense_decoding.py")
    sparse_output = _run_test_script("tests/test_flash_mla_sparse_decoding_bf16.py")

    points = _parse_dense_output(dense_output) + _parse_sparse_output(sparse_output)
    points = _filter_and_validate(points)

    out_png = pathlib.Path(args.out_png)
    out_csv = pathlib.Path(args.out_csv)
    _save_csv(points, out_csv)
    _plot(points, out_png)

    print(f"Saved parsed data to: {out_csv}")
    print(f"Saved figure to: {out_png}")


if __name__ == "__main__":
    main()
