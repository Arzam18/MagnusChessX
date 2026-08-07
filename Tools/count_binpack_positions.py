#!/usr/bin/env python3
"""Validate Stockfish .binpack files and count their decoded positions.

The script builds a tiny helper against the same sfbinpack version used by the
MagnusChessX trainers.  A file is reported as usable only when that reader
decodes the complete non-empty file.  Input data is never modified.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT = Path(r"D:\NNUE\data\Stockfish-tools\generated")
TOOLS_DIRECTORY = Path(__file__).resolve().parent
HELPER_DIRECTORY = TOOLS_DIRECTORY / "binpack_counter"
HELPER_MANIFEST = HELPER_DIRECTORY / "Cargo.toml"
HELPER_NAME = "binpack_counter.exe" if os.name == "nt" else "binpack_counter"
HELPER_PATH = HELPER_DIRECTORY / "target" / "release" / HELPER_NAME


@dataclass(frozen=True)
class Result:
    path: Path
    size: int
    positions: int | None
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate .binpack files and count decoded positions."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"directory to scan (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="scan subdirectories recursively",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="number of validator processes to run concurrently (default: up to 4)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="optionally write the per-file results to a CSV file",
    )
    return parser.parse_args()


def build_helper() -> Path:
    command = [
        "cargo",
        "build",
        "--release",
        "--manifest-path",
        str(HELPER_MANIFEST),
    ]
    if (HELPER_DIRECTORY / "Cargo.lock").is_file():
        command.insert(3, "--locked")
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or not HELPER_PATH.is_file():
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"failed to build binpack counter:\n{detail}")
    return HELPER_PATH


def validate_one(helper: Path, path: Path) -> Result:
    try:
        completed = subprocess.run(
            [str(helper), str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as error:
        return Result(path, path.stat().st_size, None, "failed", str(error))

    output = completed.stdout.strip()
    if completed.returncode == 0 and output.isdecimal() and int(output) > 0:
        return Result(path, path.stat().st_size, int(output), "usable", "")

    detail = completed.stderr.strip() or output or f"validator exit code {completed.returncode}"
    return Result(path, path.stat().st_size, None, "invalid", detail.splitlines()[-1])


def format_integer(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"


def print_results(results: list[Result]) -> None:
    name_width = max(len("file"), *(len(result.path.name) for result in results))
    print(f"{'file':<{name_width}}  {'MiB':>10}  {'positions':>15}  status")
    print(f"{'-' * name_width}  {'-' * 10}  {'-' * 15}  {'-' * 10}")
    for result in results:
        mib = result.size / (1024 * 1024)
        print(
            f"{result.path.name:<{name_width}}  {mib:10.2f}  "
            f"{format_integer(result.positions):>15}  {result.status}"
        )
        if result.detail:
            print(f"{'':<{name_width}}  detail: {result.detail}")

    usable = [result for result in results if result.status == "usable"]
    print()
    print(f"usable files : {len(usable)}/{len(results)}")
    print(f"usable bytes : {sum(result.size for result in usable):,}")
    print(f"total pos    : {sum(result.positions or 0 for result in usable):,}")


def write_csv(path: Path, results: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["path", "bytes", "positions", "status", "detail"])
        for result in results:
            writer.writerow(
                [result.path, result.size, result.positions, result.status, result.detail]
            )


def main() -> int:
    args = parse_args()
    directory = args.directory.resolve()

    if args.jobs < 1:
        print("error: --jobs must be at least 1", file=sys.stderr)
        return 2
    if not directory.is_dir():
        print(f"error: input directory does not exist: {directory}", file=sys.stderr)
        return 2
    iterator = directory.rglob("*.binpack") if args.recursive else directory.glob("*.binpack")
    files = sorted((path.resolve() for path in iterator if path.is_file()), key=str)
    if not files:
        print(f"error: no .binpack files found in {directory}", file=sys.stderr)
        return 2

    try:
        helper = build_helper()
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    workers = min(args.jobs, len(files))
    print(f"validating {len(files)} file(s) with {workers} process(es)...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda path: validate_one(helper, path), files))

    print_results(results)
    if args.csv:
        write_csv(args.csv.resolve(), results)
        print(f"CSV          : {args.csv.resolve()}")

    return 0 if all(result.status == "usable" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
