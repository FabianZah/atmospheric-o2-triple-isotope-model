"""Stable command-line entry point for the publication model repository."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def _terminate(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run(arguments: list[str]) -> int:
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(arguments, cwd=ROOT, creationflags=creation_flags)
    try:
        return process.wait()
    except KeyboardInterrupt:
        _terminate(process)
        return 130


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the atmospheric O2 triple-isotope publication model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    api = subparsers.add_parser(
        "api", help="Start the public model API and browser interface."
    )
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)

    subparsers.add_parser(
        "smoke", help="Run the fast publication-package smoke test."
    )
    subparsers.add_parser(
        "acceptance", help="Run the integrated scientific acceptance audit."
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "api":
        return _run(
            [
                sys.executable,
                str(ROOT / "code" / "web_api.py"),
                "--host",
                args.host,
                "--port",
                str(args.port),
            ]
        )
    if args.command == "smoke":
        return _run(
            [sys.executable, str(ROOT / "validation" / "smoke_publication_package.py")]
        )
    return _run(
        [
            sys.executable,
            str(ROOT / "validation" / "audit_publication_model_acceptance.py"),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
