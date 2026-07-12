#!/usr/bin/env python3
"""Unified training command for breakout-turbo-env algorithms."""

from __future__ import annotations

import argparse
import sys

_ALGORITHMS = ("jerk",)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an algorithm on breakout-turbo-env",
        usage="%(prog)s <algo> [algorithm options]",
    )
    parser.add_argument("algo", choices=_ALGORITHMS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        _parser().print_help()
        return 0
    selected = _parser().parse_args(args[:1])
    if selected.algo == "jerk":
        from train_jerk import main as train_jerk

        return train_jerk(args[1:])
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
