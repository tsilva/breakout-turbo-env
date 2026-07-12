#!/usr/bin/env python3
"""Unified policy playback command for breakout-turbo-env algorithms."""

from __future__ import annotations

import argparse
import sys

_ALGORITHMS = ("jerk",)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play a trained breakout-turbo-env policy",
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
        from play_jerk import main as play_jerk

        return play_jerk(args[1:])
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
