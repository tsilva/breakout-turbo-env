from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="breakout-turbo-env",
        description="Deterministic high-throughput Breakout environment tools",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("play", "benchmark"),
        help="command to run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not args:
        parser.print_help()
        return
    if args[0] == "play":
        from .play import main as play_main

        play_main(args[1:], prog="breakout-turbo-env play")
        return
    if args[0] == "benchmark":
        from .benchmark import main as benchmark_main

        benchmark_main(args[1:], prog="breakout-turbo-env benchmark")
        return
    parser.parse_args(args)


if __name__ == "__main__":
    main()
