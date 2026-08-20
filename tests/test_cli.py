from __future__ import annotations

from env_breakoutatari2600_turbo_native.cli import build_parser, main


def test_cli_help_and_commands():
    parser = build_parser()

    assert parser.parse_args([]).command is None
    assert parser.parse_args(["play"]).command == "play"
    assert parser.parse_args(["benchmark"]).command == "benchmark"
    assert "train" not in parser.format_help()


def test_cli_dispatches_subcommands(monkeypatch):
    calls = []

    def fake_play(argv, *, prog):
        calls.append(("play", list(argv), prog))

    monkeypatch.setattr("env_breakoutatari2600_turbo_native.play.main", fake_play)
    main(["play", "--show-obs"])
    assert calls == [("play", ["--show-obs"], "env-breakoutatari2600-turbo-native play")]
