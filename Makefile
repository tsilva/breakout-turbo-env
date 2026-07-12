.PHONY: benchmark develop develop-release play release test test-python test-rust

PYTHON ?= .venv/bin/python
UV_CACHE_DIR ?= .uv-cache
PYTEST_ARGS ?=

develop:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(PYTHON) -m maturin develop

develop-release:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(PYTHON) -m maturin develop --release

benchmark: develop-release
	$(PYTHON) -m breakout_turbo_env.benchmark

play: develop-release
	$(PYTHON) -m breakout_turbo_env.play

release:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv sync --frozen --extra dev
	scripts/release.py

test-rust:
	cargo test --lib

test-python:
	$(PYTHON) -m pytest $(PYTEST_ARGS)

test: test-rust test-python
