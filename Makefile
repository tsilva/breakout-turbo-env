.PHONY: benchmark develop develop-release lint play release-prepare test test-python test-rust test-semantic-oracle test-stable-retro verify-semantic-oracle

PYTHON ?= .venv/bin/python
UV_CACHE_DIR ?= .uv-cache
PYTEST_ARGS ?=
STABLE_RETRO_REPO ?= $(abspath ../stable-retro-turbo)
TURBOBENCH ?= $(abspath ../turbobench/.venv/bin/turbobench)
ORACLE_OUTPUT ?=
ORACLE_RECEIPT ?=

develop:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(PYTHON) -m maturin develop

develop-release:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(PYTHON) -m maturin develop --release --locked

benchmark: develop-release
	$(PYTHON) -m breakout_turbo_env.benchmark

play: develop-release
	$(PYTHON) -m breakout_turbo_env.play

lint:
	$(PYTHON) -m ruff check .
	cargo fmt --check
	cargo clippy --locked --all-targets -- -D warnings

release-prepare:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv sync --locked --extra dev
	scripts/release.py prepare

test-rust:
	cargo test --locked --lib

test-python:
	$(PYTHON) -m pytest $(PYTEST_ARGS)

test-stable-retro: develop-release
	BREAKOUT_REQUIRE_STABLE_RETRO=1 \
	BREAKOUT_STABLE_RETRO_REPO="$(STABLE_RETRO_REPO)" \
	PYTHONPATH="$(CURDIR)/python:$(STABLE_RETRO_REPO)" \
	$(PYTHON) -m pytest -m stable_retro tests/test_stable_retro_parity.py $(PYTEST_ARGS)

test-semantic-oracle:
	@output="$(ORACLE_OUTPUT)"; \
	if [ -z "$$output" ]; then output="$$(mktemp -d)/breakout-semantic-oracle"; fi; \
	$(TURBOBENCH) oracle breakout/start-v2 \
		--left stable-retro@1.0.1 \
		--right breakout-turbo-env@checkout:$(CURDIR) \
		--output "$$output" \
		--allow-dirty; \
	echo "Semantic-oracle receipt: $$output"

verify-semantic-oracle:
	@test -n "$(ORACLE_RECEIPT)" || \
		(echo "Set ORACLE_RECEIPT to an external TurboBench receipt" >&2; exit 2)
	$(TURBOBENCH) verify-oracle "$(ORACLE_RECEIPT)" \
		--require-canonical \
		--require-provider breakout-turbo-env

test: test-rust test-python
