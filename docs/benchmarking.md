# Benchmarking

The built-in benchmark measures the complete policy-facing path: fixed-point
physics, frame skipping, native rasterization, grayscale area resizing, and
four-frame CHW stacking.

## Reproduce a run

Install the release build and run from the repository root:

```bash
make develop-release
uv run breakout-turbo-env benchmark --steps 30000 --warmup 1000 --repeats 5 --threads 8
```

The fixed workload uses 16 environments, 84×84 grayscale observations, frame
skip 4, frame stack 4, safe-view output buffers, no info collection, and a
repeating action batch. Output includes batch steps, environment steps,
emulated frames, and observation-buffer throughput.

## Reporting results

Always report:

- breakout-turbo-env version and commit;
- processor, core count, memory, operating system, and architecture;
- Python version and release-build status;
- environment and thread count;
- warmup, timed steps, repeats, and the median result; and
- whether power management or other workloads could affect the run.

Do not compare a policy-facing result with a physics-only result. Comparisons
with ALE, Stable Retro, EnvPool, or another environment must use equivalent
frame skip, preprocessing, observation ownership, information collection, and
reset behavior. Include the exact script and configuration needed to reproduce
both sides.

For the closest live cartridge comparison, configure a lawful Breakout ROM in a
sibling Stable Retro checkout and run:

```bash
uv run python scripts/benchmark_comparison.py --steps 30000 --warmup 1000 --repeats 5 --threads 8
```

The comparison uses identical native actions and matched vector count,
threading, frame skip, resize, frame stack, info filtering, max-pooling, and
observation-buffer ownership. It reports raw runs, medians, and the median
speedup; pass `--json` to preserve machine-readable evidence.

## Reference community-preview run

Release-specific benchmark results live in GitHub release notes. They are
evidence for a particular machine and workload, not a universal performance
guarantee. The command above remains the source of truth for local performance.
