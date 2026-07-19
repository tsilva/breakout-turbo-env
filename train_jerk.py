#!/usr/bin/env python3
"""Train a JERK action tape for deterministic breakout-turbo-env.

JERK (Just Enough Retained Knowledge) keeps the best action sequence found so
far.  Each new candidate starts from a regular environment reset, replays a
prefix of that sequence, and then explores.  No emulator states or snapshots
are stored: the resulting JSON file contains only metadata and actions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from breakout_turbo_env import FIXED_POINT_ONE, BreakoutVecEnv


@dataclass(frozen=True)
class Candidate:
    actions: list[int]
    score: int
    reward: float
    lives: int
    solved: bool

    @property
    def rank(self) -> tuple[bool, int, float, int, int]:
        return self.solved, self.score, self.reward, self.lives, len(self.actions)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a snapshot-free JERK action tape for breakout-turbo-env."
    )
    parser.add_argument("--layout", choices=("full", "checker", "tunnel", "sparse"), default="full")
    parser.add_argument("--population", type=int, default=64, help="candidate action tapes per generation")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=30_000, help="maximum actions in one tape")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--exploration", type=float, default=0.08, help="probability of a random action")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="artifact root; policies are saved under <root>/jerk/<timestamp>",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.population <= 0 or args.generations <= 0 or args.max_steps <= 0 or args.frame_skip <= 0:
        raise ValueError("population, generations, max-steps, and frame-skip must be positive")
    if not 0.0 <= args.exploration <= 1.0:
        raise ValueError("exploration must be in [0, 1]")


def _reset(env: BreakoutVecEnv, layout: str) -> dict[str, np.ndarray]:
    starts = np.full(env.num_envs, layout, dtype=object)
    _, infos = env.reset(options={"start_ids": starts})
    return infos


def _prefix_lengths(
    champion: list[int], population: int, rng: np.random.Generator
) -> np.ndarray:
    """Choose retained prefixes, biased toward the end of the champion tape."""
    if not champion:
        return np.zeros(population, dtype=np.int64)
    lengths = (len(champion) * np.sqrt(rng.random(population))).astype(np.int64)
    lengths[0] = len(champion)  # Always measure an exact replay of the incumbent.
    return lengths


def train_generation(
    env: BreakoutVecEnv,
    *,
    layout: str,
    champion: list[int],
    max_steps: int,
    exploration: float,
    rng: np.random.Generator,
) -> Candidate:
    """Reset, replay retained prefixes, explore, and return the best action tape."""
    population = env.num_envs
    infos = _reset(env, layout)
    prefixes = _prefix_lengths(champion, population, rng)
    histories = np.empty((population, max_steps), dtype=np.uint8)
    rewards = np.zeros(population, dtype=np.float64)
    active = np.ones(population, dtype=np.bool_)
    finished = np.zeros(population, dtype=np.bool_)
    end_steps = np.full(population, max_steps, dtype=np.int64)
    final_scores = np.zeros(population, dtype=np.int64)
    final_lives = np.full(population, 5, dtype=np.int64)
    solved = np.zeros(population, dtype=np.bool_)

    # Different fixed offsets create different paddle bounce angles. Candidate
    # zero is the deterministic, zero-noise incumbent/baseline.
    offsets = rng.integers(-7, 8, size=population, dtype=np.int64) * FIXED_POINT_ONE
    offsets[0] = 0

    for step in range(max_steps):
        paddle_center = infos["paddle_x"] + 8 * FIXED_POINT_ONE
        target = infos["ball_x"] + offsets
        actions = np.where(
            target < paddle_center - FIXED_POINT_ONE,
            3,
            np.where(target > paddle_center + FIXED_POINT_ONE, 2, 0),
        ).astype(np.uint8)
        actions[infos["awaiting_fire"].astype(bool)] = 1

        replaying = step < prefixes
        if champion and replaying.any():
            actions[replaying] = champion[step]

        exploring = active & ~replaying & (rng.random(population) < exploration)
        actions[exploring] = rng.integers(0, 4, size=int(exploring.sum()), dtype=np.uint8)
        # Candidate zero is a stable reference and exact champion replay.
        if step >= prefixes[0]:
            if target[0] < paddle_center[0] - FIXED_POINT_ONE:
                actions[0] = 3
            elif target[0] > paddle_center[0] + FIXED_POINT_ONE:
                actions[0] = 2
            else:
                actions[0] = 0
            if infos["awaiting_fire"][0]:
                actions[0] = 1

        actions[~active] = 0
        histories[:, step] = actions
        _, step_rewards, terminated, _, infos = env.step(actions)
        rewards[active] += step_rewards[active]

        newly_solved = active & (infos["bricks_remaining"] == 0)
        completed = active & (terminated | newly_solved)
        if completed.any():
            final_scores[completed] = infos["score"][completed]
            final_lives[completed] = infos["lives"][completed]
            solved[newly_solved] = True
            end_steps[completed] = step + 1
            active[completed] = False
            finished[completed] = True

        if not active.any():
            break

        # Manual-reset environments reject another step after termination.
        # Reset completed lanes only to keep the vector batch step-able; their
        # later transitions are ignored and never enter their memorized tape.
        if terminated.any():
            reset_starts = np.full(population, layout, dtype=object)
            _, reset_infos = env.reset(
                options={"reset_mask": terminated, "start_ids": reset_starts}
            )
            for key, values in reset_infos.items():
                if key in infos:
                    infos[key][terminated] = values[terminated]

        if solved.any():
            break

    unfinished = ~finished
    final_scores[unfinished] = infos["score"][unfinished]
    final_lives[unfinished] = infos["lives"][unfinished]
    end_steps[unfinished] = min(step + 1, max_steps)

    # Lexicographic ranking: clearing the board, bricks cleared, return, lives,
    # then survival/tape length. This mirrors JERK's best-trajectory retention.
    ranks = [
        (bool(solved[i]), int(final_scores[i]), float(rewards[i]), int(final_lives[i]), int(end_steps[i]))
        for i in range(population)
    ]
    winner = max(range(population), key=ranks.__getitem__)
    length = int(end_steps[winner])
    return Candidate(
        actions=histories[winner, :length].astype(int).tolist(),
        score=int(final_scores[winner]),
        reward=float(rewards[winner]),
        lives=int(final_lives[winner]),
        solved=bool(solved[winner]),
    )


def verify(actions: list[int], *, layout: str, frame_skip: int) -> Candidate:
    """Replay an action tape from reset and report its deterministic result."""
    env = BreakoutVecEnv(
        num_envs=1,
        num_threads=1,
        obs_resize=(8, 8),
        frame_stack=1,
        frame_skip=frame_skip,
        info_filter="all",
    )
    infos = _reset(env, layout)
    total_reward = 0.0
    terminated = np.array([False])
    used = 0
    try:
        for used, action in enumerate(actions, start=1):
            _, reward, terminated, _, infos = env.step(np.array([action], dtype=np.uint8))
            total_reward += float(reward[0])
            if terminated[0]:
                break
    finally:
        env.close()
    return Candidate(
        actions=actions[:used],
        score=int(infos["score"][0]),
        reward=total_reward,
        lives=int(infos["lives"][0]),
        solved=bool(terminated[0] and infos["bricks_remaining"][0] == 0),
    )


def save_policy(path: Path, candidate: Candidate, *, layout: str, frame_skip: int, seed: int) -> None:
    payload = {
        "algorithm": "JERK (Just Enough Retained Knowledge)",
        "format_version": 1,
        "layout": layout,
        "frame_skip": frame_skip,
        "seed": seed,
        "score": candidate.score,
        "reward": candidate.reward,
        "lives": candidate.lives,
        "solved": candidate.solved,
        "actions": candidate.actions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def create_run_dir(runs_dir: Path, *, algo: str = "jerk") -> Path:
    """Create runs/<algo>/<local timestamp>, avoiding same-second collisions."""
    algorithm_dir = runs_dir / algo
    algorithm_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    for suffix in range(1_000):
        name = timestamp if suffix == 0 else f"{timestamp}-{suffix:03d}"
        run_dir = algorithm_dir / name
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir
    raise RuntimeError(f"could not allocate a timestamped run directory under {algorithm_dir}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    run_dir = create_run_dir(args.runs_dir)
    rng = np.random.default_rng(args.seed)
    threads = args.threads if args.threads is not None else min(args.population, 8)
    env = BreakoutVecEnv(
        num_envs=args.population,
        num_threads=threads,
        obs_resize=(8, 8),
        frame_stack=1,
        frame_skip=args.frame_skip,
        info_filter="all",
    )
    champion: Candidate | None = None
    try:
        for generation in range(1, args.generations + 1):
            candidate = train_generation(
                env,
                layout=args.layout,
                champion=[] if champion is None else champion.actions,
                max_steps=args.max_steps,
                exploration=args.exploration,
                rng=rng,
            )
            if champion is None or candidate.rank > champion.rank:
                champion = candidate
            assert champion is not None
            print(
                f"generation={generation} score={champion.score} reward={champion.reward:.1f} "
                f"lives={champion.lives} steps={len(champion.actions)} solved={champion.solved}"
            )
            if champion.solved:
                break
    finally:
        env.close()

    if champion is None:
        raise RuntimeError("training produced no action tape")
    replay = verify(champion.actions, layout=args.layout, frame_skip=args.frame_skip)
    if replay.rank != champion.rank:
        raise RuntimeError(f"deterministic replay mismatch: trained={champion.rank}, replayed={replay.rank}")
    policy_path = run_dir / "policy.json"
    save_policy(policy_path, replay, layout=args.layout, frame_skip=args.frame_skip, seed=args.seed)
    print(f"saved={policy_path} verified={replay.solved}")
    return 0 if replay.solved else 2


if __name__ == "__main__":
    raise SystemExit(main())
