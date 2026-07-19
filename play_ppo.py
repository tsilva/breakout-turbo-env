#!/usr/bin/env python3
"""Visually replay a PPO policy produced by train_ppo.py."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from breakout_turbo_env import BreakoutVecEnv
from breakout_turbo_env.play import (
    _DEFAULT_PLAY_SCALE,
    _hud_text,
    _print_episode_stats,
    _scaled_frame_size,
)
from train_ppo import _FEATURES, features


@dataclass(frozen=True)
class PpoPolicy:
    layout: str
    frame_skip: int
    weights: dict[str, np.ndarray]

    def action(self, info: dict[str, np.ndarray]) -> np.ndarray:
        values = features(info)
        hidden = np.tanh(values @ self.weights["trunk__0__weight"].T + self.weights["trunk__0__bias"])
        hidden = np.tanh(hidden @ self.weights["trunk__2__weight"].T + self.weights["trunk__2__bias"])
        logits = hidden @ self.weights["actor__weight"].T + self.weights["actor__bias"]
        return np.argmax(logits, axis=1).astype(np.uint8)


def load_policy(path: Path) -> PpoPolicy:
    try:
        artifact = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read PPO policy {path}: {exc}") from exc
    with artifact:
        try:
            metadata = json.loads(str(artifact["metadata_json"]))
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError("PPO policy has invalid metadata") from exc
        required = ("trunk__0__weight", "trunk__0__bias", "trunk__2__weight", "trunk__2__bias", "actor__weight", "actor__bias")
        if metadata.get("algorithm") != "PPO" or metadata.get("format_version") != 1:
            raise ValueError("policy is not a supported PPO artifact")
        if metadata.get("layout") not in ("full", "checker", "tunnel", "sparse"):
            raise ValueError("PPO policy has an invalid layout")
        if not isinstance(metadata.get("frame_skip"), int) or metadata["frame_skip"] <= 0:
            raise ValueError("PPO policy has an invalid frame_skip")
        try:
            weights = {name: artifact[name].astype(np.float32, copy=True) for name in required}
        except KeyError as exc:
            raise ValueError(f"PPO policy is missing {exc.args[0]}") from exc
    if weights["trunk__0__weight"].shape[1] != _FEATURES or weights["actor__weight"].shape[0] != 3:
        raise ValueError("PPO policy has incompatible network dimensions")
    return PpoPolicy(layout=metadata["layout"], frame_skip=metadata["frame_skip"], weights=weights)


def latest_policy(runs_dir: Path) -> Path:
    candidates = list((runs_dir / "ppo").glob("*/policy.npz"))
    if not candidates:
        raise ValueError("no PPO policies found; run `uv run python train.py ppo` first or pass --policy")
    return max(candidates, key=lambda path: (path.parent.name, path.stat().st_mtime_ns))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a trained PPO policy")
    parser.add_argument("--policy", type=Path, help="PPO .npz policy; defaults to the newest PPO run")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--scale",
        type=int,
        default=_DEFAULT_PLAY_SCALE,
        help=f"integer window scale (default: {_DEFAULT_PLAY_SCALE})",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser


def run(policy: PpoPolicy, *, scale: int, fps: int, loop: bool, max_frames: int = 0) -> None:
    if scale <= 0 or fps <= 0 or max_frames < 0:
        raise ValueError("scale and fps must be positive; max-frames must be non-negative")
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit(
            "policy playback requires the play extra; "
            "install `breakout-turbo-env[play]`"
        ) from exc

    pygame.init()
    env = BreakoutVecEnv(num_envs=1, num_threads=1, frame_skip=policy.frame_skip, frame_stack=1, info_filter="all")
    reset_mask = np.ones(1, dtype=np.bool_)
    starts = np.array([policy.layout], dtype=object)
    _, info = env.reset(options={"reset_mask": reset_mask, "start_ids": starts})
    height, width = env.render().shape[:2]
    game_size = _scaled_frame_size(width, height, scale)
    screen = pygame.display.set_mode(game_size)
    clock = pygame.time.Clock()
    episode = 1
    episode_return = 0.0
    episode_steps = 0
    episode_started = time.perf_counter()
    displayed = 0
    paused = False
    running = True

    def restart() -> None:
        nonlocal info, episode_return, episode_steps, episode_started
        _, info = env.reset(options={"reset_mask": reset_mask, "start_ids": starts})
        episode_return = 0.0
        episode_steps = 0
        episode_started = time.perf_counter()

    try:
        while running:
            restart_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_SPACE, pygame.K_r):
                        restart_requested = True
                    elif event.key == pygame.K_p:
                        paused = not paused
            if restart_requested:
                restart()

            finished_after_render = False
            if running and not paused:
                _, reward, terminated, _, info = env.step(policy.action(info))
                episode_return += float(reward[0])
                episode_steps += 1
                if terminated[0]:
                    _print_episode_stats(info, episode=episode, layout=policy.layout, episode_return=episode_return, display_steps=episode_steps, elapsed=time.perf_counter() - episode_started)
                    if loop:
                        episode += 1
                        restart()
                    else:
                        finished_after_render = True

            frame = env.render()
            surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            if scale != 1:
                surface = pygame.transform.scale(surface, game_size)
            text = _hud_text(info, paused=paused) + f"    PPO {episode_steps:05d}"
            screen.blit(surface, (0, 0))
            pygame.display.set_caption(
                f"Breakout Turbo | PPO replay | {policy.layout} | {text}"
            )
            pygame.display.flip()
            clock.tick(fps)
            displayed += 1
            if finished_after_render or (max_frames and displayed >= max_frames):
                running = False
    finally:
        env.close()
        pygame.quit()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        path = args.policy if args.policy is not None else latest_policy(args.runs_dir)
        print(f"policy={path}", flush=True)
        run(load_policy(path), scale=args.scale, fps=args.fps, loop=args.loop, max_frames=args.max_frames)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
