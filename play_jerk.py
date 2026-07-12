#!/usr/bin/env python3
"""Visually replay a JERK action tape produced by train_jerk.py."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from breakout_turbo_env import BreakoutVecEnv
from breakout_turbo_env.play import _hud_text, _print_episode_stats

_LAYOUTS = ("full", "checker", "tunnel", "sparse")


@dataclass(frozen=True)
class JerkPolicy:
    actions: tuple[int, ...]
    layout: str
    frame_skip: int


def load_policy(path: Path) -> JerkPolicy:
    """Load and validate an action-only JERK policy artifact."""
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"policy file does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JERK policy {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("policy must be a JSON object")
    layout = payload.get("layout")
    if layout not in _LAYOUTS:
        raise ValueError(f"policy layout must be one of {_LAYOUTS}")
    frame_skip = payload.get("frame_skip")
    if not isinstance(frame_skip, int) or isinstance(frame_skip, bool) or frame_skip <= 0:
        raise ValueError("policy frame_skip must be a positive integer")
    actions = payload.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("policy actions must be a non-empty list")
    if any(not isinstance(action, int) or isinstance(action, bool) or action not in (0, 1, 2) for action in actions):
        raise ValueError("every policy action must be 0 (noop), 1 (left), or 2 (right)")
    return JerkPolicy(actions=tuple(actions), layout=layout, frame_skip=frame_skip)


def latest_policy(runs_dir: Path, *, algo: str = "jerk") -> Path:
    """Return the policy in the lexically newest timestamped algorithm run."""
    candidates = list((runs_dir / algo).glob("*/policy.json"))
    if not candidates:
        raise ValueError(
            f"no {algo} policies found under {runs_dir / algo}; "
            f"run `uv run python train.py {algo}` first or pass --policy"
        )
    return max(candidates, key=lambda path: (path.parent.name, path.stat().st_mtime_ns))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a trained JERK action tape")
    parser.add_argument("--policy", type=Path, help="policy JSON; defaults to the latest JERK run")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="artifact root")
    parser.add_argument("--scale", type=int, default=8, help="integer window scale")
    parser.add_argument("--fps", type=int, default=60, help="action-tape steps per second")
    parser.add_argument("--loop", action="store_true", help="restart the tape after it ends")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="exit after this many displayed frames; 0 plays the complete tape",
    )
    return parser


def run(policy: JerkPolicy, *, scale: int, fps: int, loop: bool, max_frames: int = 0) -> None:
    if scale <= 0 or fps <= 0 or max_frames < 0:
        raise ValueError("scale and fps must be positive; max-frames must be non-negative")

    import pygame

    pygame.init()
    env = BreakoutVecEnv(
        num_envs=1,
        num_threads=1,
        frame_skip=policy.frame_skip,
        frame_stack=1,
        info_filter="all",
    )
    reset_mask = np.ones(1, dtype=np.bool_)
    start_ids = np.array([policy.layout], dtype=object)
    _, info = env.reset(options={"reset_mask": reset_mask, "start_ids": start_ids})
    raw_height, raw_width = env.render().shape[:2]
    game_size = (raw_width * scale, raw_height * scale)
    hud_height = max(28, scale * 4)
    screen = pygame.display.set_mode((game_size[0], game_size[1] + hud_height))
    hud_font = pygame.font.Font(None, max(18, scale * 3))
    clock = pygame.time.Clock()
    action_index = 0
    episode = 1
    episode_return = 0.0
    episode_started = time.perf_counter()
    displayed_frames = 0
    paused = False
    running = True

    def restart() -> None:
        nonlocal action_index, episode_return, episode_started, info
        _, info = env.reset(options={"reset_mask": reset_mask, "start_ids": start_ids})
        action_index = 0
        episode_return = 0.0
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
                if action_index >= len(policy.actions):
                    print(
                        f"tape_end episode={episode} outcome=exhausted actions={action_index} "
                        f"score={int(info['score'][0])}",
                        flush=True,
                    )
                    if loop:
                        episode += 1
                        restart()
                    else:
                        finished_after_render = True
                else:
                    action = policy.actions[action_index]
                    _, reward, terminated, _, info = env.step(
                        np.array([action], dtype=np.uint8)
                    )
                    action_index += 1
                    episode_return += float(reward[0])
                    if terminated[0]:
                        _print_episode_stats(
                            info,
                            episode=episode,
                            layout=policy.layout,
                            episode_return=episode_return,
                            display_steps=action_index,
                            elapsed=time.perf_counter() - episode_started,
                        )
                        if loop:
                            episode += 1
                            restart()
                        else:
                            finished_after_render = True

            frame = env.render()
            surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            if scale != 1:
                surface = pygame.transform.scale(surface, game_size)
            screen.fill((16, 18, 24))
            status = _hud_text(info, paused=paused)
            status += f"    JERK {action_index:05d}/{len(policy.actions):05d}"
            hud_surface = hud_font.render(status, True, (245, 245, 245))
            screen.blit(hud_surface, (10, (hud_height - hud_surface.get_height()) // 2))
            screen.blit(surface, (0, hud_height))
            pygame.display.set_caption(
                f"Breakout Turbo | JERK replay | {policy.layout} | "
                f"action {action_index}/{len(policy.actions)}"
            )
            pygame.display.flip()
            clock.tick(fps)
            displayed_frames += 1
            if finished_after_render or (max_frames and displayed_frames >= max_frames):
                running = False
    finally:
        env.close()
        pygame.quit()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy_path = args.policy if args.policy is not None else latest_policy(args.runs_dir)
        policy = load_policy(policy_path)
        print(f"policy={policy_path}", flush=True)
        run(policy, scale=args.scale, fps=args.fps, loop=args.loop, max_frames=args.max_frames)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
