#!/usr/bin/env python3
"""Differential probe for Stable Retro Breakout and breakout-turbo-env.

This is an optional developer tool.  It deliberately does not make the ROM or
Stable Retro a package dependency.  Run it with the Python environment from a
local stable-retro-turbo checkout; the probe discovers the reference ball-x
RAM variable from rendered motion instead of relying on a recorded trace.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


RED = np.array([200, 72, 72], dtype=np.uint8)
GAME = "Breakout-Atari2600-v0"


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class Transition:
    frame: int
    stable: Point
    turbo: Point
    stable_delta: Point
    turbo_delta: Point
    stable_reward: float
    turbo_reward: float


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def find_ball(frame: np.ndarray, *, min_y: int = 32, max_y: int = 188) -> Point | None:
    """Find an isolated 2x4 Atari-red sprite outside the brick bands."""
    red = np.all(frame == RED, axis=2)
    for y in range(min_y, max_y - 3):
        starts = _runs(red[y])
        for x0, x1 in starts:
            if x1 - x0 != 2:
                continue
            if np.all(red[y : y + 4, x0:x1]) and not np.any(red[max(0, y - 1), x0:x1]):
                if y + 4 == red.shape[0] or not np.any(red[y + 4, x0:x1]):
                    return Point(int(x0), int(y))
    return None


def find_paddle_x(frame: np.ndarray) -> int:
    """Return the left edge of the active 16x4 red paddle sprite."""
    red = np.all(frame[190] == RED, axis=1)
    candidates = [(x0, x1) for x0, x1 in _runs(red) if x0 < 152 and x1 - x0 >= 16]
    if not candidates:
        raise RuntimeError("could not locate the Stable Retro paddle")
    x0, x1 = max(candidates, key=lambda run: min(run[1], 152) - run[0])
    return min(int(x0), 144 if x1 >= 152 else int(x0))


def ram_index_to_address(memory: Any, index: int) -> int:
    cursor = 0
    for address in sorted(memory.blocks):
        size = len(memory.blocks[address])
        if cursor <= index < cursor + size:
            return int(address) + index - cursor
        cursor += size
    raise IndexError(f"RAM index {index} is outside registered memory blocks")


def discover_coordinates(env: Any, *, samples_needed: int = 100) -> tuple[int, int, int, int, int]:
    """Discover ball/paddle RAM addresses and screen offsets from live motion."""
    frame, _ = env.reset()
    action = np.zeros(env.action_space.shape, dtype=np.int8)
    action[0] = 1  # fire
    observations: list[tuple[np.ndarray, Point, int, int]] = []
    for step in range(1200):
        frame, _, terminated, _, info = env.step(action)
        action[0] = 0
        if terminated:
            raise RuntimeError("reference game ended while discovering ball coordinates")
        ball_y = int(info["ball_y"])
        point = find_ball(frame, min_y=96, max_y=180)
        if point is not None and ball_y > 0:
            observations.append((env.get_ram().copy(), point, ball_y, find_paddle_x(frame)))
            if len(observations) >= samples_needed:
                break
        # Keep the paddle under the ball so coordinate discovery survives a
        # full downward flight on installations with a different analog seed.
        if point is not None:
            paddle_x = find_paddle_x(frame)
            action = np.zeros(env.action_space.shape, dtype=np.int8)
            if point.x < paddle_x + 6:
                action[6] = 1
            elif point.x > paddle_x + 10:
                action[7] = 1
    if len(observations) < samples_needed:
        raise RuntimeError("could not collect enough isolated ball frames")

    ram = np.stack([row[0] for row in observations]).astype(np.int16)
    screen_x = np.asarray([row[1].x for row in observations], dtype=np.int16)
    candidates: list[tuple[int, int]] = []
    for index in range(ram.shape[1]):
        offsets = ram[:, index] - screen_x
        if np.all(offsets == offsets[0]) and np.unique(ram[:, index]).size >= 8:
            candidates.append((index, int(offsets[0])))
    if len(candidates) != 1:
        raise RuntimeError(f"ball-x discovery was ambiguous: {candidates}")
    index, x_offset = candidates[0]
    y_offsets = np.asarray([row[2] - row[1].y for row in observations])
    if not np.all(y_offsets == y_offsets[0]):
        raise RuntimeError("ball-y screen offset changed during coordinate discovery")
    paddle_x = np.asarray([row[3] for row in observations], dtype=np.int16)
    paddle_candidates: list[tuple[int, int]] = []
    for paddle_index in range(ram.shape[1]):
        offsets = ram[:, paddle_index] - paddle_x
        if np.all(offsets == offsets[0]) and np.unique(ram[:, paddle_index]).size >= 8:
            paddle_candidates.append((paddle_index, int(offsets[0])))
    if len(paddle_candidates) != 1:
        raise RuntimeError(f"paddle-x discovery was ambiguous: {paddle_candidates}")
    paddle_index, paddle_offset = paddle_candidates[0]
    return (
        ram_index_to_address(env.data.memory, index),
        x_offset,
        int(y_offsets[0]),
        ram_index_to_address(env.data.memory, paddle_index),
        paddle_offset,
    )


class StableReference:
    def __init__(self, data_dir: Path):
        import stable_retro as retro

        self._retro = retro
        self._data_dir = data_dir
        self._base_info = data_dir / "data.json"
        self._scenario = data_dir / "scenario.json"
        discovery = retro.make(
            GAME,
            state="Start",
            info=str(self._base_info),
            scenario=str(self._scenario),
            inttype=retro.data.Integrations.ALL,
            render_mode="rgb_array",
        )
        try:
            address, self.x_offset, self.y_offset, paddle_address, self.paddle_offset = discover_coordinates(discovery)
        finally:
            discovery.close()

        data = json.loads(self._base_info.read_text())
        data.setdefault("info", {})["ball_x_probe"] = {"address": address, "type": "|u1"}
        data["info"]["paddle_x_probe"] = {"address": paddle_address, "type": "|u1"}
        self._temporary = tempfile.TemporaryDirectory(prefix="breakout-parity-")
        info_path = Path(self._temporary.name) / "data.json"
        info_path.write_text(json.dumps(data))
        self.env = retro.make(
            GAME,
            state="Start",
            info=str(info_path),
            scenario=str(self._scenario),
            inttype=retro.data.Integrations.ALL,
            render_mode="rgb_array",
        )

    def close(self) -> None:
        self.env.close()
        self._temporary.cleanup()

    def point(self) -> Point:
        return Point(
            int(self.env.data.lookup_value("ball_x_probe")) - self.x_offset,
            int(self.env.data.lookup_value("ball_y")) - self.y_offset,
        )

    def action(self, direction: int = 0, *, fire: bool = False) -> np.ndarray:
        value = np.zeros(self.env.action_space.shape, dtype=np.int8)
        if fire:
            value[0] = 1
        if direction < 0:
            value[6] = 1
        elif direction > 0:
            value[7] = 1
        return value

    def find_flight(self, dx_sign: int, dy_sign: int) -> bytes:
        self.env.reset()
        action = self.action(fire=True)
        previous: Point | None = None
        for _ in range(5000):
            frame, _, terminated, _, info = self.env.step(action)
            action = self.action()
            if terminated:
                self.env.reset()
                action = self.action(fire=True)
                previous = None
                continue
            point = self.point()
            if previous is not None:
                dx, dy = point.x - previous.x, point.y - previous.y
                if np.sign(dx) == dx_sign and np.sign(dy) == dy_sign and 100 <= point.y <= 160:
                    return bytes(self.env.em.get_state())
            previous = point
            try:
                paddle_x = find_paddle_x(frame)
            except RuntimeError:
                continue
            action = self.action(-1 if point.x < paddle_x + 6 else 1 if point.x > paddle_x + 10 else 0)
        raise RuntimeError(f"could not find flight direction ({dx_sign}, {dy_sign})")

    def force(self, state: bytes, point: Point) -> None:
        self.env.em.set_state(state)
        self.env.data.update_ram()
        self.env.set_value("ball_x_probe", point.x + self.x_offset)
        self.env.set_value("ball_y", point.y + self.y_offset)
        self.env.data.update_ram()

    def force_paddle(self, x: int) -> None:
        self.env.set_value("paddle_x_probe", x + self.paddle_offset)
        self.env.data.update_ram()


def turbo_ball_point(frame: np.ndarray) -> Point:
    point = find_ball(frame, min_y=31, max_y=188)
    if point is None:
        raise RuntimeError("could not locate turbo ball in rendered frame")
    return point


def calibrate_turbo(env: Any, target: Point, dx_sign: int, dy_sign: int) -> tuple[int, int]:
    """Find source coordinates whose renderer lands nearest the target pixel."""
    from breakout_turbo_env import FIXED_POINT_ONE, RAW_HEIGHT, RAW_WIDTH

    common = dict(
        paddle_x=40 * FIXED_POINT_ONE,
        ball_vx=dx_sign * FIXED_POINT_ONE,
        ball_vy=dy_sign * FIXED_POINT_ONE,
        bricks=(1 << 108) - 1,
        lives=5,
    )
    best_x = (10**9, 0)
    for source_x in range(RAW_WIDTH):
        env.configure_lane(0, ball_x=source_x * FIXED_POINT_ONE, ball_y=50 * FIXED_POINT_ONE, **common)
        point = find_ball(env.render(), min_y=31, max_y=188)
        if point is None:
            continue
        best_x = min(best_x, (abs(point.x - target.x), source_x))
    best_y = (10**9, 0)
    for source_y in range(RAW_HEIGHT):
        env.configure_lane(0, ball_x=48 * FIXED_POINT_ONE, ball_y=source_y * FIXED_POINT_ONE, **common)
        point = find_ball(env.render(), min_y=28, max_y=189)
        if point is None:
            continue
        best_y = min(best_y, (abs(point.y - target.y), source_y))
    return best_x[1], best_y[1]


def compare_corner(reference: StableReference, corner: str, frames: int) -> list[Transition]:
    from breakout_turbo_env import BreakoutVecEnv, FIXED_POINT_ONE

    if corner == "top-left":
        dx_sign, start = -1, Point(10, 34)
    elif corner == "top-right":
        dx_sign, start = 1, Point(148, 34)
    else:
        raise ValueError(corner)
    state = reference.find_flight(dx_sign, -1)
    reference.force(state, start)

    turbo = BreakoutVecEnv(num_envs=1, num_threads=1, frame_skip=1, frame_stack=1)
    turbo.reset()
    source_x, source_y = calibrate_turbo(turbo, start, dx_sign, -1)
    turbo.configure_lane(
        0,
        paddle_x=40 * FIXED_POINT_ONE,
        ball_x=source_x * FIXED_POINT_ONE,
        ball_y=source_y * FIXED_POINT_ONE + FIXED_POINT_ONE // 2,
        ball_vx=dx_sign * FIXED_POINT_ONE,
        ball_vy=-3 * FIXED_POINT_ONE // 2,
        bricks=(1 << 108) - 1,
        lives=5,
    )
    stable_previous = reference.point()
    turbo_previous = Point(source_x, source_y)
    rows: list[Transition] = []
    try:
        for index in range(frames):
            _, stable_reward, _, _, _ = reference.env.step(reference.action())
            _, turbo_reward, _, _, turbo_info = turbo.step(np.array([0], dtype=np.uint8))
            stable_point = reference.point()
            turbo_point = Point(
                int(turbo_info["ball_x"][0] // FIXED_POINT_ONE),
                int(turbo_info["ball_y"][0] // FIXED_POINT_ONE),
            )
            rows.append(
                Transition(
                    frame=index + 1,
                    stable=stable_point,
                    turbo=turbo_point,
                    stable_delta=Point(stable_point.x - stable_previous.x, stable_point.y - stable_previous.y),
                    turbo_delta=Point(turbo_point.x - turbo_previous.x, turbo_point.y - turbo_previous.y),
                    stable_reward=float(stable_reward),
                    turbo_reward=float(turbo_reward[0]),
                )
            )
            stable_previous, turbo_previous = stable_point, turbo_point
    finally:
        turbo.close()
    return rows


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    default_stable = repo.parent / "stable-retro-turbo"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-repo", type=Path, default=default_stable)
    parser.add_argument("--scenario", choices=("top-left", "top-right", "both"), default="both")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "python"))
    data_dir = args.stable_repo / "stable_retro/data/stable" / GAME
    missing = [name for name in ("Start.state", "data.json", "scenario.json") if not (data_dir / name).is_file()]
    if missing:
        raise SystemExit(f"Stable Retro reference data is incomplete: {', '.join(missing)}")
    reference = StableReference(data_dir)
    scenarios = ("top-left", "top-right") if args.scenario == "both" else (args.scenario,)
    report: dict[str, Any] = {
        "coordinate_discovery": {"x_offset": reference.x_offset, "y_offset": reference.y_offset},
        "scenarios": {},
    }
    try:
        for scenario in scenarios:
            report["scenarios"][scenario] = [asdict(row) for row in compare_corner(reference, scenario, args.frames)]
    finally:
        reference.close()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for scenario, rows in report["scenarios"].items():
            print(f"{scenario}:")
            for row in rows:
                stable, turbo = row["stable"], row["turbo"]
                sd, td = row["stable_delta"], row["turbo_delta"]
                print(
                    f"  {row['frame']:02d} stable=({stable['x']:3d},{stable['y']:3d}) d=({sd['x']:+d},{sd['y']:+d}) "
                    f"turbo=({turbo['x']:3d},{turbo['y']:3d}) d=({td['x']:+d},{td['y']:+d}) "
                    f"reward={row['stable_reward']:.1f}/{row['turbo_reward']:.1f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
