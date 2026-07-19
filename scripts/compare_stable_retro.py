#!/usr/bin/env python3
"""Differential probe for Stable Retro Breakout and breakout-turbo-env.

This reusable developer/test harness deliberately does not make the ROM or
Stable Retro a package dependency. It discovers the reference ball-x RAM
variable from rendered motion instead of relying on a recorded trace.
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


@dataclass(frozen=True)
class EpisodeResult:
    policy: str
    seed: int | None
    exact: bool
    frames: int
    completed: bool
    score: int
    lives: int
    mismatch: dict[str, Any] | None


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
            inttype=self._retro.data.Integrations.ALL,
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
        self._info_path = Path(self._temporary.name) / "data.json"
        self._info_path.write_text(json.dumps(data))
        self.env = self._make_env()

    def _make_env(self) -> Any:
        return self._retro.make(
            GAME,
            state="Start",
            info=str(self._info_path),
            scenario=str(self._scenario),
            inttype=self._retro.data.Integrations.ALL,
            render_mode="rgb_array",
        )

    def reopen(self) -> None:
        """Start with fresh emulator-side analog input state."""
        self.env.close()
        self.env = self._make_env()

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

    def paddle_x(self) -> int:
        return int(self.env.data.lookup_value("paddle_x_probe")) - self.paddle_offset

    def awaiting_fire(self) -> bool:
        return int(self.env.data.lookup_value("ball_y")) == 0

    def score(self) -> int:
        return int(self.env.data.lookup_value("score"))

    def lives(self) -> int:
        return int(self.env.data.lookup_value("lives"))


def compare_corner(reference: StableReference, corner: str, frames: int) -> list[Transition]:
    from breakout_turbo_env import FIXED_POINT_ONE, BreakoutVecEnv

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
    source_x, source_y = start.x, start.y
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


def reflect_playfield_x(value: float) -> float:
    """Reflect a projected ball coordinate inside the Atari wall bounds."""
    low, high = 7.0, 151.0
    while value < low or value > high:
        if value < low:
            value = 2 * low - value
        if value > high:
            value = 2 * high - value
    return value


def compare_episode(
    reference: StableReference,
    *,
    policy: str,
    max_frames: int,
    aim: int = 8,
    seed: int | None = None,
) -> EpisodeResult:
    """Run one live cartridge/Turbo episode with identical generated actions."""
    from breakout_turbo_env import FIXED_POINT_ONE, BreakoutVecEnv

    reference.reopen()
    stable_frame, stable_info = reference.env.reset()
    turbo = BreakoutVecEnv(num_envs=1, num_threads=1, frame_skip=1, frame_stack=1)
    _, turbo_info = turbo.reset()
    turbo_frame = turbo.render()
    rng = np.random.default_rng(seed)
    label = f"predictive-aim{aim}" if policy == "predictive" else policy

    def mismatch(
        frame: int,
        stable_reward: float,
        turbo_reward: float,
        stable_terminated: bool,
        turbo_terminated: bool,
        stable_truncated: bool,
        turbo_truncated: bool,
    ) -> dict[str, Any] | None:
        different = np.any(stable_frame != turbo_frame, axis=2)
        pixel_count = int(np.count_nonzero(different))
        reward_differs = float(stable_reward) != float(turbo_reward)
        terminal_differs = stable_terminated != turbo_terminated or stable_truncated != turbo_truncated
        stable_score = reference.score()
        turbo_score = int(turbo_info["score"][0])
        stable_lives = reference.lives()
        turbo_lives = int(turbo_info["lives"][0])
        state_differs = stable_score != turbo_score or stable_lives != turbo_lives
        if not (pixel_count or reward_differs or terminal_differs or state_differs):
            return None
        bbox = None
        stable_colors: list[list[int]] = []
        turbo_colors: list[list[int]] = []
        if pixel_count:
            ys, xs = np.where(different)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            stable_colors = np.unique(stable_frame[ys, xs], axis=0).astype(int).tolist()
            turbo_colors = np.unique(turbo_frame[ys, xs], axis=0).astype(int).tolist()
        return {
            "frame": frame,
            "different_pixels": pixel_count,
            "bbox": bbox,
            "stable_colors": stable_colors,
            "turbo_colors": turbo_colors,
            "stable_reward": float(stable_reward),
            "turbo_reward": float(turbo_reward),
            "stable_terminated": stable_terminated,
            "turbo_terminated": turbo_terminated,
            "stable_truncated": stable_truncated,
            "turbo_truncated": turbo_truncated,
            "stable_score": stable_score,
            "turbo_score": turbo_score,
            "stable_lives": stable_lives,
            "turbo_lives": turbo_lives,
            "stable_ball": asdict(reference.point()),
            "turbo_ball": {
                "x": int(turbo_info["ball_x"][0] // FIXED_POINT_ONE),
                "y": int(turbo_info["ball_y"][0] // FIXED_POINT_ONE),
            },
            "turbo_velocity": {
                "x": int(turbo_info["ball_vx"][0]),
                "y": int(turbo_info["ball_vy"][0]),
            },
            "turbo_collision_events": int(turbo_info["collision_events"][0]),
        }

    try:
        initial_mismatch = mismatch(0, 0.0, 0.0, False, False, False, False)
        if initial_mismatch is not None:
            return EpisodeResult(label, seed, False, 0, False, 0, 5, initial_mismatch)

        for frame in range(1, max_frames + 1):
            fire = reference.awaiting_fire()
            if fire:
                direction = 0
            elif policy == "tracking":
                ball_x = reference.point().x
                paddle_x = reference.paddle_x()
                direction = -1 if ball_x < paddle_x + 6 else 1 if ball_x > paddle_x + 10 else 0
            elif policy == "predictive":
                ball_x = float(turbo_info["ball_x"][0]) / FIXED_POINT_ONE
                ball_y = float(turbo_info["ball_y"][0]) / FIXED_POINT_ONE
                ball_vx = float(turbo_info["ball_vx"][0]) / FIXED_POINT_ONE
                ball_vy = float(turbo_info["ball_vy"][0]) / FIXED_POINT_ONE
                paddle_x = float(turbo_info["paddle_x"][0]) / FIXED_POINT_ONE
                target = paddle_x
                if ball_vy > 0:
                    target = reflect_playfield_x(ball_x + ball_vx * (186 - ball_y) / ball_vy) - aim
                direction = -1 if target < paddle_x else 1 if target > paddle_x else 0
            elif policy == "random":
                direction = int(rng.integers(-1, 2))
            else:
                raise ValueError(policy)

            turbo_action = 1 if fire else 3 if direction < 0 else 2 if direction > 0 else 0
            stable_frame, stable_reward, stable_terminated, stable_truncated, stable_info = reference.env.step(
                reference.action(direction, fire=fire)
            )
            _, turbo_reward, turbo_terminated, turbo_truncated, turbo_info = turbo.step(
                np.asarray([turbo_action], dtype=np.uint8)
            )
            turbo_frame = turbo.render()
            difference = mismatch(
                frame,
                float(stable_reward),
                float(turbo_reward[0]),
                bool(stable_terminated),
                bool(turbo_terminated[0]),
                bool(stable_truncated),
                bool(turbo_truncated[0]),
            )
            if difference is not None:
                return EpisodeResult(
                    label,
                    seed,
                    False,
                    frame,
                    False,
                    reference.score(),
                    reference.lives(),
                    difference,
                )
            if stable_terminated or stable_truncated:
                return EpisodeResult(
                    label,
                    seed,
                    True,
                    frame,
                    True,
                    reference.score(),
                    reference.lives(),
                    None,
                )
        return EpisodeResult(
            label,
            seed,
            True,
            max_frames,
            False,
            reference.score(),
            reference.lives(),
            None,
        )
    finally:
        turbo.close()


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    default_stable = repo.parent / "stable-retro-turbo"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-repo", type=Path, default=default_stable)
    parser.add_argument("--mode", choices=("corners", "episodes", "all"), default="all")
    parser.add_argument("--scenario", choices=("top-left", "top-right", "both"), default="both")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--policy", choices=("tracking", "predictive", "random", "all"), default="all")
    parser.add_argument("--aims", default="4,6,8,10,12", help="comma-separated predictive paddle offsets")
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated random-policy seeds")
    parser.add_argument("--max-frames", type=int, default=8000)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "python"))
    data_dir = args.stable_repo / "stable_retro/data/stable" / GAME
    missing = [
        name
        for name in ("rom.a26", "Start.state", "data.json", "scenario.json")
        if not (data_dir / name).is_file()
    ]
    if missing:
        raise SystemExit(f"Stable Retro reference data is incomplete: {', '.join(missing)}")
    reference = StableReference(data_dir)
    scenarios = ("top-left", "top-right") if args.scenario == "both" else (args.scenario,)
    report: dict[str, Any] = {
        "coordinate_discovery": {"x_offset": reference.x_offset, "y_offset": reference.y_offset},
        "scenarios": {},
        "corners_exact": True,
        "episodes": [],
    }
    try:
        if args.mode in ("corners", "all"):
            for scenario in scenarios:
                report["scenarios"][scenario] = [asdict(row) for row in compare_corner(reference, scenario, args.frames)]
            report["corners_exact"] = all(
                row["stable"] == row["turbo"]
                and row["stable_delta"] == row["turbo_delta"]
                and row["stable_reward"] == row["turbo_reward"]
                for rows in report["scenarios"].values()
                for row in rows
            )
        if args.mode in ("episodes", "all"):
            if args.policy in ("tracking", "all"):
                report["episodes"].append(
                    asdict(compare_episode(reference, policy="tracking", max_frames=args.max_frames))
                )
            if args.policy in ("predictive", "all"):
                for aim in (int(value) for value in args.aims.split(",") if value):
                    report["episodes"].append(
                        asdict(compare_episode(reference, policy="predictive", aim=aim, max_frames=args.max_frames))
                    )
            if args.policy in ("random", "all"):
                for seed in (int(value) for value in args.seeds.split(",") if value):
                    report["episodes"].append(
                        asdict(compare_episode(reference, policy="random", seed=seed, max_frames=args.max_frames))
                    )
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
        for episode in report["episodes"]:
            status = "exact" if episode["exact"] else "MISMATCH"
            suffix = " terminal" if episode["completed"] else ""
            seed = "" if episode["seed"] is None else f" seed={episode['seed']}"
            print(
                f"{episode['policy']}{seed}: {status} through {episode['frames']} frames{suffix}; "
                f"score={episode['score']} lives={episode['lives']}"
            )
            if episode["mismatch"] is not None:
                print(f"  {json.dumps(episode['mismatch'], sort_keys=True)}")
    episodes_exact = all(episode["exact"] for episode in report["episodes"])
    return 0 if report["corners_exact"] and episodes_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
