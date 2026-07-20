#!/usr/bin/env python3
"""Capture a short deterministic native-render gameplay GIF."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from breakout_turbo_env import FIXED_POINT_ONE, BreakoutVecEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--output", type=Path, default=Path("demo.gif"))
    return parser


def tracking_action(info: dict[str, np.ndarray]) -> int:
    if int(info["ball_y"][0]) == 0:
        return 1
    ball_x = int(info["ball_x"][0]) // FIXED_POINT_ONE
    paddle_x = int(info["paddle_x"][0]) // FIXED_POINT_ONE
    if ball_x < paddle_x + 6:
        return 3
    if ball_x > paddle_x + 10:
        return 2
    return 0


def capture(frames: int, output: Path) -> None:
    if frames <= 0:
        raise ValueError("frames must be positive")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required to capture the demo")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "160x210",
        "-framerate",
        "60",
        "-i",
        "-",
        "-filter_complex",
        (
            "fps=30,scale=320:420:flags=neighbor,split[a][b];"
            "[a]palettegen=max_colors=64:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
        ),
        "-loop",
        "0",
        str(output),
    ]
    env = BreakoutVecEnv(
        num_envs=1,
        num_threads=1,
        frame_skip=1,
        frame_stack=1,
        info_filter="all",
    )
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        _, info = env.reset()
        for _ in range(frames):
            action = np.asarray([tracking_action(info)], dtype=np.uint8)
            _, _, terminated, truncated, info = env.step(action)
            process.stdin.write(env.render().tobytes())
            if bool(terminated[0] or truncated[0]):
                _, info = env.reset(
                    options={"reset_mask": np.asarray([True], dtype=np.bool_)}
                )
    finally:
        env.close()
        process.stdin.close()
    if process.wait() != 0:
        raise SystemExit("ffmpeg failed to create the demo")
    print(output)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    capture(args.frames, args.output)


if __name__ == "__main__":
    main()
