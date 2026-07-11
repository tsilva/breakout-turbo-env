from __future__ import annotations

import argparse
import time

import numpy as np

from .env import BreakoutVecEnv

_LAYOUTS = ("full", "checker", "tunnel", "sparse")


class _ObservationViewer:
    def __init__(self, pygame, observation: np.ndarray, scale: int = 2):
        from pygame._sdl2 import Renderer, Window

        stack, height, width = observation.shape[1:]
        self._pygame = pygame
        self._size = (width * stack * scale, height * scale)
        self._window = Window("Breakout Turbo | processed observation", size=self._size)
        self._renderer = Renderer(self._window)

    def show(self, observation: np.ndarray) -> None:
        from pygame._sdl2 import Texture

        tiled = np.concatenate(tuple(observation[0]), axis=1)
        rgb = np.repeat(tiled[..., None], 3, axis=2)
        surface = self._pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        texture = Texture.from_surface(self._renderer, surface)
        self._renderer.clear()
        self._renderer.blit(texture, self._pygame.Rect(0, 0, *self._size))
        self._renderer.present()

    def close(self) -> None:
        self._window.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play breakout-turbo-env manually")
    parser.add_argument("--layout", choices=_LAYOUTS, default="full")
    parser.add_argument("--scale", type=int, default=8, help="integer window scale")
    parser.add_argument("--fps", type=int, default=60, help="display updates per second")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument(
        "--show-obs",
        action="store_true",
        help="show the four processed policy frames in a second window",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="exit after this many frames; 0 runs until the window closes",
    )
    return parser


def _print_episode_stats(
    info: dict[str, np.ndarray],
    *,
    episode: int,
    layout: str,
    episode_return: float,
    display_steps: int,
    elapsed: float,
) -> None:
    score = int(info["score"][0])
    lives = int(info["lives"][0])
    bricks_remaining = int(info["bricks_remaining"][0])
    native_ticks = int(info["tick"][0])
    outcome = "cleared" if bricks_remaining == 0 else "game_over"
    print(
        "episode_end"
        f" episode={episode} layout={layout} outcome={outcome}"
        f" score={score} return={episode_return:.1f} lives={lives}"
        f" bricks_cleared={score} native_ticks={native_ticks}"
        f" display_steps={display_steps} elapsed_seconds={elapsed:.2f}",
        flush=True,
    )


def _hud_text(info: dict[str, np.ndarray], *, paused: bool) -> str:
    score = int(info.get("score", np.zeros(1, dtype=np.int64))[0])
    lives = int(info.get("lives", np.ones(1, dtype=np.int64))[0])
    bricks = int(info.get("bricks_remaining", np.zeros(1, dtype=np.int64))[0])
    status = "  PAUSED" if paused else ""
    return f"SCORE {score:03d}    LIVES {lives}    BRICKS {bricks:02d}{status}"


def run(
    *,
    layout: str,
    scale: int,
    fps: int,
    frame_skip: int,
    max_frames: int = 0,
    show_obs: bool = False,
) -> None:
    if scale <= 0 or fps <= 0 or frame_skip <= 0 or max_frames < 0:
        raise ValueError("scale, fps, and frame-skip must be positive; max-frames must be non-negative")

    import pygame

    pygame.init()
    env = BreakoutVecEnv(
        num_envs=1,
        num_threads=1,
        frame_skip=frame_skip,
        frame_stack=4,
        info_filter="all",
    )
    layout_index = _LAYOUTS.index(layout)
    reset_mask = np.ones(1, dtype=np.bool_)
    start_indices = np.array([layout_index], dtype=np.int32)
    observation, info = env.reset(
        options={"reset_mask": reset_mask, "start_indices": start_indices}
    )
    raw_height, raw_width = env.render().shape[:2]
    game_size = (raw_width * scale, raw_height * scale)
    hud_height = max(28, scale * 4)
    screen = pygame.display.set_mode((game_size[0], game_size[1] + hud_height))
    hud_font = pygame.font.Font(None, max(18, scale * 3))
    observation_viewer = _ObservationViewer(pygame, observation) if show_obs else None
    clock = pygame.time.Clock()
    running = True
    paused = False
    frame_count = 0
    episode = 1
    episode_return = 0.0
    episode_steps = 0
    episode_started = time.perf_counter()

    try:
        while running:
            reset_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_SPACE, pygame.K_r):
                        reset_requested = True
                    elif event.key == pygame.K_p:
                        paused = not paused

            if reset_requested:
                observation, info = env.reset(
                    options={"reset_mask": reset_mask, "start_indices": start_indices}
                )
                episode_return = 0.0
                episode_steps = 0
                episode_started = time.perf_counter()

            if running and not paused:
                keys = pygame.key.get_pressed()
                left = keys[pygame.K_LEFT] or keys[pygame.K_a]
                right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
                action = 1 if left and not right else 2 if right and not left else 0
                observation, reward, terminated, _, info = env.step(
                    np.array([action], dtype=np.uint8)
                )
                episode_return += float(reward[0])
                episode_steps += 1
                if terminated[0]:
                    _print_episode_stats(
                        info,
                        episode=episode,
                        layout=layout,
                        episode_return=episode_return,
                        display_steps=episode_steps,
                        elapsed=time.perf_counter() - episode_started,
                    )
                    episode += 1
                    # The environment never autoresets. The player performs an
                    # explicit masked reset so the same lifecycle is exercised.
                    observation, info = env.reset(
                        options={"reset_mask": reset_mask, "start_indices": start_indices}
                    )
                    episode_return = 0.0
                    episode_steps = 0
                    episode_started = time.perf_counter()

            frame = env.render()
            surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            if scale != 1:
                surface = pygame.transform.scale(surface, game_size)
            screen.fill((16, 18, 24))
            hud_surface = hud_font.render(
                _hud_text(info, paused=paused), True, (245, 245, 245)
            )
            screen.blit(hud_surface, (10, (hud_height - hud_surface.get_height()) // 2))
            screen.blit(surface, (0, hud_height))
            score = int(info.get("score", np.zeros(1, dtype=np.int64))[0])
            lives = int(info.get("lives", np.ones(1, dtype=np.int64))[0])
            bricks = int(info.get("bricks_remaining", np.zeros(1, dtype=np.int64))[0])
            state = "PAUSED" if paused else "playing"
            pygame.display.set_caption(
                f"Breakout Turbo | {state} | score {score} | lives {lives} | bricks {bricks}"
            )
            pygame.display.flip()
            if observation_viewer is not None:
                observation_viewer.show(observation)
            clock.tick(fps)
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                running = False
    finally:
        if observation_viewer is not None:
            observation_viewer.close()
        env.close()
        pygame.quit()


def main() -> None:
    args = build_parser().parse_args()
    run(
        layout=args.layout,
        scale=args.scale,
        fps=args.fps,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        show_obs=args.show_obs,
    )


if __name__ == "__main__":
    main()
