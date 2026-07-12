#!/usr/bin/env python3
"""Train a PPO policy for deterministic breakout-turbo-env."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from breakout_turbo_env import BreakoutVecEnv, FIXED_POINT_ONE

_LAYOUTS = ("full", "checker", "tunnel", "sparse")
_FEATURES = 5
_HIDDEN = 64


@dataclass(frozen=True)
class EvalResult:
    score: int
    reward: float
    lives: int
    steps: int
    solved: bool


def create_run_dir(runs_dir: Path, *, algo: str = "ppo") -> Path:
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PPO policy for breakout-turbo-env.")
    parser.add_argument("--layout", choices=_LAYOUTS, default="full")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--updates", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--tracking-reward", type=float, default=0.05)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--max-eval-steps", type=int, default=30_000)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    return parser


def _validate(args: argparse.Namespace) -> None:
    positive_ints = ("num_envs", "horizon", "updates", "epochs", "minibatch_size", "frame_skip", "max_eval_steps", "eval_every")
    if any(getattr(args, name) <= 0 for name in positive_ints):
        raise ValueError("num-envs, horizon, updates, epochs, minibatch-size, frame-skip, max-eval-steps, and eval-every must be positive")
    if args.minibatch_size > args.num_envs * args.horizon:
        raise ValueError("minibatch-size cannot exceed num-envs * horizon")
    if args.learning_rate <= 0 or args.tracking_reward < 0:
        raise ValueError("learning-rate must be positive and tracking-reward must be non-negative")
    if not 0 < args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        raise ValueError("gamma must be in (0, 1] and gae-lambda must be in [0, 1]")
    if not 0 < args.clip_ratio < 1 or args.entropy_coef < 0 or args.value_coef < 0:
        raise ValueError("clip-ratio must be in (0, 1); entropy-coef and value-coef must be non-negative")


def features(info: dict[str, np.ndarray]) -> np.ndarray:
    """Compact Markov features from the environment's documented native signals."""
    return np.stack(
        (
            (info["paddle_x"] + 9 * FIXED_POINT_ONE) / (96 * FIXED_POINT_ONE),
            info["ball_x"] / (96 * FIXED_POINT_ONE),
            info["ball_y"] / (96 * FIXED_POINT_ONE),
            info["ball_vx"] / (2 * FIXED_POINT_ONE),
            info["ball_vy"] / (2 * FIXED_POINT_ONE),
        ),
        axis=1,
    ).astype(np.float32)


def tracking_actions(info: dict[str, np.ndarray]) -> np.ndarray:
    """Dense shaping target: center the paddle beneath the ball."""
    delta = info["ball_x"] - (info["paddle_x"] + 9 * FIXED_POINT_ONE)
    return np.where(delta < -FIXED_POINT_ONE, 1, np.where(delta > FIXED_POINT_ONE, 2, 0)).astype(np.int64)


def _reset(env: BreakoutVecEnv, layout: str, mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if mask is None:
        mask = np.ones(env.num_envs, dtype=np.bool_)
    starts = np.full(env.num_envs, layout, dtype=object)
    _, info = env.reset(options={"reset_mask": mask, "start_ids": starts})
    return info


def _actor_critic(torch: Any):
    class ActorCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(_FEATURES, _HIDDEN),
                torch.nn.Tanh(),
                torch.nn.Linear(_HIDDEN, _HIDDEN),
                torch.nn.Tanh(),
            )
            self.actor = torch.nn.Linear(_HIDDEN, 3)
            self.critic = torch.nn.Linear(_HIDDEN, 1)

        def forward(self, values):
            hidden = self.trunk(values)
            return self.actor(hidden), self.critic(hidden).squeeze(-1)

    return ActorCritic()


def evaluate(model: Any, torch: Any, *, layout: str, frame_skip: int, max_steps: int) -> EvalResult:
    env = BreakoutVecEnv(
        num_envs=1,
        num_threads=1,
        obs_resize=(8, 8),
        frame_stack=1,
        frame_skip=frame_skip,
        info_filter="all",
    )
    info = _reset(env, layout)
    total_reward = 0.0
    try:
        model.eval()
        for step in range(1, max_steps + 1):
            with torch.no_grad():
                logits, _ = model(torch.from_numpy(features(info)))
                action = torch.argmax(logits, dim=1).numpy().astype(np.uint8)
            _, reward, terminated, _, info = env.step(action)
            total_reward += float(reward[0])
            if terminated[0]:
                return EvalResult(
                    score=int(info["score"][0]),
                    reward=total_reward,
                    lives=int(info["lives"][0]),
                    steps=step,
                    solved=bool(info["bricks_remaining"][0] == 0),
                )
        return EvalResult(
            score=int(info["score"][0]),
            reward=total_reward,
            lives=int(info["lives"][0]),
            steps=max_steps,
            solved=False,
        )
    finally:
        env.close()


def save_policy(path: Path, model: Any, *, layout: str, frame_skip: int, seed: int, result: EvalResult) -> None:
    """Save a safe NumPy artifact so playback does not deserialize Python objects."""
    metadata = {
        "algorithm": "PPO",
        "format_version": 1,
        "layout": layout,
        "frame_skip": frame_skip,
        "seed": seed,
        "feature_names": ["paddle_center", "ball_x", "ball_y", "ball_vx", "ball_vy"],
        "hidden_size": _HIDDEN,
        "score": result.score,
        "reward": result.reward,
        "lives": result.lives,
        "steps": result.steps,
        "solved": result.solved,
    }
    arrays = {name.replace(".", "__"): tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, separators=(",", ":")))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate(args)
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PPO requires torch; run `uv sync` to install the locked dependency") from exc

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    run_dir = create_run_dir(args.runs_dir, algo="ppo")
    model = _actor_critic(torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    threads = args.threads if args.threads is not None else min(args.num_envs, 8)
    env = BreakoutVecEnv(
        num_envs=args.num_envs,
        num_threads=threads,
        obs_resize=(8, 8),
        frame_stack=1,
        frame_skip=args.frame_skip,
        info_filter="all",
    )
    info = _reset(env, args.layout)
    batch_size = args.num_envs * args.horizon
    latest_eval: EvalResult | None = None

    try:
        for update in range(1, args.updates + 1):
            observations = np.empty((args.horizon, args.num_envs, _FEATURES), dtype=np.float32)
            actions = np.empty((args.horizon, args.num_envs), dtype=np.int64)
            log_probs = np.empty((args.horizon, args.num_envs), dtype=np.float32)
            values = np.empty((args.horizon, args.num_envs), dtype=np.float32)
            rewards = np.empty((args.horizon, args.num_envs), dtype=np.float32)
            dones = np.empty((args.horizon, args.num_envs), dtype=np.float32)

            model.train()
            for step in range(args.horizon):
                observation = features(info)
                observations[step] = observation
                target_actions = tracking_actions(info)
                with torch.no_grad():
                    logits, value = model(torch.from_numpy(observation))
                    distribution = torch.distributions.Categorical(logits=logits)
                    action = distribution.sample()
                    actions[step] = action.numpy()
                    log_probs[step] = distribution.log_prob(action).numpy()
                    values[step] = value.numpy()

                _, extrinsic_reward, terminated, _, next_info = env.step(actions[step].astype(np.uint8))
                shaped_reward = np.where(actions[step] == target_actions, args.tracking_reward, -args.tracking_reward)
                rewards[step] = extrinsic_reward + shaped_reward.astype(np.float32)
                dones[step] = terminated.astype(np.float32)
                if terminated.any():
                    reset_info = _reset(env, args.layout, terminated)
                    for key, value in reset_info.items():
                        if key in next_info:
                            next_info[key][terminated] = value[terminated]
                info = next_info

            with torch.no_grad():
                _, next_value = model(torch.from_numpy(features(info)))
                next_value_np = next_value.numpy()
            advantages = np.empty_like(rewards)
            last_advantage = np.zeros(args.num_envs, dtype=np.float32)
            for step in range(args.horizon - 1, -1, -1):
                nonterminal = 1.0 - dones[step]
                bootstrap = next_value_np if step == args.horizon - 1 else values[step + 1]
                delta = rewards[step] + args.gamma * bootstrap * nonterminal - values[step]
                last_advantage = delta + args.gamma * args.gae_lambda * nonterminal * last_advantage
                advantages[step] = last_advantage
            returns = advantages + values

            flat_observations = torch.from_numpy(observations.reshape(batch_size, _FEATURES))
            flat_actions = torch.from_numpy(actions.reshape(batch_size))
            flat_log_probs = torch.from_numpy(log_probs.reshape(batch_size))
            flat_advantages = torch.from_numpy(advantages.reshape(batch_size))
            flat_returns = torch.from_numpy(returns.reshape(batch_size))
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

            for _ in range(args.epochs):
                for indices in torch.randperm(batch_size).split(args.minibatch_size):
                    logits, predicted_values = model(flat_observations[indices])
                    distribution = torch.distributions.Categorical(logits=logits)
                    new_log_probs = distribution.log_prob(flat_actions[indices])
                    ratio = (new_log_probs - flat_log_probs[indices]).exp()
                    unclipped = ratio * flat_advantages[indices]
                    clipped = torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio) * flat_advantages[indices]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = torch.nn.functional.mse_loss(predicted_values, flat_returns[indices])
                    entropy = distribution.entropy().mean()
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

            if update % args.eval_every == 0 or update == args.updates:
                latest_eval = evaluate(
                    model, torch, layout=args.layout, frame_skip=args.frame_skip, max_steps=args.max_eval_steps
                )
                print(
                    f"update={update} eval_score={latest_eval.score} eval_reward={latest_eval.reward:.1f} "
                    f"lives={latest_eval.lives} steps={latest_eval.steps} solved={latest_eval.solved}",
                    flush=True,
                )
                if latest_eval.solved:
                    break
    finally:
        env.close()

    if latest_eval is None:
        latest_eval = evaluate(model, torch, layout=args.layout, frame_skip=args.frame_skip, max_steps=args.max_eval_steps)
    policy_path = run_dir / "policy.npz"
    save_policy(policy_path, model, layout=args.layout, frame_skip=args.frame_skip, seed=args.seed, result=latest_eval)
    print(f"saved={policy_path} verified={latest_eval.solved}")
    return 0 if latest_eval.solved else 2


if __name__ == "__main__":
    raise SystemExit(main())
