"""Train an SB3 PPO policy while preserving Breakout's manual-reset contract."""

from breakout_turbo_env import BreakoutVecEnv
from breakout_turbo_env.sb3 import make_sb3_vec_env
from stable_baselines3 import PPO


def main() -> None:
    native_env = BreakoutVecEnv(num_envs=16, num_threads=8, obs_copy="safe_view")
    env = make_sb3_vec_env(native_env)
    try:
        model = PPO("CnnPolicy", env, verbose=1)
        model.learn(total_timesteps=100_000)
    finally:
        env.close()


if __name__ == "__main__":
    main()
