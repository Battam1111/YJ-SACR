import warnings

import gymnasium as gym

from tianshou.env import ShmemVectorEnv, VectorEnvNormObs, DummyVectorEnv

try:
    import envpool
except ImportError:
    envpool = None

# 渲染相关，需要注意代码修改问题
def make_mujoco_env(task, seed, training_num, test_num, obs_norm):
    """Wrapper function for Mujoco env.

    If EnvPool is installed, it will automatically switch to EnvPool's Mujoco env.

    :return: a tuple of (single env, training envs, test envs).
    """
    if envpool is not None:
        train_envs = env = envpool.make_gymnasium(
            task, num_envs=training_num, seed=seed
        )
        test_envs = envpool.make_gymnasium(task, num_envs=test_num, seed=seed)
    else:
        warnings.warn(
            "Recommend using envpool (pip install envpool) "
            "to run Mujoco environments more efficiently."
        )
        env = gym.make(task)
        train_envs = ShmemVectorEnv(
            [lambda: gym.make(task) for _ in range(training_num)]
        )
        # 渲染测试环境，因为只进行推理方法的改进
        test_envs = ShmemVectorEnv([lambda: gym.make(task) for _ in range(test_num)])
        train_envs.seed(seed)
        test_envs.seed(seed)
    if obs_norm:
        # obs norm wrapper
        train_envs = VectorEnvNormObs(train_envs)
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
        test_envs.set_obs_rms(train_envs.get_obs_rms())
    return env, train_envs, test_envs
