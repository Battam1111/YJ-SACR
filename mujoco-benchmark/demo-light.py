import os
import gymnasium as gym
import torch
import numpy as np
from tianshou.data import Batch
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.monitoring import video_recorder

from demonstrations.ddpg_policy import get_ddpg_policy
from demonstrations.td3_policy import get_td3_policy
from demonstrations.sac_policy import get_sac_policy
from demonstrations.redq_policy import get_redq_policy
from demonstrations.reinforce_policy import get_reinforce_policy
from demonstrations.a2c_policy import get_a2c_policy
from demonstrations.npg_policy import get_npg_policy
from demonstrations.trpo_policy import get_trpo_policy
from demonstrations.ppo_policy import get_ppo_policy


class CustomRecordVideo(RecordVideo):
    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.terminated = False
        self.truncated = False
        self.recorded_frames = 0  # Reset recorded frames for the new episode
        if self._video_enabled():
            self.start_video_recorder()
        return observations

    def start_video_recorder(self):
        self.close_video_recorder()
        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"
        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={"step_id": self.step_id, "episode_id": self.episode_id},
            disable_logger=self.disable_logger,
        )
        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True
        print(f"[DEBUG] Started recording: {video_name}")

    def step(self, action):
        observations, rewards, terminateds, truncateds, infos = self.env.step(action)
        self.step_id += 1
        if not self.is_vector_env:
            if terminateds or truncateds:
                self.episode_id += 1
                self.terminated = terminateds
                self.truncated = truncateds
        elif terminateds[0] or truncateds[0]:
            self.episode_id += 1
            self.terminated = terminateds[0]
            self.truncated = truncateds[0]

        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            # print(f"[DEBUG] Recording frame: {self.recorded_frames} frames recorded.")
            if self.video_length > 0:
                if self.recorded_frames >= self.video_length:
                    self.close_video_recorder()
            else:
                if not self.is_vector_env:
                    if terminateds or truncateds:
                        self.close_video_recorder()
                elif terminateds[0] or truncateds[0]:
                    self.close_video_recorder()

        elif self._video_enabled():
            self.start_video_recorder()

        return observations, rewards, terminateds, truncateds, infos

    def close_video_recorder(self):
        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.close()
            print("[DEBUG] Closed video recorder.")
        self.recording = False
        self.recorded_frames = 0

def load_policy(policy_type, env, policy_path, task, hidden_sizes, eval_method):
    kwargs = {
        "env": env,
        "hidden_sizes": hidden_sizes,
        "task": task,
        "policy_path": policy_path
    }

    if policy_type == "ddpg":
        policy, args = get_ddpg_policy(**kwargs)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device))
    elif policy_type == "td3":
        policy, args = get_td3_policy(**kwargs)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device))
    elif policy_type == "sac":
        policy, args = get_sac_policy(**kwargs)
        policy.eval_method = eval_method
        policy.load_state_dict(torch.load(policy_path, map_location=args.device))
    elif policy_type == "redq":
        policy, args = get_redq_policy(**kwargs)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device))
    elif policy_type == "reinforce":
        policy, args = get_reinforce_policy(**kwargs)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device)["model"])
    elif policy_type == "a2c":
        policy, args = get_a2c_policy(**kwargs)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device)["model"])
    elif policy_type == "npg":
        policy, args = get_npg_policy(**kwargs)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device)["model"])
    elif policy_type == "trpo":
        policy, args = get_trpo_policy(**kwargs)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device)["model"])
    elif policy_type == "ppo":
        policy, args = get_ppo_policy(**kwargs)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device)["model"])
    else:
        raise Exception("Unknown policy.")

    return policy, args

def simulate(task, policy_type, policy_path, hidden_sizes, eval_method, video_suffix):
    video_name_prefix = f"{policy_type.upper()}_{task}_{video_suffix}"
    video_folder = os.path.join("videos", task, policy_type)

    env = CustomRecordVideo(
        env=gym.make(task, render_mode="rgb_array"),
        video_folder=video_folder,
        name_prefix=video_name_prefix,
        video_length=0  # 设置为0表示整个episode录制
    )
    observation, info = env.reset()

    policy, args = load_policy(policy_type=policy_type, env=env, policy_path=policy_path, task=task, hidden_sizes=hidden_sizes, eval_method=eval_method)
    print(f"从 {policy_path} 加载的代理")

    max_steps = 2000  # 合理的步数
    total_steps = 0
    while total_steps < max_steps:
        batch = Batch(obs=[observation], info=info)
        action = policy.forward(batch=batch, state=observation).act[0].cpu().detach().numpy()

        observation, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        if terminated or truncated:
            print(f"[DEBUG] Episode terminated at total step {total_steps}, resetting environment.")
            observation, info = env.reset()

    env.close()

if __name__ == '__main__':
    policies = {
        "ddpg": [256, 256],
        "td3": [256, 256],
        "sac": [256, 256],
        "redq": [256, 256],
        "reinforce": [64, 64],
        "a2c": [64, 64],
        "npg": [64, 64],
        "trpo": [64, 64],
        "ppo": [64, 64]
    }

    tasks = [
        "Ant-v4",
        "HalfCheetah-v4",
        "Hopper-v4",
        "Humanoid-v4",
        "HumanoidStandup-v4",
        "InvertedDoublePendulum-v4",
        "InvertedPendulum-v4",
        "Pusher-v4",
        "Reacher-v4",
        "Swimmer-v4",
        "Walker2d-v4"
    ]

    selected_policy = "sac"
    selected_task = "Humanoid-v4"
    selected_policy_path = "log/Humanoid-v4/sac/4/240728-132838/policy.pth"

    simulate(
        task=selected_task,
        policy_type=selected_policy,
        policy_path=selected_policy_path,
        hidden_sizes=policies[selected_policy],
        eval_method='deterministic',
        video_suffix='deterministic'
    )

    simulate(
        task=selected_task,
        policy_type=selected_policy,
        policy_path=selected_policy_path,
        hidden_sizes=policies[selected_policy],
        eval_method='uniform_v1',
        video_suffix='uniform_v1'
    )
