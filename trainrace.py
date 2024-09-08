# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import argparse

import carb
import torch as th
from envracetrack import JetBotEnv

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

log_dir = "/workspace/cnn_policy/05_09_AHK_6"
# set headles to false to visualize training
my_env = JetBotEnv(headless=True)

# in test mode we manually install sb3
if args.test is True:
    import omni.kit.pipapi

    omni.kit.pipapi.install("stable-baselines3==2.0.0", module="stable_baselines3")
    omni.kit.pipapi.install("tensorboard")

# import stable baselines
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.ppo import MlpPolicy
except Exception as e:
    carb.log_error(e)
    carb.log_error(
        "please install stable-baselines3 in the current python environment or run the following to install into the builtin python environment ./python.sh -m pip install stable-baselines3"
    )
    exit()

try:
    import tensorboard
except Exception as e:
    carb.log_error(e)
    carb.log_error(
        "please install tensorboard in the current python environment or run the following to install into the builtin python environment ./python.sh -m pip install tensorboard"
    )
    exit()

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(vf=[256, 256, 256, 256], pi=[256, 256, 256, 256])])
policy = MlpPolicy
total_timesteps = 1500000

if args.test is True:
    total_timesteps = 1500000

checkpoint_callback = CheckpointCallback(save_freq=200000, save_path=log_dir, name_prefix="jetbot_policy_checkpoint")
model = PPO(
    policy,
    my_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=10240,
    batch_size=2048,
    learning_rate=0.00015,
    gamma=0.9,
    ent_coef=7.5e-08,
    clip_range=0.3,
    n_epochs=3,
    gae_lambda=1.0,
    max_grad_norm=0.9,
    vf_coef=0.95,
    device="cuda:0",
    tensorboard_log=log_dir,
)
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])

model.save(log_dir + "/jetbot_policy")

my_env.close()
