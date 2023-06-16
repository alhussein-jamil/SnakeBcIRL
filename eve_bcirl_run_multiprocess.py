# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
import tracemalloc



import gym.spaces as spaces
import numpy as np
import torch
from hydra.utils import instantiate as hydra_instantiate
from rl_utils.common import ( get_size_for_space,
                             set_seed)
from rl_utils.logging import Logger
import os.path as osp
import random
import warnings
from typing import Dict
from eve_bcirl_env import EVE
import gym.spaces as spaces
import mediapy as media
import numpy as np
import torch
import yaml
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig
from rl_utils.logging import Logger
from tensordict.tensordict import TensorDict
from imitation_learning.policy_opt.storage import RolloutStorage
from imitation_learning.policy_opt.policy import Policy
from imitation_learning.policy_opt.storage import RolloutStorage
from gym.wrappers import FlattenObservation
from jiminy_py.viewer import Viewer
from torchrl.envs import ParallelEnv
from torchrl.envs.libs.gym import GymWrapper
import paralellEnvs as pe
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    Viewer.connect_backend()
    cfg = yaml.load(open("bcirl-eve.yaml", "r"), Loader=yaml.SafeLoader)
    cfg = DictConfig(cfg)

    cfg.num_steps = cfg.batch_size // cfg.num_envs


    print("num_envs: ", cfg.num_envs)
    print("num_steps: ", cfg.num_steps)
    def flatten_dict(d: Dict, parent_key = "") -> Dict:
        """
        Flattens a nested dictionary.
        """
        flat_dict = {}
        for key, value in d.items():
            if isinstance(value, dict) or isinstance(value, TensorDict) or isinstance(value, DictConfig):
                flat_dict.update(flatten_dict(value, parent_key + key + "."))
            else:
                flat_dict[parent_key + key] = float(value.detach().cpu())
        return flat_dict

    def set_seed(seed: int) -> None:
        """
        Sets the seed for numpy, python random, and pytorch.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    def print_dictionary(dictionary,name="", spaces = 0 ):
        print("\t"*spaces + name+":")
        for key, value in dictionary.items():
            if isinstance(value, dict):
                print_dictionary(value, key, spaces+1)
            else:
                print("\t"*(spaces+1) + key+": "+str(value))    

    set_seed(cfg.seed)


    device = torch.device(cfg.device)

    set_env_settings = {
        k: hydra_instantiate(v) if isinstance(v, DictConfig) else v
        for k, v in cfg.env.env_settings.items()
    }

    # envs = create_vectorized_envs(
    #     cfg.env.env_name,
    #     cfg.num_envs,
    #     seed=cfg.seed,
    #     device=device,
    #     **set_env_settings,
    # )

    env_make = lambda: GymWrapper(FlattenObservation(EVE({})), device=device)
    pe.envs = ParallelEnv(cfg.num_envs, env_make)

    steps_per_update = cfg.num_steps * cfg.num_envs
    num_updates = int(cfg.num_env_steps) // steps_per_update
    pe.envs.reset()
    sample_env = FlattenObservation(EVE({}))
    sample_env.reset()

    cfg.obs_shape = sample_env.observation_space.shape
    cfg.action_dim = get_size_for_space(sample_env.action_space)
    cfg.action_is_discrete = isinstance(cfg.action_dim, spaces.Discrete)
    cfg.total_num_updates = num_updates

    logger: Logger = hydra_instantiate(cfg.logger, full_cfg=cfg)

    storage: RolloutStorage = hydra_instantiate(cfg.storage, device=device)
    policy: Policy = hydra_instantiate(cfg.policy)
    policy = policy.to(device)
    updater = hydra_instantiate(cfg.policy_updater, policy=policy, device=device)


    start_update = 0
    if cfg.load_checkpoint is not None:
        ckpt = torch.load(cfg.load_checkpoint)
        updater.load_state_dict(ckpt["updater"], should_load_opt=cfg.resume_training)
        if cfg.load_policy:
            policy.load_state_dict(ckpt["policy"])
        if cfg.resume_training:
            start_update = ckpt["update_i"] + 1

    recording_env = FlattenObservation(EVE(cfg.env.env_settings.params.config))    
    storage.init_storage( pe.envs.reset()["observation"])

    for update_i in range(start_update, num_updates):
        is_last_update = update_i == num_updates - 1 
        frames = []
        logging = cfg.log_interval != -1 and (
            update_i % cfg.log_interval == 0 or is_last_update)
        for step_idx in range(cfg.num_steps):
            
            with torch.no_grad():
                act_data = policy.act(
                    storage.get_obs(step_idx),
                    storage.recurrent_hidden_states[step_idx],
                    storage.masks[step_idx],
                )
            # if logging:
            #     frames.append(envs.render(mode="rgb_array")[0])
            step_act = TensorDict({"action" : act_data["actions"]}, batch_size=(cfg.num_envs), device=device)
            results =  pe.envs.step(step_act)                                                                                              
            next_obs, reward, done= results["next"]["observation"], results["next"]["reward"], results["next"]["done"]
            reward_components_names = ['odometry_x',
 'odometry_y',
 'odometry_roll',
 'odometry_pitch',
 'odometry_yaw',
 'left_contact_cost',
 'right_contact_cost',
 'velocity_left',
 'velocity_right',
 'left_phase_weight_speed',
 'right_phase_weight_speed',
 'left_phase_weight_force',
 'right_phase_weight_force',
 'friction']
            
            info = [{"episode":{k:float(v) for (k,v) in zip(reward_components_names,reward[i])}} for i in range(len(reward))]
            storage.insert(next_obs, reward, done, info, **act_data)
            logger.collect_env_step_info(info)

        updater.update(policy, storage, logger)

        storage.after_update()

        # print("SIZE OF STORAGE: ", getsizeof(storage))
        # print("SIZE OF POLICY: ", getsizeof(policy))
        # print("SIZE OF UPDATER: ", getsizeof(updater))
        # print("SIZE OF ENV: ", getsizeof(envs))
        # print("SIZE OF LOGGER: ", getsizeof(logger))
        # print("SIZE OF EVALUATOR: ", getsizeof(recording_env))
        # print("Logger size")
        if cfg.eval_interval != -1 and (
            update_i % cfg.eval_interval == 0 or is_last_update
        ):
            pass
        if logging:
            logger.interval_log(update_i, steps_per_update * (update_i + 1))

            recording_storage: RolloutStorage = hydra_instantiate(cfg.recording_storage, device=device)
            recording_storage.init_storage(torch.from_numpy(recording_env.reset()))
            step = 0 
            frames = []
            while True: 
                with torch.no_grad():
                    act_data = policy.act(
                        recording_storage.get_obs(step_idx),
                        recording_storage.recurrent_hidden_states[step_idx],
                        recording_storage.masks[step_idx],
                    )
                step_act = TensorDict({"action" : act_data["actions"]}, batch_size=(1), device=device)
                next_obs, reward, done, info  = recording_env.step(act_data["actions"][0].numpy())
                recording_storage.insert(torch.from_numpy(next_obs), reward.view(1,-1), torch.tensor([done]), [info], **act_data)
                frames.append(recording_env.render(mode="rgb_array"))
                step += 1
                
                if(done or step > cfg.horizon):
                    break
            
            media.write_video(
            osp.join(logger.save_path, f"video.{update_i}.mp4"), frames, fps=30
        )
            Viewer.close()
            print(f"Saved to {osp.join(logger.save_path, f'video.{update_i}.mp4')}")

        if cfg.save_interval != -1 and (
            (update_i + 1) % cfg.save_interval == 0 or is_last_update
        ):
            save_name = osp.join(logger.save_path, f"ckpt.{update_i}.pth")
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "updater": updater.state_dict(),
                    "update_i": update_i,
                },
                save_name,
            )
            print(f"Saved to {save_name}")

    logger.close()
