from ray.rllib.algorithms.ppo import PPOConfig 
import mediapy as media
from ray.tune.registry import register_env
from snake_env_ray import SnakeEnv, MyCallbacks
import yaml 
from pathlib import Path
import os 
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument("--cleanrun", action="store_true",default=False)

cleanrun = argparser.parse_args().cleanrun

load = False
if not cleanrun:
    runs = Path.glob( Path.home() / "ray_results" ,"PPO_snake-v0*")

    for run in list(runs)[::-1]:
        print("Loading run", run)
        checkpoints = Path.glob(run, "checkpoint_*")
        for checkpoint in list(checkpoints)[::-1]:
            print("Loading checkpoint", checkpoint)
            load = True
            break
        if load:
            break
        else:
            print("No checkpoint found here")
    

register_env("snake-v0", lambda config: SnakeEnv(config))

configs = yaml.safe_load(open("SnakeDeepQ.yaml"))["env"]

trainer= PPOConfig().from_dict({"callbacks": MyCallbacks, "observation_filter" : "MeanStdFilter" })

trainer = trainer.environment(env = "snake-v0", env_config=configs)\
                     .resources(num_gpus=1, num_cpus_per_worker=1)\
                     .rollouts(num_rollout_workers=16, recreate_failed_workers= True )\
                     .training(clip_param= 0.3,gamma = 0.99 , kl_coeff = 0.3,model ={
  'fcnet_hiddens' : [2048, 2048],
  'fcnet_activation': 'swish',
  'vf_share_layers': False,
  'free_log_std': True }, train_batch_size= 5000 ,sgd_minibatch_size = 1000,num_sgd_iter= 5
)
trainer = trainer.build()
if(load):
    trainer.restore(str(checkpoint))
# trainer.load_checkpoint("~/ray_results/PPO_snake-v0_2023-06-03_15-54-10lqaromzd/checkpoint_000800")

fps = 30

# Training loop
max_test_i = 0
checkpoint_frequency = 50
simulation_frequency = 20
env = SnakeEnv(config=configs)
env.render_mode = "rgb_array"
sim_dir = "ray sims"
# Create sim directory if it doesn't exist
if not os.path.exists(sim_dir):
    os.makedirs(sim_dir)

# Find the latest directory named test_i in the sim directory
latest_directory = max(
    [int(d.split("_")[-1]) for d in os.listdir(sim_dir) if d.startswith("test_")],
    default=0,
)
max_test_i = latest_directory + 1

# Create folder for test
test_dir = os.path.join(sim_dir, "test_{}".format(max_test_i))
os.makedirs(test_dir, exist_ok=True)

# Define video codec and framerate
fps = 10

# Set initial iteration count
i = trainer.iteration if hasattr(trainer, "iteration") else 0

while True:
    # Train for one iteration
    result = trainer.train()
    #get the current filter params
    i += 1
    print(
        "Episode {} Reward Mean {}".format(
            i,
            result["episode_reward_mean"]
        )
    )

    # Save model every 10 epochs
    if i % checkpoint_frequency == 0:
        checkpoint_path = trainer.save()
        print("Checkpoint saved at", checkpoint_path)

    # Run a test every 20 epochs
    if i % simulation_frequency == 0:
        # make a steps counter
        steps = 0

        # Run test
        video_path = os.path.join(test_dir, "sim_{}.mp4".format(i))
        filterfn = trainer.workers.local_worker().filters["default_policy"]
        env.reset()
        obs = env.reset()[0]
        done = False
        frames = []

        while not done:
            # Increment steps
            steps += 1
            obs = filterfn(obs)
            action = trainer.compute_single_action(obs)
            obs, _, done, _, _ = env.step(action)
            frame = env.render()
            frames.append(frame)

        # Save video
        media.write_video(video_path, frames, fps=fps)
        print("Test saved at", video_path)
        # Increment test index
        max_test_i += 1
