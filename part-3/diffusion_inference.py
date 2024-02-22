from push_t_env import PushTImageEnv
import collections 
from tqdm import tqdm
import numpy as np
from dataset import normalize_data, PushTImageDataset, unnormalize_data
from model import get_resnet, replace_bn_with_gn, ConditionalUnet1D
import os
import gdown
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from skvideo.io import vwrite
import time
from torch.profiler import profile, record_function, ProfilerActivity

vision_encoder = get_resnet('resnet18')
vision_encoder = replace_bn_with_gn(vision_encoder)
vision_feature_dim = 512
lowdim_obs_dim = 2
obs_dim = vision_feature_dim + lowdim_obs_dim
action_dim = 2
pred_horizon = 16
obs_horizon = 2
action_horizon = 8

noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
})

# for this demo, we use DDPMScheduler with 100 diffusion iterations
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

device = torch.device('cuda')
_ = nets.to(device)

num_epochs = 100

ckpt_path = "pusht_vision_100ep.ckpt"
if not os.path.isfile(ckpt_path):
    id = "1XKpfNSlwYMGaF5CncoFaLKCDTWoLAHf1&confirm=t"
    gdown.download(id=id, output=ckpt_path, quiet=False)

state_dict = torch.load(ckpt_path, map_location='cuda')
nets.load_state_dict(state_dict)
print('Pretrained weights loaded.')

dataset_path = "pusht_cchi_v7_replay.zarr.zip"
if not os.path.isfile(dataset_path):
    id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
    gdown.download(id=id, output=dataset_path, quiet=False)
dataset = PushTImageDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)

stats = dataset.stats

# limit enviornment interaction to 200 steps before termination
max_steps = 200
env = PushTImageEnv()
torch.set_default_device('cuda')
env.seed(np.random.randint(1, 100001))

# get first observation
obs, info = env.reset()

# keep a queue of last 2 steps of observations
obs_deque = collections.deque(
    [obs] * obs_horizon, maxlen=obs_horizon)
# save visualization and rewards
imgs = [env.render(mode='rgb_array')]
rewards = list()
done = False
step_idx = 0
total_time = 0
vision_encoder_time = 0 
unet_time = 0
denoising_time = 0
total_start = time.time() 

with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
    while not done:
        B = 1
        # stack the last obs_horizon number of observations
        images = np.stack([x['image'] for x in obs_deque])
        agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

        # normalize observation
        nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
        # images are already normalized to [0,1]
        nimages = images

        # device transfer
        nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
        # (2,3,96,96)
        nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
        # (2,2)

        # infer action
        with torch.no_grad():
            # get image features
            start_time = time.time()
            image_features = nets['vision_encoder'](nimages)
            vision_encoder_time += time.time() - start_time
            # (2,512)

            # concat with low-dim observations
            obs_features = torch.cat([image_features, nagent_poses], dim=-1)

            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn((1, pred_horizon, action_dim), device='cuda')  
            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                start_time_unet = time.perf_counter()
                noise_pred = nets['noise_pred_net'](
                    noisy_action,
                    k,
                    obs_cond
                )
                torch.cuda.synchronize()
                unet_time += time.perf_counter() - start_time_unet
                
                start_time_denoise = time.perf_counter()
                noisy_action = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,    
                    sample=noisy_action
                ).prev_sample
                torch.cuda.synchronize()
                denoising_time += time.perf_counter() - start_time_denoise

        # unnormalize action
        naction = noisy_action.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=stats['action'])
        
        # only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[start:end,:]
        # (action_horizon, action_dim)

        # execute action_horizon number of steps
        # without replanning
        
        for i in range(len(action)):
            # stepping env
            obs, reward, _, _, info = env.step(action[i])
            # save observations
            obs_deque.append(obs)
            # and reward/vis
            rewards.append(reward)
            imgs.append(env.render(mode='rgb_array'))

            # update progress bar
            step_idx += 1
            pbar.update(1)
            pbar.set_postfix(reward=reward)
            if step_idx > max_steps:
                done = True
            if done:
                break

total = time.time() - total_start
print(f'Time spent in vision encoder: {vision_encoder_time}')
print(f'Time spent in u-net: {unet_time}')
print(f'Time spent in denoising: {denoising_time}')
print(f'Time spend outside of that: {total - vision_encoder_time - unet_time - denoising_time}')
print(f'Total time: {total}')

print(f'Tracing forward pass through U-Net.')
with torch.no_grad():
    with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/diffusion'), with_stack=True) as prof:
        noisy_pred = nets['noise_pred_net'](noisy_action, k, obs_cond)
        noisy_action = noise_scheduler.step(
            model_output=noisy_pred,
            timestep=k,
            sample=noisy_action
        ).prev_sample
