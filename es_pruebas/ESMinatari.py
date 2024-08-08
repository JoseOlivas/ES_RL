#!/usr/bin/env python3
import gym
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f

from tensorboardX import SummaryWriter

from minatar import Environment
MAX_BATCH_EPISODES = 100
MAX_BATCH_STEPS = 10000
NOISE_STD = 0.01
LEARNING_RATE = 0.001

device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self,obs_size,action_size):
        super(Net, self).__init__()

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(obs_size, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        # Output layer:
        self.output = nn.Linear(in_features=128, out_features=action_size)

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        # Rectified output from the first conv layer
        x = f.relu(self.conv(x))

        # Rectified output from the final hidden layer
        x = f.relu(self.fc_hidden(x.view(x.size(0), -1)))

        # Returns the output from the fully-connected linear layer
        return self.output(x)
    

def get_state(s):
    return (torch.tensor(s,device=device).permute(2, 0, 1)).unsqueeze(0).float()


def evaluate(env,Qmodel):
    Qmodel.eval()
    s = env.reset()
    state = get_state(env.state())
    reward = 0.0
    steps = 0
    done = False
    while not done:
        state = torch.Tensor(state).to(device)
        with torch.no_grad():
            action = Qmodel(state).max(1)[1].view(1, 1)
        r, done = env.act(action)
        reward+=r
        steps+=1
        state = get_state(env.state())
    return reward, steps

def sample_noise(net):
    pos = []
    neg = []
    for p in net.parameters():
        noise = np.random.normal(size=p.data.size())
        noise_t = torch.FloatTensor(noise)
        pos.append(noise_t)
        neg.append(-noise_t)
    return pos, neg


def eval_with_noise(env, net, noise):
    old_params = net.state_dict()
    for p, p_n in zip(net.parameters(), noise):
        p.data += NOISE_STD * p_n
    r, s = evaluate(env, net)
    net.load_state_dict(old_params)
    return r, s


def train_step(net, batch_noise, batch_reward, writer, step_idx):
    weighted_noise = None
    norm_reward = np.array(batch_reward)
    norm_reward -= np.mean(norm_reward)
    s = np.std(norm_reward)
    if abs(s) > 1e-6:
        norm_reward /= s

    for noise, reward in zip(batch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n
    m_updates = []
    for p, p_update in zip(net.parameters(), weighted_noise):
        update = p_update / (len(batch_reward) * NOISE_STD)
        p.data += LEARNING_RATE * update
        m_updates.append(torch.norm(update))
    writer.add_scalar("update_l2", np.mean(m_updates), step_idx)


if __name__ == "__main__":
    writer = SummaryWriter(comment="-breakout-es")
    #env = gym.make("CartPole-v0")
    env = Environment("breakout")
    net = Net(env.state_shape()[2],env.num_actions())
    print(net)
    steps2=0
    step_idx = 0
    while True:
        t_start = time.time()
        batch_noise = []
        batch_reward = []
        batch_steps = 0
        for _ in range(MAX_BATCH_EPISODES):
            noise, neg_noise = sample_noise(net)
            batch_noise.append(noise)
            batch_noise.append(neg_noise)
            reward, steps = eval_with_noise(env, net, noise)
            batch_reward.append(reward)
            batch_steps += steps
            reward, steps = eval_with_noise(env, net, neg_noise)
            batch_reward.append(reward)
            batch_steps += steps
            steps2+=steps
            if batch_steps > MAX_BATCH_STEPS:
                break

        step_idx += 1
        m_reward = np.mean(batch_reward)
        if m_reward > 199:
            print("Solved in %d steps" % step_idx)
            break

        train_step(net, batch_noise, batch_reward,
                   writer, step_idx)
        writer.add_scalar("reward_mean", m_reward, step_idx)
        writer.add_scalar("reward_std", np.std(batch_reward),
                          step_idx)
        writer.add_scalar("reward_max", np.max(batch_reward),
                          step_idx)
        writer.add_scalar("batch_episodes", len(batch_reward),
                          step_idx)
        writer.add_scalar("batch_steps", batch_steps, step_idx)
        speed = batch_steps / (time.time() - t_start)
        writer.add_scalar("speed", speed, step_idx)
        print("%d: reward=%.2f, speed=%.2f f/s, step=%.2f" % (
            step_idx, m_reward, speed,steps2))

    pass