#!/usr/bin/env python3
import gym
#from pettingzoo.sisl import multiwalker_v9

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from keras.models import load_model

from tensorboardX import SummaryWriter


MAX_BATCH_EPISODES = 10
MAX_BATCH_STEPS = 1000
NOISE_STD = 0.0000001
LEARNING_RATE = 0.00000001
BINS = 12

PLAYER_1_ID = 'walker_0'
PLAYER_2_ID = 'walker_1'

np.random.seed(42)


class Net(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(obs_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, X_input):
        X = torch.relu(self.fc1(X_input))
        X = torch.relu(self.fc2(X))
        X = torch.relu(self.fc3(X))
        output = torch.tanh(self.fc4(X))
        return output

def get_action(net, obs_v):
    #mu_v, var_v, _ = net(obs_v)
    #mu = mu_v.data.numpy()
    #sigma = torch.sqrt(var_v).data.numpy()
    actions = net(obs_v).data.numpy()[0]
    #actions = np.random.normal(mu, sigma)
    actions = np.clip(actions, -1, 1)
    #actions = np.reshape(actions,(4,))
    return actions
    

def evaluate(env, net):
    obs = env.reset()
    reward = 0.0
    steps = 0
    while True:
        obs_v = torch.FloatTensor([obs])
        acts = get_action(net,obs_v)
        #act_prob = net(obs_v)
        #acts = act_prob.max(dim=1)[1]
        #esto = acts.data.numpy()[0]
        #acts=[0.5, -1., 0.5, 0.5]
        #env.step(acts.tolist())
        obs, r, done, _ = env.step(acts)
        _ = env.render(mode="human")
        #obs, r, done, _ , _= env.last()
        reward += r
        steps += 1
        if done:
            break
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
    ##LOAD MODEL
    actor = load_model("BipedalWalkerHardcore-v3_actor.h5", compile=False)
    #actor.summary()
    weights = actor.get_weights()


    
    writer = SummaryWriter(comment="-Walker_gym-es")
    env = gym.make("BipedalWalker-v3")
    #env = multiwalker_v9.env(n_walkers=1, position_noise=1e-3, angle_noise=1e-3, forward_reward=10.0, terminate_reward=-100.0, fall_reward=-10.0, shared_reward=True,
    #                        terminate_on_fall=True, remove_on_fall=True, terrain_length=200, max_cycles=500,render_mode="human")
    #env = pursuit_v4.env(render_mode="human")
    #env.reset(seed=42)
    #net = Net(env.observation_space(PLAYER_1_ID).shape[0], env.action_space(PLAYER_1_ID).shape[0])
    #esto = env.observation_space.shape[0]
    #esto1 = env.action_space.shape[0]
    net = Net(env.observation_space.shape[0], env.action_space.shape[0])
    print(net)  
    net.fc1.weight.data = torch.from_numpy(np.transpose(weights[0]))
    net.fc1.bias.data = torch.from_numpy(weights[1])
    net.fc2.weight.data = torch.from_numpy(np.transpose(weights[2]))
    net.fc2.bias.data = torch.from_numpy(weights[3])
    net.fc3.weight.data = torch.from_numpy(np.transpose(weights[4]))
    net.fc3.bias.data = torch.from_numpy(weights[5])
    net.fc4.weight.data = torch.from_numpy(np.transpose(weights[6]))
    net.fc4.bias.data = torch.from_numpy(weights[7])
    
    print(net.fc1.weight)
    print("------------------------")
    print(weights[0])
    
    step_idx = 0
    steps2 = 0
    contador=0
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
        if m_reward > 100:
            #print("Solved in %d steps" % step_idx)
            contador+=1
            if contador>500:
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