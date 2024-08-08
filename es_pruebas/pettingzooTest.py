from pettingzoo.sisl import multiwalker_v9
import pettingzoo.utils.wrappers as wrappers
#from supersuit import frame_stack_v1, resize_v0, frame_skip_v0, agent_indicator_v0
from gym import wrappers
#import supersuit
import random
IMAGE_SIZE = 84
PLAYER_1_ID = 'walker_0'
PLAYER_2_ID = 'walker_1'
PLAYER_3_ID = 'walker_2'

env = multiwalker_v9.env(n_walkers=2, position_noise=1e-3, angle_noise=1e-3, forward_reward=10.0, terminate_reward=-100.0, fall_reward=-10.0, shared_reward=True,
                             terminate_on_fall=True, remove_on_fall=True, terrain_length=200, max_cycles=500,render_mode="human")
#env = wrappers.ResizeObservation(env, IMAGE_SIZE)
#env = frame_skip_v0(env, 4)
#env = supersuit.resize_v0(env, IMAGE_SIZE,IMAGE_SIZE)
#env = supersuit.frame_stack_v1(env,4)
#env = wrappers.ResizeObservation(env, (IMAGE_SIZE, IMAGE_SIZE))
#env = wrappers.GrayScaleObservation(env)
#env = wrappers.FrameStack(env,4)
obs = env.reset() 
#esto = env.observe(PLAYER_1_ID)
#esto = env.state()
#obs1 = obs[PLAYER_1_ID]
num_agents = len(env.possible_agents)
num_actions = env.action_space(env.possible_agents[0]).shape
observation_size = env.observation_space(env.possible_agents[0]).shape

completed = 0
episodes = 10

while completed < episodes:
    env.reset()
    #r = {PLAYER_1_ID:0.0,PLAYER_2_ID: 0.0,PLAYER_3_ID: 0.0}
    for agent in env.agent_iter(100000):
        observation, reward, done, info, _ = env.last()
        #r[agent]+=env.rewards[agent]
        if done or info:
            break
        else:
            action = env.action_space(agent).sample()
        #if env.rewards[agent]!=0:
            #print(agent,env.rewards[agent])
        env.step(action)
        env.render()
    completed+=1
    print(completed)
env.close()
