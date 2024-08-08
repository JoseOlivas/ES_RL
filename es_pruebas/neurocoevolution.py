import time
import gym
from pettingzoo.atari import pong_v3
import pettingzoo.utils.wrappers as wrappers
#from supersuit import frame_stack_v1, resize_v0, frame_skip_v0, agent_indicator_v0
from gym import wrappers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import cv2
import random

from tensorboardX import SummaryWriter
IMAGE_SIZE = 84
PLAYER_1_ID = 'first_0'
PLAYER_2_ID = 'second_0'
MAX_BATCH_EPISODES = 100
MAX_BATCH_STEPS = 10000
NOISE_STD = 0.01
LEARNING_RATE = 0.001

device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self,obs_size,action_size):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(obs_size, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.bn1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.bn2(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.bn3(x)
        x = x.view(x.size(0), -1) 
        x = nn.functional.relu(self.fc1(x))
        x = self.bn4(x)
        x = self.fc2(x)
        return nn.functional.softmax(x, dim=1)

class Agent:
    def __init__(self, env):
        self.env = env
        self.action_size = self.env.action_space(env.possible_agents[0]).n
        #self.obs_size = self.env.observation_space(env.possible_agents[0]).shape
        
        self.ROWS = IMAGE_SIZE
        self.COLS = IMAGE_SIZE
        self.REM_STEP = 4
        ##REM_STEP + 2 PARA ONE-HOT AGENT_INDICATOR
        self.state_size = (self.REM_STEP+2, self.ROWS, self.COLS)
        self.image_memory = np.zeros(self.state_size)

        self.net = Net(self.state_size[0],self.action_size)
        self.old_net = Net(self.state_size[0],self.action_size)
        #print(self.net)

        self.states, self.actions, self.rewards = [], [], []
        self.scores, self.episodes, self.average = [], [], []

        self.writer = SummaryWriter(comment="-breakout-es")
    def sample_noise(self, net):
        pos = []
        neg = []
        for p in net.parameters():
            noise = np.random.normal(size=p.data.size())
            noise_t = torch.FloatTensor(noise)
            pos.append(noise_t)
            neg.append(-noise_t)
        return pos, neg
    
    def imshow(self, image, rem_step=0):
        cv2.imshow(str(rem_step), image[rem_step,...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return
        
    def GetState(self, frame, agent=PLAYER_1_ID):
        # croping frame to 80x80 size
        frame_cropped = frame[35:195:2, ::2,:]
        if frame_cropped.shape[0] != self.COLS or frame_cropped.shape[1] != self.ROWS:
            # OpenCV resize function 
            frame_cropped = cv2.resize(frame, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)

        frame_rgb = 0.299*frame_cropped[:,:,0] + 0.587*frame_cropped[:,:,1] + 0.114*frame_cropped[:,:,2]

        frame_rgb[frame_rgb < 100] = 0
        frame_rgb[frame_rgb >= 100] = 255

        #frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2GRAY)     

        new_frame = np.array(frame_rgb).astype(np.float32) / 255.0

        self.image_memory = np.roll(self.image_memory, 1, axis = 0)

        self.image_memory[0,:,:] = new_frame
        #self.imshow(self.image_memory,0)

        unos = np.ones((IMAGE_SIZE,IMAGE_SIZE))
        zeros = np.zeros((IMAGE_SIZE,IMAGE_SIZE))
        #obs_spaces = [env.observation_space(agent) for agent in env.possible_agents]
        #print(len(obs_spaces))
        ###     UNCOMMENT       ###
        #Nueva función ->
        #([0,1] if agente1 else [1,0])
        if agent==PLAYER_1_ID:
            self.image_memory[4,:,:] = zeros
            self.image_memory[5,:,:] = unos
        elif agent==PLAYER_2_ID:
            self.image_memory[4,:,:] = unos
            self.image_memory[5,:,:] = zeros
        
        return (torch.tensor(self.image_memory,device=device)).unsqueeze(0).float()
    
    def reset(self):
        frame = self.env.reset()
        #print(self.env.possible_agents)
        frame = self.env.observe(PLAYER_1_ID)
        for i in range(self.REM_STEP):
            state = self.GetState(frame)
        return state
            
    def last(self,agent):
        flag = False
        observation, reward, done, info, _ = self.env.last()
        #if reward!=0:
            #print("ojo")
        if done:
            flag=True
        #convertior observation según jugador
        obs = self.GetState(observation,agent)
        return obs, reward, flag, done, 0

    def evaluate(self, env, net, elite): #Siempre net(1) vs elite(2)
        net.eval()
        elite.eval()
        redes = {PLAYER_1_ID:net, PLAYER_2_ID:elite}
        self.reset()
        #state = self.reset()
        #state = self.get_state(env.state())
        r = {PLAYER_1_ID:0.0, PLAYER_2_ID: 0.0}
        steps = 0
        done = False
        while not done:
            for agent in env.agent_iter(5000):
                negagent = (PLAYER_2_ID if agent == PLAYER_1_ID else PLAYER_1_ID)
                #_ , _ , done, info, _ = self.env.last()
                state, reward, done, info , _ = self.last(agent)
                r[agent]+=reward
                r[negagent]-=reward
                
                if abs(r[PLAYER_1_ID])>20 or abs(r[PLAYER_2_ID])>20:
                    return r[PLAYER_1_ID],r[PLAYER_2_ID], steps
                else:
                    state = torch.Tensor(state).to(device)
                    with torch.no_grad():
                        action = redes[agent](state).max(1)[1].view(1, 1).item()#Hacer arreglo de redes
                        #tal que net[agente](state) sirva para ambos.
                self.env.step(action)
                steps+=1
        return r[PLAYER_1_ID],r[PLAYER_2_ID], steps

    def eval_with_noise(self, env, elite, net, noise):
        old_params = net.state_dict()
        elite.load_state_dict(old_params)
        for p, p_n in zip(net.parameters(), noise):
            p.data += NOISE_STD * p_n
        ran = random.random() 
        if ran<0.5: #Menor a 0.5 -> player1
            r1, _, s1 = self.evaluate(env, net, elite)
        else:
            _, r1, s1 = self.evaluate(env, elite, net)
        net.load_state_dict(old_params)
        #prom = (r1+r2)/2
        #sum = s1+s2
        return r1, s1, ran
    
    def train_step(self, net, batch_noise, batch_reward, writer, step_idx):
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

    def run(self):
        steps2=0
        step_idx = 0
        while True:
            t_start = time.time()
            batch_noise = []
            batch_reward = []
            batch_steps = 0
            reward = 0.0
            rand_count = 0.0
            for _ in range(MAX_BATCH_EPISODES):
                noise, neg_noise = self.sample_noise(self.net)
                batch_noise.append(noise)
                batch_noise.append(neg_noise)
                reward, steps, ran1 = self.eval_with_noise(self.env, self.old_net, self.net, noise)
                batch_reward.append(reward)
                batch_steps += steps
                rand_count+=round(ran1)
                reward, steps, ran2 = self.eval_with_noise(self.env, self.old_net, self.net, neg_noise)
                batch_reward.append(reward)
                batch_steps += steps
                steps2+=batch_steps
                rand_count+=round(ran2)
                if batch_steps > MAX_BATCH_STEPS:
                    break

            step_idx += 1
            m_reward = np.mean(batch_reward)
            rand_mean = rand_count/(len(batch_reward)/2)
            if step_idx > 10:
                #print("Solved in %d steps" % step_idx)
                break

            self.train_step(self.net, batch_noise, batch_reward,
                    self.writer, step_idx)
            #Agregar función de evaluación vs random 50 juegos
            self.writer.add_scalar("reward_mean", m_reward, step_idx)
            self.writer.add_scalar("reward_std", np.std(batch_reward),
                            step_idx)
            self.writer.add_scalar("reward_max", np.max(batch_reward),
                            step_idx)
            self.writer.add_scalar("batch_episodes", len(batch_reward),
                            step_idx)
            self.writer.add_scalar("batch_steps", batch_steps, step_idx)
            speed = batch_steps / (time.time() - t_start)
            self.writer.add_scalar("speed", speed, step_idx)
            print("%d: reward=%.2f, speed=%.2f f/s, step=%.2f, rand_mean=%.2f" % (
                step_idx, m_reward, speed,steps2, rand_mean))
        
if __name__ == "__main__":
    #env_name = 'PongDeterministic-v4'
    #env_name = 'pong_v3'
    env = pong_v3.env()
    agent = Agent(env)
    inicio = time.time()
    agent.run()
    fin = time.time()
    print("Tiempo 10 ejecuciones: ", fin-inicio)
    #agent.run()
