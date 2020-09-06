import gym
import gym_Aircraft

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# check environment
# from stable_baselines3.common.env_checker import check_env
# env=gym.make("acav-v0")
# check_env(env)

# num_gpus=4
size=5
batch_size=128
env=gym.make("acav-v0")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """transition """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Modelstr1(torch.nn.Module):
    def __init__(self, num_of_nodes):
        super(Modelstr1, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(5, num_of_nodes),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        out1 = self.fc1(x)
        return out1


class Modelstr2(torch.nn.Module):
    def __init__(self, num_of_nodes, past_nodes):
        super(Modelstr2, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(past_nodes, num_of_nodes),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        out1 = self.fc1(x)
        return out1


class Model(torch.nn.Module):
    def __init__(self, num_of_nodes):
        super(Model, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(num_of_nodes, num_of_nodes),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        out1 = self.fc1(x)
        return out1


class Modelfin(torch.nn.Module):
    def __init__(self, num_of_nodes):
        super(Modelfin, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(num_of_nodes, 3),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        out1 = self.fc1(x)
        return out1


class Modelinit(torch.nn.Module):
    def __init__(self):
        super(Modelinit, self).__init__()

    def forward(self, x):
        out1 = x.view(x.size(0), -1)
        return out1


num_of_lays1 = 2
num_of_nodes1 = 40
num_of_lays2 = 2
num_of_nodes2 = 40
num_of_lays3 = 2
num_of_nodes3 = 40
model_start1 = Modelstr1(num_of_nodes1)
model_start2 = Modelstr2(num_of_nodes2, num_of_nodes1)
model_start3 = Modelstr2(num_of_nodes3, num_of_nodes2)
model_re1 = Model(num_of_nodes1)
model_re2 = Model(num_of_nodes2)
model_re3 = Model(num_of_nodes3)
model_fin = Modelfin(num_of_nodes3)
model_temp1 = Modelinit()
model_temp2 = Modelinit()
model_temp3 = Modelinit()

for i in range(num_of_lays1):
    model_temp1 = nn.Sequential(model_temp1, model_re1)
for i in range(num_of_lays2):
    model_temp2 = nn.Sequential(model_temp2, model_re2)
for i in range(num_of_lays3):
    model_temp3 = nn.Sequential(model_temp3, model_re3)

gamma=0.999
eps_start=0.9
eps_end=0.05
eps_decay=200
target_update=10

n_actions=env.action_space.n

policy_net = nn.Sequential(model_start1, model_temp1, model_start2, model_temp2,
                                         model_start3, model_temp3, model_fin).cuda()

# policy_net.load_state_dict(torch.load("train_res"))
target_net=nn.Sequential(model_start1, model_temp1, model_start2, model_temp2,
                                         model_start3, model_temp3, model_fin).cuda()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer=optim.RMSprop(policy_net.parameters())
memory=ReplayMemory(10000)

steps_done=0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1) # error
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
episode_durations=[]

def plot_durations():
    plt.figure(2)
    plt.clf()
    duration_t=torch.tensor(episode_durations,dtype=torch.float)
    plt.title("training...")
    plt.xlabel("episode")
    plt.ylabel("duration")
    plt.plot(duration_t.numpy())
    if len(duration_t)>=100:
        means=duration_t.unfold(0,100,1).mean(1).view(-1)
        means=torch.cat((torch.zeros(99),means))
        plt.plot(means.numpy())
    plt.pause(0.001)

def optimize_model():
    if len(memory)<batch_size:
        return
    transitions=memory.sample(batch_size)
    batch=Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * gamma) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 50
for i_episode in range(num_episodes):
    env.reset()
    state=torch.tensor(np.zeros(size),device=device,dtype=torch.float)
    for t in count():
        action = select_action(state)
        st, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device,dtype=torch.float)

        # new state check

        if not done:
            next_state=torch.tensor(st,device=device,dtype=torch.float)
        else:
            next_state=None
        memory.push(state,action,next_state,reward)

        state=next_state

        optimize_model()
        if done:
            episode_durations.append(t+1)
            plot_durations()
            break

    if i_episode%target_update==0:
        target_net.load_state_dict(policy_net.state_dict())
print("complete")
env.render()
plt.show()

