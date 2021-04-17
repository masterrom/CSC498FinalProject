import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as fn
import numpy as np
import tqdm
import wandb
from collections import deque
from dataclasses import dataclass
from typing import Any
from random import sample

wandb.init(project='Cartpole', entity='masterrom')

@dataclass
class Data:
    state: Any
    action: int
    reward: float
    nextState: Any
    done: bool

class ReplayBuffer():
    def __init__(self, bufferSize=10000):
        self.bufferSize = bufferSize
        self.buffer = deque(maxlen=self.bufferSize)

    def insert(self, data):
        self.buffer.append(data)

    def sample(self, numSample):
        assert numSample <= len(self.buffer)
        return sample(self.buffer, numSample)

class Agent():
    def __init__(self, observation_dim, params = None, action_bounds = None):
        pass

    def __call__(self, obs):
        return self.act(obs)

class DQN(Agent):

    def __init__(self, observation_dim, action_dim, buffer, gamma=0.99):

        self.replayBuffer = buffer

        self.actions = action_dim
        self.obs_dim = observation_dim

        self.q = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.ReLU(),
            # nn.Linear(32, 32),
            # nn.ReLU(),
            nn.Linear(256, self.actions)
        ).double()
        self.q_target = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.ReLU(),
            # nn.Linear(32, 32),
            # nn.ReLU(),
            nn.Linear(256, self.actions)
        ).double()

        self.optim = torch.optim.Adam(self.q.parameters(), lr=1e-4)
        self.N_STEPS = 700
        self.BATCH_SIZE = 512
        self.N_EPOCHS = 200
        self.gamma = gamma

    def compute_target(self, states, rewards):
        """
        states: torch.Tensor of size (batch, obs_dim) with s' from the dataset
        rewards: torch.Tensor of size (batch) with single step rewards (float)

        returns torch.Tensor of size (batch) with the 1-step Q learning target

        Target = R_{t+1} + gamma * max{a}Q(S_{t+1}, a)
        """

        with torch.no_grad():
            qVals = torch.zeros(states.shape[0], dtype=torch.float)  # Initializing QVals
            for i in range(states.shape[0]):  # Looping through batch
                qs = self.q_target(torch.tensor(states[i]))  # Computing Q values from the target network
                maxActionIndex = torch.argmax(qs)  # Selecting the best action
                qVals[i] = qs[maxActionIndex]  # selecting the max qValues from the dataset

            qTargetVals = rewards + (self.gamma * qVals.numpy())  # Computing the final Target

        return qTargetVals

    def loss(self, states, actions, target):
        """
        states: torch.Tensor of size (batch, obs_dim) with s from the dataset
        actions: torch.Tensor of size (batch, 1) with action from the dataset
        target: torch.Tensor of size (batch) with computed target (see self.compute_target)

        returns torch.Tensor of size (1) with squared Q error

        Hint: you will need the torch.gather function
        """

        # Computing the Q values from the policy network, then selecting the Q values of actions specificed in
        # in the action matrix
        actions = np.reshape(actions, (actions.shape[0], 1)) # Puting actions into a single column =
        policyQ = self.q(torch.tensor(states)).gather(1, torch.tensor(actions))
        target = np.reshape(target, (target.shape[0], 1))  # Reshaping the target to be in a single column

        # import ipdb; ipdb.set_trace()

        loss = torch.square(torch.tensor(target) - policyQ)  # Computing the squared loss for each time step

        return torch.sum(loss)  # Summing the loss

    def __call__(self, state):
        """
        states: np.array of size (obs_dim,) with the current state

        returns np.array of size (1,) with the optimal action
        """

        state = torch.from_numpy(state).view(1, -1)
        actions = self.q(state)
        return torch.argmax(actions)

    def computeTrainingStep(self, transitions):

        nextStates = np.stack([s.nextState for s in transitions])
        states = np.stack([s.state for s in transitions])
        actions = np.stack([np.array(s.action) for s in transitions])
        rewards = np.stack([s.reward for s in transitions])

        # import ipdb;
        # ipdb.set_trace()

        target = self.compute_target(nextStates, rewards)
        loss = self.loss(states, actions, target)

        return loss

    def train(self, task):

        # Collect initial Data
        self.fillBuffer(task)
        for i in tqdm.tqdm(range(self.N_EPOCHS)):
            epoch_losses = self.train_epoch(task, False)
            # losses.extend(epoch_losses)

            self.getAverageReward(task, i)


            torch.save({
                'targetModel_state_dict': self.q_target.state_dict(),
                'qModel_state_dict': self.q.state_dict(),
            }, "./targetModels/modelTimeStamp-" + str(i) + ".pt")
        # return losses


    def train_epoch(self, task, random=False):

        lastObs = task.reset()
        for i in range(self.N_STEPS):

            act = np.random.choice([self(lastObs), np.random.randint(self.actions)], p=[0.9, 0.1])
            if random:
                act = np.random.randint(self.actions)
            obs, rew, done, info = task.step(act)
            exper = Data(lastObs, act, rew, obs, done)
            self.replayBuffer.insert(exper)
            lastObs = obs

            samples = self.replayBuffer.sample(self.BATCH_SIZE)
            loss = self.computeTrainingStep(samples)

            wandb.log({"Loss": loss, "title": "DQN with Replay Buffer Loss"})

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        with torch.no_grad():
            print("Updating Target Model")
            self.q_target.load_state_dict(self.q.state_dict())


    def fillBuffer(self, task, random=False):
        # Do not modify
        lastObs = task.reset()
        for exp in range(self.replayBuffer.bufferSize):

            act = np.random.choice([self(lastObs), np.random.randint(self.actions)], p=[0.9, 0.1])
            if random:
                act = np.random.randint(self.actions)

            obs, rew, done, info = task.step(act)

            exper = Data(lastObs, act, rew, obs, done)
            self.replayBuffer.insert(exper)
            lastObs = obs

    def getAverageReward(self, task, epoch):
        rewards = np.zeros((100, 100))

        for run in range(100):
            obs = task.reset()
            for step in range(100):
                act = self(obs)
                obs, rew, done, info = task.step(act.item())
                rewards[run, step] = rew

        avgReward = rewards.sum(1).std()
        wandb.log({"avgReward": avgReward, "epoch": epoch})


def testModel(path):
    checkpoint = torch.load(path)
    task = gym.make("CartPole-v1")
    replayBuffer = ReplayBuffer()
    agent = DQN(task.observation_space.shape[-1], task.action_space.n, replayBuffer)

    agent.q_target.load_state_dict(checkpoint['targetModel_state_dict'])

    # obs = task.reset()
    # while True:
    #     act = agent(obs)
    #     obs, rew, done, info = task.step(act.item())
    #     task.render()
    #     if done:
    #         print("Reseting")
    #         obs = task.reset()
    #
    rewards = np.zeros((100, 100))

    for run in range(100):
        obs = task.reset()
        for step in range(100):
            act = agent(obs)
            obs, rew, done, info = task.step(act.item())
            rewards[run, step] = rew
            task.render()

    print("Average return: {}".format(rewards.sum(1).mean()))
    print("Standard deviation: {}".format(rewards.sum(1).std()))

def training(path=None):
    task = gym.make("CartPole-v1")
    replayBuffer = ReplayBuffer()
    agent = DQN(task.observation_space.shape[-1], task.action_space.n, replayBuffer)

    if path != None:
        checkpoint = torch.load(path)
        agent.q.load_state_dict(checkpoint['qModel_state_dict'])
        agent.q_target.load_state_dict(checkpoint['targetModel_state_dict'])

    agent.train(task)

    # Final Benchmarking
    print("Final Benchmarking")

    rewards = np.zeros((100, 100))

    for run in range(100):
        obs = task.reset()
        for step in range(100):
            act = agent(obs)
            obs, rew, done, info = task.step(act.item())
            rewards[run, step] = rew
            task.render()

    print("Average return: {}".format(rewards.sum(1).mean()))
    print("Standard deviation: {}".format(rewards.sum(1).std()))


if __name__ == '__main__':

    # Model at episode 38
    # Model at episode 68
    training("./targetModels-Run1/modelTimeStamp-74.pt")
    # testModel("./targetModels/modelTimeStamp-68.pt")

