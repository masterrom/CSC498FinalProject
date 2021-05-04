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
from random import sample, random

wandb.init(project='Cartpole', entity='masterrom')

# Data type class used for storing transitions into the replay buffer
@dataclass
class Data:
    state: Any
    action: int
    reward: float
    nextState: Any
    done: bool

# Class to represents the Experience-Replay, with the functionality of inserting and sampling
# from the buffer. Using a Dequeue as it is faster to insert new elements than list
class ReplayBuffer():
    def __init__(self, bufferSize=10000):
        self.bufferSize = bufferSize
        self.buffer = deque(maxlen=self.bufferSize)

    def insert(self, data):
        self.buffer.append(data)

    def sample(self, numSample):
        assert numSample <= len(self.buffer)
        return sample(self.buffer, numSample)

# Simple agent class, taken from HW3
class Agent():
    def __init__(self, observation_dim, params = None, action_bounds = None):
        pass

    def __call__(self, obs):
        return self.act(obs)

# DQN class to represent different components of the algorithm
class DQN(Agent):

    def __init__(self, observation_dim, action_dim, buffer, gamma=0.99):

        self.replayBuffer = buffer

        self.actions = action_dim
        self.obs_dim = observation_dim

        # acting - network
        self.q = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.actions)
        ).double()

        # target - network
        self.q_target = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.actions)
        ).double()

        self.optim = torch.optim.Adam(self.q.parameters(), lr=1e-4)

        # Parameters
        self.N_STEPS = 1000 # (Size of each epoch)
        self.ENV_STEPS = 100 # Number of steps to take before updating models
        self.STEP_BEFORE_TARGET_UPDATE = 400
        self.BATCH_SIZE = 2500 # Sample Size from the replay buffer
        self.N_EPOCHS = 200 # Episodes of training
        self.gamma = gamma

        self.EPSILON = 1.0
        self.EPS_DECAY = 0.99998


        wandb.config.N_STEPS = self.N_STEPS
        wandb.config.ENV_STEPS = self.ENV_STEPS
        wandb.config.STEPS_BEFORE_TARGET_UPDATE = self.STEP_BEFORE_TARGET_UPDATE
        wandb.config.BATCH_SIZE = self.BATCH_SIZE
        wandb.config.N_EPOCHS = self.N_EPOCHS
        wandb.config.gamma = self.gamma

    def compute_target(self, states, rewards):
        """
        states: torch.Tensor of size (batch, obs_dim) with s' from the dataset
        rewards: torch.Tensor of size (batch) with single step rewards (float)

        returns torch.Tensor of size (batch) with the 1-step Q learning target

        Target = R_{t+1} + gamma * max{a}Q(S_{t+1}, a)
        """

        with torch.no_grad():
            # QVals
            qVals = torch.zeros(states.shape[0], dtype=torch.float)  # initializing QVals matrix
            for i in range(states.shape[0]):  # Looping through the batch
                qs = self.q(torch.tensor(states[i]))  # Computing the Q values using the policy network
                maxActionIndex = torch.argmax(qs)  # selecting the action with the highest q value

                # Computing the Q values using the target network, and selecting the action choosen by
                # the policy network.
                qVals[i] = self.q_target(torch.tensor(states[i]))[maxActionIndex]

            qTargetVals = rewards + (self.gamma * qVals.numpy())  # Computing the final target value

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
        """
        computeTrainingSteps takes in a batch of transitions,  and computes the
        Squared Bellman loss
        :param transitions: array of transitions
        :return:
        """
        nextStates = np.stack([s.nextState for s in transitions])
        states = np.stack([s.state for s in transitions])
        actions = np.stack([np.array(s.action) for s in transitions])
        rewards = np.stack([s.reward for s in transitions])

        target = self.compute_target(nextStates, rewards)
        loss = self.loss(states, actions, target)

        return loss

    def train(self, task):
        """
        train function is the main methods used to iniitial self.N_EPOCHS training sesssion
        :param task: Gym Instance of the environment
        :return:
        """
        # Collect initial Data
        self.fillBuffer(task, self.replayBuffer.bufferSize)
        for i in tqdm.tqdm(range(self.N_EPOCHS)):
            self.train_epoch(task, False)

    def train_epoch(self, task, avgMaxReward, randn=False):
        """
        train_epoch conducts the training within each epoch.
        :param task: gym instance
        :param avgMaxReward: last known max avg reward
        :param randn:
        :return:
        """
        lastObs = task.reset()
        # avgRewardMax = -np.inf
        for i in range(self.N_STEPS): # Step through the length of each epoch

            if i % self.ENV_STEPS != 0: # check if the number of environment steps have be conducted or not
                self.EPSILON = self.EPSILON * self.EPS_DECAY # Decay epsilon

                # Determine if the action will be randomly sampled or be selected from the acting network
                if random() < self.EPSILON:
                    act = task.action_space.sample()
                else:
                    act = self(lastObs).item()
                # act = np.random.choice([self(lastObs), np.random.randint(self.actions)], p=[0.9, 0.1])
                # if random:
                #     act = np.random.randint(self.actions)
                wandb.log({"EPSILON": self.EPSILON}) # Logging epsilon value

                obs, rew, done, info = task.step(act) # Stepping through
                exper = Data(lastObs, act, rew, obs, done) # converting to dataclass instance
                self.replayBuffer.insert(exper) # inserting transition to buffer
                lastObs = obs
            else:
                # ENV_STEPS in the environment have stepped, conduct training
                samples = self.replayBuffer.sample(self.BATCH_SIZE) # sample from buffer
                loss = self.computeTrainingStep(samples) # Compute loss

                wandb.log({"Loss": loss, "title": "DQN with Replay Buffer Loss"}) # Log
                # Perform backpropagation
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            if i % self.STEP_BEFORE_TARGET_UPDATE == 0:
                # Time to syncronize the target network with the acting network
                with torch.no_grad():
                    print("Updating Target Model")
                    self.q_target.load_state_dict(self.q.state_dict())

                avgReward = self.getAverageReward(task, i) # Compute average reward

                # Save model if performance is better
                if avgReward > avgMaxReward:
                    torch.save({
                        'targetModel_state_dict': self.q_target.state_dict(),
                        'qModel_state_dict': self.q.state_dict(),
                    }, "./targetModels/modelTimeStamp-" + str(avgReward) + ".pt")
                    avgMaxReward = avgReward

        return avgMaxReward

    def fillBuffer(self, task, experienceSize, random=False):
        """
        fillBuffer is used to generate iniitial transitions to fill up the buffer
        :param task: gym instance
        :param experienceSize: number of transitions to generate
        :param random:
        :return:
        """
        lastObs = task.reset()
        for exp in range(experienceSize):

            act = np.random.choice([self(lastObs), np.random.randint(self.actions)], p=[0.9, 0.1])
            if random:
                act = np.random.randint(self.actions)

            obs, rew, done, info = task.step(act)

            exper = Data(lastObs, act, rew, obs, done)
            self.replayBuffer.insert(exper)
            lastObs = obs

    def getAverageReward(self, task, epoch):
        """
        getAverageReward will run the given task 100 times and return the average score
        produced by the target-network
        :param task: gym instance
        :param epoch: None
        :return: averageRewaard
        """
        rewards = np.zeros((100, 100))

        for run in range(100):
            obs = task.reset()
            for step in range(100):

                # state = torch.from_numpy(obs).view(1, -1)
                actions = self.q_target(torch.tensor(obs))
                act = torch.argmax(actions)

                obs, rew, done, info = task.step(act.item())
                rewards[run, step] = rew

        avgReward = rewards.sum(1).std()
        wandb.log({"avgReward": avgReward})

        return avgReward


def testModel(path):
    """
    testModel is used to load in a stored model and perfrom 100 runs for the maiin task
    :param path: path to the model
    :return:
    """
    checkpoint = torch.load(path)
    task = gym.make("CartPole-v1")
    replayBuffer = ReplayBuffer()
    agent = DQN(task.observation_space.shape[-1], task.action_space.n, replayBuffer)

    agent.q_target.load_state_dict(checkpoint['targetModel_state_dict'])


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
            actions = agent.q_target(torch.tensor(obs))
            act = torch.argmax(actions)

            obs, rew, done, info = task.step(act.item())
            rewards[run, step] = rew
            task.render()

    print("Average return: {}".format(rewards.sum(1).mean()))
    print("Standard deviation: {}".format(rewards.sum(1).std()))


if __name__ == '__main__':

    training()
    # testModel("./targetModels/modelTimeStamp-68.pt")

