import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as fn
import numpy as np
import tqdm
import wandb

wandb.init(project='Cartpole', entity='masterrom')


class Agent():
    def __init__(self, observation_dim, params = None, action_bounds = None):
        pass

    def __call__(self, obs):
        return self.act(obs)

class DQN(Agent):

    def __init__(self, observation_dim, action_dim, gamma=0.99):
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
                qs = self.q_target(states[i])  # Computing Q values from the target network
                maxActionIndex = torch.argmax(qs)  # Selecting the best action
                qVals[i] = qs[maxActionIndex]  # selecting the max qValues from the dataset

            qTargetVals = rewards + (self.gamma * qVals)  # Computing the final Target

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
        policyQ = self.q(states).gather(1, actions)
        # target = torch.reshape(target, (target.shape[0], 1))  # Reshaping the target to be in a single column


        loss = torch.square(target - policyQ)  # Computing the squared loss for each time step

        return torch.sum(loss)  # Summing the loss

    def __call__(self, state):
        """
        states: np.array of size (obs_dim,) with the current state

        returns np.array of size (1,) with the optimal action
        """

        state = torch.from_numpy(state).view(1, -1)
        actions = self.q(state)
        return torch.argmax(actions)

    def train_epoch(self, states, actions, rewards, next_states):
        # Do not modify
        num_runs = states.shape[0]
        len_runs = states.shape[1]
        losses = []

        for i in range(self.N_STEPS):
            batch_x = np.random.randint(num_runs, size=(self.BATCH_SIZE,))
            batch_y = np.random.randint(len_runs, size=(self.BATCH_SIZE,))
            batch_states = torch.from_numpy(states[batch_x, batch_y])
            batch_actions = torch.from_numpy(actions[batch_x, batch_y]).to(int)
            batch_rewards = torch.from_numpy(rewards[batch_x, batch_y])
            batch_next_states = torch.from_numpy(next_states[batch_x, batch_y])

            target = self.compute_target(batch_next_states, batch_rewards)
            loss = self.loss(batch_states, batch_actions, target)

            wandb.log({"Loss": loss})

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses.append(loss.item())
        with torch.no_grad():
            print("Updating Target Model")
            self.q_target.load_state_dict(self.q.state_dict())

        return losses

    def train(self, task):
        # Do not modify
        losses = []
        states, actions, rewards = self.collect_data(task, random=True)

        for i in tqdm.tqdm(range(self.N_EPOCHS)):
            states, actions, rewards = self.collect_data(task, random=False)
            epoch_losses = self.train_epoch(states[:, :-1], actions, rewards, states[:, 1:])
            losses += epoch_losses

            torch.save({
                'targetModel_state_dict': self.q_target.state_dict(),
                'qModel_state_dict': self.q.state_dict(),
            }, "./targetModels/modelTimeStamp-" + str(i) + ".pt")
        return losses

    def collect_data(self, task, random=False):
        # Do not modify

        rewards = np.zeros((100, 100, 1))
        states = np.zeros((100, 101, self.obs_dim))
        actions = np.zeros((100, 100, 1))

        for run in range(100):
            obs = task.reset()
            for step in range(100):
                states[run, step] = obs

                act = np.random.choice([self(obs), np.random.randint(self.actions)], p=[0.9, 0.1])
                if random:
                    act = np.random.randint(self.actions)
                obs, rew, done, info = task.step(act)
                rewards[run, step] = rew
                actions[run, step] = act
            states[run, -1] = obs

        # print(f"Average return in training: {np.mean(rewards)}")
        return states, actions, rewards




if __name__ == '__main__':

    # Do not modify
    task = gym.make("CartPole-v1")
    agent = DQN(task.observation_space.shape[-1], task.action_space.n)

    losses = agent.train(task)

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
