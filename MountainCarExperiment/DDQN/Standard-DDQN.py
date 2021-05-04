import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as fn
import numpy as np
import tqdm
import wandb

wandb.init(project='MountainCar', entity='masterrom')


# Simple agent class, taken from HW3
class Agent():
    def __init__(self, observation_dim, params = None, action_bounds = None):
        pass

    def __call__(self, obs):
        return self.act(obs)

# DQN class to represent different components of the algorithm
class DQN(Agent):

    def __init__(self, observation_dim, action_dim, gamma=0.99):
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

        for i in range(self.N_STEPS):  # Step through the length of the epoch
            # Extract out batches
            batch_x = np.random.randint(num_runs, size=(self.BATCH_SIZE,))
            batch_y = np.random.randint(len_runs, size=(self.BATCH_SIZE,))
            batch_states = torch.from_numpy(states[batch_x, batch_y])
            batch_actions = torch.from_numpy(actions[batch_x, batch_y]).to(int)
            batch_rewards = torch.from_numpy(rewards[batch_x, batch_y])
            batch_next_states = torch.from_numpy(next_states[batch_x, batch_y])

            # Compute Loss
            target = self.compute_target(batch_next_states, batch_rewards)
            loss = self.loss(batch_states, batch_actions, target)

            wandb.log({"Loss": loss})
            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses.append(loss.item())
        # Syncronizing target and acting network
        with torch.no_grad():
            print("Updating Target Model")
            self.q_target.load_state_dict(self.q.state_dict())

        return losses

    def train(self, task):
        """
        train function is the main methods used to iniitial self.N_EPOCHS training sesssion
        :param task: Gym Instance of the environment
        :return:
        """
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
        """
        collect_data is used to collect the experience of the acting network interacting with
        the environment
        :param task: gym instance
        :param random:
        :return:
        """
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

        return states, actions, rewards

    def reward(self, state, nextState, action, reward):
        """
        reward function to promote momentum building action sequence
        :param state:
        :param nextState:
        :param action:
        :param reward:
        :return:  modified reward
        """
        if nextState[0] - state[0] > 0 and action == 2: reward = 1
        if nextState[0] - state[0] < 0 and action == 0: reward = 1
        return reward

    def getAverageReward(self, task):
        """
        getAverageReward will run the given task 100 times and return the average score
        produced by the target-network
        :param task: gym instance
        :param epoch: None
        :return: averageRewaard
        """
        rewards = np.zeros((100, 200))

        scores = []
        for run in range(100):
            score = 0
            obs = task.reset()
            for step in range(200):
                # state = torch.from_numpy(obs).view(1, -1)
                actions = self.q_target(torch.tensor(obs))
                act = torch.argmax(actions)
                obs, rew, done, info = task.step(act.item())
                score += rew

                if done:
                    break

            scores.append(score)

        avgReward = np.mean(scores)
        # avgReward = rewards.sum(1).std()
        wandb.log({"avgReward": avgReward})

        return avgReward


def testModel(path):
    """
    testModel is used to load in a stored model and perfrom 100 runs for the maiin task
    :param path: path to the model
    :return:
    """

    checkpoint = torch.load(path)
    task = gym.make("MountainCar-v0")
    agent = DQN(task.observation_space.shape[-1], task.action_space.n)

    agent.q_target.load_state_dict(checkpoint['targetModel_state_dict'])

    rewards = np.zeros((100, 200))

    scores = []
    for run in range(100):
        obs = task.reset()
        score = 0
        for step in range(200):

            actions = agent.q_target(torch.tensor(obs))
            act = torch.argmax(actions)

            obs, rew, done, info = task.step(act.item())
            score += rew
            task.render()
            if done:
                break

        print("Score for iteration:{} = {}".format(run, score))
        scores.append(score)

    print("Average return: {}".format(np.mean(scores)))
    # print("Standard deviation: {}".format(rewards.sum(1).std()))



if __name__ == '__main__':

    # testModel("./targetModels/modelTimeStamp-199.pt")

    # Do not modify
    task = gym.make("MountainCar-v0")
    agent = DQN(task.observation_space.shape[-1], task.action_space.n)

    losses = agent.train(task)

    # Final Benchmarking
    print("Final Benchmarking")

    rewards = np.zeros((100, 200))
    scores = []
    for run in range(100):
        obs = task.reset()
        score = 0
        for step in range(200):
            act = agent(obs)
            obs, rew, done, info = task.step(act.item())
            score += rew
            if done:
                break
            task.render()

        print("iteration {} : Score {}".format(run, score))
        scores.append(score)


    print("Average return: {}".format(np.mean(scores)))
    # print("Standard deviation: {}".format(rewards.sum(1).std()))



