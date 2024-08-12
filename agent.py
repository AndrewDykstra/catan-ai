import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from catan_env import CatanEnv

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CatanAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 32
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state).clone()
            target_f[0][action] = target
            output = self.model(state)
            loss = self.criterion(output, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

class RuleBasedAgent:
    def __init__(self):
        pass

    def choose_action(self, state):
        wood = state[150]  # Assumes wood is at index 150
        brick = state[151]
        wheat = state[152]
        sheep = state[153]
        ore = state[154]

        if wood > 0 and brick > 0:
            return 0  # Build a road
        if wood > 0 and brick > 0 and wheat > 0 and sheep > 0:
            return 1  # Build a settlement
        return random.choice([i for i in range(2, 22)])  # Trade with bank randomly

def train_agent(env, agent, rule_based_agents, episodes=1000):
    for e in range(episodes):
        state = env.reset()
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2}")
                break
            agent.replay()

        if e % 100 == 0:
            agent.save(f"catan_agent_{e}.pth")

if __name__ == "__main__":
    env = CatanEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = CatanAgent(state_size, action_size)
    rule_based_agents = [RuleBasedAgent() for _ in range(3)]  # Other players are rule-based

    # Rule based training
    train_agent(env, agent, rule_based_agents)
    agent.save("final_catan_agent.pth")
