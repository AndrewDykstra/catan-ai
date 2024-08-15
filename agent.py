import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from catan_env import CatanEnv
import matplotlib.pyplot as plt

os.makedirs('results', exist_ok=True)

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

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
        if len(self.memory) > 2000:  # Limit the size of the memory
            self.memory.pop(0)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        minibatch = random.sample(self.memory, self.batch_size)
        total_loss = 0

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward

            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()

            target_f = self.model(state).clone()
            target_f[0][action] = target
            output = self.model(state)
            loss = self.criterion(output, target_f)
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_loss / len(minibatch)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

def train_agent(env, agents, episodes):
    episode_rewards = []  # To track total rewards per episode
    episode_lengths = []  # To track the number of steps per episode
    episode_losses = []   # To track losses per episode
    action_counts = {i: 0 for i in range(env.action_space.n)}  # Track action usage

    for e in range(episodes):
        print(f"Starting episode {e + 1}/{episodes}")
        state = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        episode_loss = 0  # Initialize loss for this episode

        while not done:
            current_player = env.current_player
            agent = agents[current_player]
            action = agent.act(state)
            action_counts[action] += 1  # Track action usage

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()  # Replay will return the loss
            episode_loss += loss if loss is not None else 0  # Accumulate loss

            state = next_state
            step_count += 1

            if step_count > 10000:
                print(f"Episode {e+1} exceeded 10,000 steps. Skipping...")
                break  # Skip to the next episode if exceeding 10,000 steps

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        episode_losses.append(episode_loss)  # Total loss for the episode

        print(f"Episode {e+1}/{episodes} completed with total reward: {total_reward}, "
              f"total loss: {episode_loss:.4f} after {step_count} steps. "
              f"Final Epsilon: {agent.epsilon:.2f}")
        
        if done:
            print("Game ended naturally, preparing for next episode.")
        
        if (e + 1) % 100 == 0:
            agent.save(f"catan_agent_{e + 1}.pth")
            print(f"Agent model saved after {e + 1} episodes.")

    return episode_rewards, episode_lengths, episode_losses, action_counts

if __name__ == "__main__":
    env = CatanEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agents = [CatanAgent(state_size, action_size) for _ in range(env.num_players)]  # All players now using Q-learning

    # Train the agent
    episode_rewards, episode_lengths, episode_losses, action_counts = train_agent(env, agents, episodes=10)
    agents[0].save("final_catan_agent.pth")
    print("Final agent model saved.")

    # Total reward per episode
    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.savefig('results/total_reward_per_episode.png')
    plt.close()

    # Number of steps per episode
    plt.figure()
    plt.plot(episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Steps per Episode')
    plt.title('Steps per Episode')
    plt.savefig('results/steps_per_episode.png')
    plt.close()

    # Total loss per episode
    plt.figure()
    plt.plot(episode_losses)
    plt.xlabel('Episode')
    plt.ylabel('Total Loss')
    plt.title('Total Loss per Episode')
    plt.savefig('results/total_loss_per_episode.png')
    plt.close()

    # Action distribution
    actions, counts = zip(*action_counts.items())
    plt.figure()
    plt.bar(actions, counts)
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.title('Action Frequency Distribution')
    plt.savefig('results/action_frequency_distribution.png')
    plt.close()
