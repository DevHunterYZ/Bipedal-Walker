import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Hyperparameters
EPISODES = 50
MAX_STEPS = 1600  # Maximum steps per episode
MEMORY_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-3
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500

# Discretize the action space
def create_discrete_actions(num_bins):
    # For BipedalWalker-v3, action space is Box(-1,1,shape=(4,))
    # We'll discretize each dimension into num_bins
    # Then create all possible combinations
    bins = np.linspace(-1, 1, num_bins)
    actions = []
    for a0 in bins:
        for a1 in bins:
            for a2 in bins:
                for a3 in bins:
                    actions.append([a0, a1, a2, a3])
    return np.array(actions)

NUM_BINS = 3  # Adjust based on computational resources
DISCRETE_ACTIONS = create_discrete_actions(NUM_BINS)
NUM_ACTIONS = len(DISCRETE_ACTIONS)
print(f"Number of discrete actions: {NUM_ACTIONS}")

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.steps_done = 0
        self.epsilon = EPS_START
        self.action_dim = action_dim
    
    def select_action(self, state):
        self.steps_done += 1
        # Epsilon decay
        self.epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * self.steps_done / EPS_DECAY)
        if random.random() < self.epsilon:
            action_idx = random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
                action_idx = q_values.argmax().item()
        return action_idx
    
    def push_memory(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return
        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)
        
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # Current Q values
        q_values = self.q_network(state).gather(1, action)
        
        # Target Q values
        with torch.no_grad():
            max_next_q = self.target_network(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + (1 - done) * GAMMA * max_next_q
        
        # Loss
        loss = F.mse_loss(q_values, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Training Loop
def train():
    # Ortamı render etmek için 'render_mode' parametresini ekliyoruz
    env = gym.make("BipedalWalker-v3", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = NUM_ACTIONS  # Discrete actions
    agent = DQNAgent(state_dim, action_dim)
    rewards_history = []
    
    for episode in range(1, EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0
        for step in range(MAX_STEPS):
            action_idx = agent.select_action(state)
            action = DISCRETE_ACTIONS[action_idx]
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.push_memory(state, action_idx, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward

            # Ortamı render etmek için adımda render fonksiyonunu çağırıyoruz
            env.render()
            
            if done:
                break
        rewards_history.append(total_reward)
        agent.update_target_network()
        print(f"Episode {episode} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    # Plotting the rewards
    plt.figure(figsize=(12,8))
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN on BipedalWalker-v3')
    plt.show()

if __name__ == "__main__":
    train()
