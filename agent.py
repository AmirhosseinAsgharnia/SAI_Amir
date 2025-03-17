import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from nn_models import neural_network_model
from nn_models import exploration_network
    
class agent_class:
    def __init__(self):
        
        # Hyperparameters
        self.state_shape = (13, 19, 13)
        self.action_dim = 10
        self.gamma = 0.995
        self.epsilon = 0.3
        self.epsilon_min = 0.01
        self.epsilon_decay = 1
        self.batch_size = 64
        self.target_update_freq = 500
        self.buffer_size = 10000
        self.learning_rate = 1e-3

        # Q Network
        self.Q_network = neural_network_model(input_channels=self.state_shape[0] , action_dim=self.action_dim)

        # Target Network
        self.target_network = neural_network_model(input_channels=self.state_shape[0],action_dim=self.action_dim)
        
        self.target_network.load_state_dict(self.Q_network.state_dict())

        # Replay Buffer
        self.memory = deque(maxlen=self.buffer_size)

        # Optimizer & Loss
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.learning_rate)
        
        self.loss_fn = nn.MSELoss()

        # Steps for updating Target net
        self.steps = 0  

        # Exploration Network
        self.model = exploration_network(input_channels=self.state_shape[0])
        
        self.visited_states = {}

    def remember(self, obs_1, action, reward, obs_2, done):

        self.memory.append((obs_1, action, reward, obs_2, done))

    def act(self, obs_1):

        # Epsilon-greedy 
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.tensor(obs_1, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            q_values = self.Q_network(state_tensor)

        return torch.argmax(q_values, dim=1).item()

    def train_step(self):

        if len(self.memory) < self.batch_size:
            return

        # batch
        minibatch = random.sample(self.memory, self.batch_size)
        obs_1, actions, rewards, obs_2, dones = zip(*minibatch)

        # Convert to tensors
        obs_1   = torch.tensor(obs_1, dtype=torch.float32).reshape(self.batch_size, *self.state_shape)
        obs_2   = torch.tensor(obs_2, dtype=torch.float32).reshape(self.batch_size, *self.state_shape)

        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones   = torch.tensor(dones, dtype=torch.float32)

        # Compute Q-values
        with torch.no_grad():
            next_q_values = self.target_network(obs_2)
            best_next_q_values = torch.max(next_q_values, dim=1)[0]

            visit_penalty = torch.tensor(self.visit_num(obs_2), dtype=torch.float32)
            # visit penalty converges to 5
            visit_penalty = 5 - 5 * torch.exp(-0.01*visit_penalty)

            targets = rewards + (1.0 - dones) * (self.gamma * best_next_q_values - visit_penalty) # Temporal difference

        q_values = self.Q_network(obs_1)
        q_values = q_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

        # Update net weights and biases
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon (For now no decay)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update Target net at every 500 steps
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.Q_network.state_dict())
    # Updating number of times a state is visited
    def visit_state(self, state):

        with torch.no_grad():
            latent_state = self.model(state) 
            state_key = tuple(latent_state.flatten().tolist())

        if state_key not in self.visited_states:
            self.visited_states[state_key] = 1
        else:
            self.visited_states[state_key] += 1

    # Returning number of times a state is visited
    def visit_num(self, state):

        if len(state.shape) == 4:
            counts = []
            for i in range(state.shape[0]):
                with torch.no_grad():
                    latent_output = self.model(state[i]) 
                state_key = tuple(latent_output.flatten().tolist())
                counts.append(self.visited_states.get(state_key, 0))
            return counts 
