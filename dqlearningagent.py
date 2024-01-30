import numpy as np
import random
import pickle
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from Assets.checkers_model import CheckersGame
from Agents.agent_base import Agent


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Define your neural network architecture here
        # Example: Two fully connected layers
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQLAgent(Agent):
    def __init__(self, player_id, state_size, action_size, learning_rate=0.001, discount_factor=0.9, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, memory_size=10000, batch_size=64):
        super().__init__(player_id)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def state_to_tensor(self, state):
        # Convert state to a tensor
        return torch.tensor(state, dtype=torch.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = self.state_to_tensor(state)
        q_values = self.model(state_tensor).detach()
        return np.argmax(q_values.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = self.state_to_tensor(next_state)
                target = reward + self.discount_factor * torch.max(self.target_model(next_state_tensor).detach())
            state_tensor = self.state_to_tensor(state)
            target_f = self.model(state_tensor)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state_tensor), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        self.model.eval()


def train_agent(num_episodes, dql_agent, opponent_agent, game, update_target_every=5):
    for episode in range(num_episodes):
        game.reset()
        total_reward = 0
        current_player = game.player
        state = game.get_state()

        while not game.is_game_over():
            if current_player == dql_agent.player_id:
                # DQL Agent's turn
                action = dql_agent.select_action(state)
                new_state, reward, done = game.perform_action_and_evaluate(action, dql_agent.player_id)
                dql_agent.remember(state, action, reward, new_state, done)
                dql_agent.replay()  # Train the model with experiences
                total_reward += reward
                state = new_state
            else:
                # Opponent Agent's turn
                action = opponent_agent.select_action(state)
                new_state, _ = game.perform_action_and_evaluate(action, opponent_agent.player_id)

            current_player = game.player

        # Update the target model every few episodes
        if episode % update_target_every == 0:
            dql_agent.update_target_model()

        print(f"Episode {episode}, Total Reward: {total_reward}")

    dql_agent.save_model("dql_checkers_agent.pth")


def train_agent(num_episodes, dql_agent, game):
    update_target_every = 5
    for episode in range(num_episodes):
        game.reset()
        total_reward = 0
        state = game.get_state()

        while not game.is_game_over():
            # DQL Agent's turn
            action = dql_agent.select_action(state)
            new_state, reward, done = game.perform_action_and_evaluate(action, dql_agent.player_id)
            dql_agent.remember(state, action, reward, new_state, done)
            dql_agent.replay()  # Train the model with experiences
            total_reward += reward

            if done:
                break  # End the episode if the game is over

            # Simulate opponent's turn (if applicable)
            # This could be a random move or some simple heuristic
            opponent_action = game.simulate_opponent_action()
            game.perform_opponent_action(opponent_action)

            # Update state and continue
            state = game.get_state()

        # Update the target model every few episodes
        if episode % update_target_every == 0:
            dql_agent.update_target_model()

        print(f"Episode {episode}, Total Reward: {total_reward}")

    dql_agent.save_model("dql_checkers_agent_solo.pth")

# Usage
# Initialize DQLAgent and CheckersGame as needed
# train_agent(100, dql_agent, game)

# Usage
# Initialize DQLAgent, OpponentAgent, and CheckersGame as needed
# train_agent(100, dql_agent, opponent_agent, game)

# Example usage
# state_size = # Define the size of the state representation
# action_size = # Define the number of possible actions
# dql_agent = DQLAgent(player_id=1, state_size=state_size, action_size=action_size)

# Train and use the agent similar to the CheckersAgent example
