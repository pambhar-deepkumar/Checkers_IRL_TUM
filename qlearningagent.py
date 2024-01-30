import numpy as np
import pickle
import random
from checkers_model import CheckersGame
from Agents.alphabeta_agent import AlphaBetaAgent
from Agents.agent_base import Agent
class CheckersAgent(Agent):
    def __init__(self, player_id, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        super().__init__(player_id)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}  

    def state_to_key(self, state):
        return str(state)

    def select_action(self, game_state):
        state_key = self.state_to_key(game_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}

        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return random.choice(game_state.generate_legal_moves(self.player_id))
        else:
            # Exploit: choose the best known action
            max_q_value = float('-inf')
            best_action = None
            for action in game_state.generate_legal_moves(self.player_id):
                action_q_value = self.q_table[state_key].get(action, 0)
                if action_q_value > max_q_value:
                    max_q_value = action_q_value
                    best_action = action

            if best_action is None:
                return random.choice(game_state.generate_legal_moves(self.player_id))
            return best_action

    def update_q_table(self, old_state, action, reward, new_state):
        old_state_key = self.state_to_key(old_state)
        new_state_key = self.state_to_key(new_state)

        old_q_value = self.q_table[old_state_key].get(action, 0)
        max_future_q = max(self.q_table[new_state_key].values(), default=0)

        # Q-learning update rule
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * max_future_q - old_q_value)
        self.q_table[old_state_key][action] = new_q_value

    def save_model(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self.q_table, file)
        print(f"Model saved to {file_name}")

    def load_model(self, file_name):
        with open(file_name, 'rb') as file:
            self.q_table = pickle.load(file)
        print(f"Model loaded from {file_name}")
    
    
def train_agent(num_episodes, agent, game):
    for episode in range(num_episodes):
        game.reset()
        total_reward = 0

        while not game.is_game_over():
            current_state = game.get_state()
            action = agent.select_action(current_state)
            new_state, reward = game.perform_action_and_evaluate(action, agent.player_id)
            agent.update_q_table(current_state, action, reward, new_state)
            total_reward += reward

        print(f"Episode {episode}, Total Reward: {total_reward}")

def train_agent(num_episodes, rl_agent, opponent_agent, game):
    for episode in range(num_episodes):
        game.reset()
        total_reward = 0
        current_player = game.player

        while not game.is_game_over():
            current_state = game.get_state()

            if current_player == rl_agent.player_id:
                action = rl_agent.select_action(current_state)
                new_state, reward = game.perform_action_and_evaluate(action, rl_agent.player_id)
                rl_agent.update_q_table(current_state, action, reward, new_state)
                total_reward += reward
            else:
                action = opponent_agent.select_action(current_state)
                new_state, _ = game.perform_action_and_evaluate(action, opponent_agent.player_id)

            current_player = game.player

        print(f"Episode {episode}, Total Reward: {total_reward}")



rl_agent = CheckersAgent(player_id=1)
opponent_agent = AlphaBetaAgent(player_id=-1)  # Replace with your actual opponent agent class
game = CheckersGame()

train_agent(10, rl_agent, opponent_agent, game)

rl_agent.save_model("checkers_q_learning_agent.pkl")
