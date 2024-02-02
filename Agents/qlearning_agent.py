import numpy as np
import random
import random
import json
import random
from Agents.agent_base import Agent


# ==================================================================
def save_q_table(q_table, filename):
    # Convert the Q-table keys to strings
    q_table_str_keys = {str(key): value for key, value in q_table.items()}

    # Write to a file
    with open(filename, 'w') as file:
        json.dump(q_table_str_keys, file)
        
def load_q_table(filename):
    with open(filename, 'r') as file:
        q_table_str_keys = json.load(file)
    
    # Convert string keys back to tuple keys
    q_table = {eval(key): value for key, value in q_table_str_keys.items()}
    return q_table


# ==================================================================
# Qlearning Agent
class QLearningAgent(Agent):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, player_id=1):
        self.q_table = {} 
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.player_id = player_id  

    def get_q_value(self, state, action):
        """Returns the Q-value for a given state-action pair."""
        state_tuple = tuple(map(tuple, state))
        return self.q_table.get((state_tuple, action), 0)


    def select_action(self, state, possible_actions):
        """Choose an action based on ε-greedy strategy."""
        if not possible_actions:
            return None  

        if random.random() < self.epsilon:
            return random.choice(possible_actions)  # Exploration
        else:
            q_values = [self.get_q_value(state, action) for action in possible_actions]
            max_q_value = max(q_values)
            
            # Get all actions that have the maximum Q-value
            max_q_actions = [action for action, q_value in zip(possible_actions, q_values) if q_value == max_q_value]
            
            # Randomly select among the actions with the maximum Q-value
            return random.choice(max_q_actions)

    def update_q_table(self, state, action, reward, next_state, next_possible_actions):
        """Update the Q-table using the Bellman equation."""
        # Convert numpy array states to tuple of tuples
        state_tuple = tuple(map(tuple, state))
        next_state_tuple = tuple(map(tuple, next_state))

        # Find the maximum Q-value for the possible actions in the next state
        max_future_q = max([self.get_q_value(next_state_tuple, next_action) for next_action in next_possible_actions], default=0)
        
        # Get the current Q-value and calculate the new Q-value using the Bellman equation
        current_q = self.get_q_value(state_tuple, action)
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        
        # Update the Q-table
        self.q_table[(state_tuple, action)] = new_q



