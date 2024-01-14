import random
from .agent_base import Agent

class RandomAgent(Agent):
    def __init__(self, player_id):
        super().__init__(player_id)

    def select_action(self, game_state):
        legal_moves = game_state.generate_legal_moves(self.player_id)
        return random.choice(legal_moves) if legal_moves else None

    def learn_from_experience(self, *args):
        pass  # Random agent does not learn

    def save_model(self, file_path):
        pass  # Random agent has no model to save

    def load_model(self, file_path):
        pass  # Random agent has no model to load
