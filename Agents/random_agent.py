import random
from .agent_base import Agent

class RandomAgent(Agent):
    def __init__(self, player_id):
        super().__init__(player_id)

    def select_action(self, game):
        legal_moves = game.get_legal_moves(self.player_id)
        return random.choice(legal_moves) if legal_moves else None
