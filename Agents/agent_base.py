from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, player_id):
        """
        Initialize the agent.

        Args:
            player_id (int): The identifier for the player (1 or -1 in the case of Checkers).
        """
        self.player_id = player_id

    @abstractmethod
    def select_action(self):
        """
        Abstract method to select an action based on the current game state.

        Args:
            game_state (CheckersGame): The current state of the game.

        Returns:
            action (tuple): The selected action.
        """
        pass

