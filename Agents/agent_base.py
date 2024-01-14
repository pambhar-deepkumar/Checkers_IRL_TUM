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
    def select_action(self, game_state):
        """
        Abstract method to select an action based on the current game state.

        Args:
            game_state (CheckersGame): The current state of the game.

        Returns:
            action (tuple): The selected action.
        """
        pass

    @abstractmethod
    def learn_from_experience(self, game_state, action, reward, next_game_state):
        """
        Abstract method for the agent to learn from its experience. 
        This method will be used in reinforcement learning agents.

        Args:
            game_state (CheckersGame): The state of the game before the action.
            action (tuple): The action taken.
            reward (float): The reward received after taking the action.
            next_game_state (CheckersGame): The state of the game after the action.
        """
        pass

    @abstractmethod
    def save_model(self, file_path):
        """
        Abstract method to save the agent's model.

        Args:
            file_path (str): The path to save the model.
        """
        pass

    @abstractmethod
    def load_model(self, file_path):
        """
        Abstract method to load the agent's model.

        Args:
            file_path (str): The path to load the model from.
        """
        pass
