import CNN
import checkers_env
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import checkers_env
import numpy as np

class solver:

    def __init__(self, step_size, epsilon, env):

        self.step_size = step_size
        self.epsilon = epsilon
        self.env = checkers_env.CheckersGame( )
        self.q_table = np.zeros(len(env.state_space), len(env.action_space))

    def solve(self, episode):
        for i in episode:
            state = checkers_env.CheckersGame.initialize_board()
            a = self.policy_improvement()

            # update Q value
            q[state, action] =

    def policy_improvement(self):

        return a

    def __init__(self, capacity, q_func, env, player):
        """

        :param capacity: initialize replay memory to capacity
        :param q_func: initialize q_function, with input [state, action]
        """
        self.memory = deque([], maxlen=capacity)
        self.q_func = q_func
        self.env = checkers_env.CheckersGame(env.initialize_board(), player)
        self.player = player

    def push(self, transition):
        """
        save a transition
        :param transition:
        :return:
        """
        self.memory.append(transition)

    def generate_transition(self, state, action):
        """
        execute action and observe reward and next state
        :param state:
        :param action:
        :return:
        """
        next_state = self.env.step(action, self.player)
        reward = self.env.game_winner(next_state)
        transition = [state, action, next_state, reward]
        return transition


    def policy_improvement(self, state, epsilon, player):
        """
        With probability epsilon, choose a random action,
        otherwise, choose the action that maximizes Q.
        :param state:
        :param epsilon:
        :return: action
        """
        sample = random.random()
        if sample > epsilon:
            if self.env.possible_actions(player) > 0:
                max_action = self.env.possible_actions(player)[0]
                max_value = self.q_func.predict(state, max_action)
                for a in self.env.possible_actions(player):
                    if self.q_func.predict(state, a) > max_value:
                        max_action = a
                        max_value = self.q_func.predict(state, a)
            else:
                return None
        else:
            max_action = random.sample(self.env.possible_actions(player), 1)
        return max_action

    def evaluation(self):

    def solve(self, episode, epsilon, batch_size):
        for t in episode:
            state = self.env.initialize_board()
            action = self.policy_improvement(state, epsilon, self.player)
            transition = self.policy_evaluation(state, action)
            state = transition[2]
            self.push(transition)
            samples = random.sample(self.memory, batch_size)
            self.policy_evaluation()
            self.q_func.fit()





