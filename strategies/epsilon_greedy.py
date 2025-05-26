# strategies/epsilon_greedy.py
import numpy as np
import random

class EpsilonGreedyStrategy:
    def __init__(self, epsilon=0.1):
        """
        Initialize the EpsilonGreedyStrategy.

        Parameters:
        epsilon (float): The probability of selecting a random action (exploration).
                         Should be a value between 0 and 1.
        """
        self.epsilon = epsilon

    def select_action(self, q_values, episode):
        """
        Select an action based on the epsilon-greedy strategy.

        Parameters:
        q_values (list or np.ndarray): A list or array of Q-values for each action.

        Returns:
        int: The index of the selected action.
        """
        if random.random() < self.epsilon:
            return random.randint(0, len(q_values) - 1)
        else:
            return int(np.argmax(q_values))
