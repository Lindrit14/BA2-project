# strategies/decay_epsilon.py
import numpy as np
import random
import math

class DecayEpsilonStrategy:
    def __init__(self, initial_epsilon=1.0, min_epsilon=0.01, decay_rate=0.001):
        """
        Initialize the DecayEpsilonStrategy.

        Parameters:
        - initial_epsilon (float): The starting value of epsilon for exploration.
        - min_epsilon (float): The minimum value of epsilon to ensure some exploration.
        - decay_rate (float): The rate at which epsilon decays over episodes.
        """
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def select_action(self, q_values, episode):
        """
        Select an action based on the decaying epsilon-greedy strategy.

        Parameters:
        - q_values (list or np.ndarray): The Q-values for the current state.
        - episode (int): The current episode number, used to calculate epsilon decay.

        Returns:
        - int: The index of the selected action.
        """
        epsilon = max(self.min_epsilon, self.initial_epsilon * math.exp(-self.decay_rate * episode))
        if random.random() < epsilon:
            return random.randint(0, len(q_values) - 1)
        else:
            return int(np.argmax(q_values))
