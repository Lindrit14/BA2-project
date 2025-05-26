# strategies/softmax.py
import numpy as np
import math

class SoftmaxStrategy:
    def __init__(self, initial_temp=1.0, min_temp=0.1, decay_rate=0.001):
        """
        Initialize the SoftmaxStrategy.

        Parameters:
        initial_temp (float): The starting temperature for the softmax function.
        min_temp (float): The minimum temperature to avoid division by zero.
        decay_rate (float): The rate at which the temperature decays over episodes.
        """
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.decay_rate = decay_rate

    def select_action(self, q_values, episode):

        """
        Select an action based on the softmax probability distribution.

        Parameters:
        q_values (list or np.ndarray): The Q-values for each action.
        episode (int): The current episode number, used to adjust the temperature.

        Returns:
        int: The index of the selected action.
        """
        
        T = max(self.min_temp, self.initial_temp * math.exp(-self.decay_rate * episode))
        preferences = q_values / T
        max_pref = np.max(preferences)
        exp_preferences = np.exp(preferences - max_pref)  # for numerical stability
        probs = exp_preferences / np.sum(exp_preferences)
        return int(np.random.choice(len(q_values), p=probs))
