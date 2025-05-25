# q_learning_agent.py
import numpy as np
import random
import math

class QLearningAgent:
    def __init__(self, action_space=4, alpha=0.1, gamma=0.95,
                 exploration_mode='epsilon_greedy'):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.Q = {}  # Q-table, dict with keys as (state) and subdict for actions
        self.exploration_mode = exploration_mode
        # For epsilon modes
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.decay_rate = 0.001
        # For softmax
        self.temp = 1.0
        self.temp_min = 0.1
        self.temp_decay = 0.001

    def get_Q(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.action_space)
        return self.Q[state]

    def select_action(self, state, episode=1):
        qvals = self.get_Q(state)
        if self.exploration_mode == 'epsilon_greedy':
            if random.random() < self.epsilon:
                return random.randint(0, self.action_space-1)
            else:
                return np.argmax(qvals)

        elif self.exploration_mode == 'decay_epsilon':
            # reduce epsilon over time
            eps = max(self.epsilon_min, self.epsilon * math.exp(-self.decay_rate * episode))
            if random.random() < eps:
                return random.randint(0, self.action_space-1)
            else:
                return np.argmax(qvals)

        elif self.exploration_mode == 'softmax':
            # reduce temperature over time
            T = max(self.temp_min, self.temp * math.exp(-self.temp_decay * episode))
            # compute preferences
            exp_q = np.exp(qvals / T)
            probs = exp_q / np.sum(exp_q)
            return np.random.choice(range(self.action_space), p=probs)

    def update(self, state, action, reward, next_state):
        # Standard Q-learning update
        qvals = self.get_Q(state)
        qvals_next = self.get_Q(next_state)
        td_target = reward + self.gamma * np.max(qvals_next)
        td_error = td_target - qvals[action]
        qvals[action] = qvals[action] + self.alpha * td_error
