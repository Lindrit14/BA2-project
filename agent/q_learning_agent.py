# agent/q_learning_agent.py
import numpy as np

class QLearningAgent:
    def __init__(self, strategy, action_space=4, alpha=0.1, gamma=0.95):
        """
        strategy: an object with method select_action(q_values, episode)
        """
        self.strategy = strategy
        self.action_space = action_space
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.Q = {}  # Q-table: { state_tuple: [Q_a0, Q_a1, Q_a2, Q_a3] }

    def get_qvals(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.action_space)
        return self.Q[state]

    def select_action(self, state, episode):
        qvals = self.get_qvals(state)
        return self.strategy.select_action(qvals, episode)

    def update(self, state, action, reward, next_state):
        qvals = self.get_qvals(state)
        next_qvals = self.get_qvals(next_state)
        td_target = reward + self.gamma * np.max(next_qvals)
        td_error = td_target - qvals[action]
        qvals[action] += self.alpha * td_error
