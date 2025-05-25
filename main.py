# main.py
import matplotlib.pyplot as plt
import numpy as np
import os
from save_results import save_to_csv
from static_maze import StaticMaze
from dynamic_maze import DynamicMaze
from q_learning_agent import QLearningAgent

def run_qlearning(env, agent, episodes=500):
    episode_rewards = []
    success_list = []
    steps_list = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < 1000:
            action = agent.select_action(state, ep)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            steps += 1
        episode_rewards.append(total_reward)
        success_list.append(1 if done else 0)
        steps_list.append(steps)
    return episode_rewards, success_list, steps_list

def train_and_plot(strategy, env_type, episodes=500):
    # Create the environment (either static or dynamic)
    env = StaticMaze() if env_type == 'static' else DynamicMaze()
    agent = QLearningAgent(exploration_mode=strategy)
    
    # Run training loop
    rewards, success_list, steps_list = run_qlearning(env, agent, episodes)

    # ----- SAVE RESULTS TO CSV -----
    save_to_csv(rewards, success_list, steps_list, strategy, env_type)
 

    # Create subfolder for the given strategy (if it doesn't exist)
    os.makedirs(strategy, exist_ok=True)

    #### 1) Plot total rewards ####
    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(rewards, label="Total Reward")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f"{strategy} on {env_type}: Rewards")
    plt.legend()
    # Save figure to subfolder
    plt.savefig(os.path.join(strategy, f"{strategy}_{env_type}_rewards.png"))
    plt.close()

    #### 2) Plot success rate ####
    success_rate = np.cumsum(success_list) / np.arange(1, len(success_list)+1)
    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(success_rate, label="Success Rate")
    plt.xlabel('Episode')
    plt.ylabel('Rate')
    plt.title(f"{strategy} on {env_type}: Success Rate")
    plt.legend()
    # Save figure to subfolder
    plt.savefig(os.path.join(strategy, f"{strategy}_{env_type}_success.png"))
    plt.close()

    #### 3) Plot steps per episode ####
    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(steps_list, label="Steps per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title(f"{strategy} on {env_type}: Steps to Goal")
    plt.legend()
    # Save figure to subfolder
    plt.savefig(os.path.join(strategy, f"{strategy}_{env_type}_steps.png"))
    plt.close()

def main():
    strategies = ['epsilon_greedy', 'decay_epsilon', 'softmax']
    env_types = ['static', 'dynamic']

    for strategy in strategies:
        for env_type in env_types:
            print(f"Running {strategy} on {env_type} maze...")
            train_and_plot(strategy, env_type, episodes=500)

if __name__ == '__main__':
    main()
