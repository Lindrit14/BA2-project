# main.py
import os
import time
import numpy as np
import random

from environments.static_maze import StaticMaze
from environments.dynamic_maze import DynamicMaze
from agent.q_learning_agent import QLearningAgent

from strategies.epsilon_greedy import EpsilonGreedyStrategy
from strategies.decay_epsilon import DecayEpsilonStrategy
from strategies.softmax import SoftmaxStrategy

from utils.logger import (
    save_to_csv,
    save_timing_info,
    save_mean_results
)
from utils.plotting import (
    plot_metrics,
    plot_time_runs,
    plot_mean_metrics
)

def run_training(env, agent, episodes=500, max_steps=100):
    """
    Runs Q-learning training for one run.
    Returns lists: rewards[i], steps[i], successes[i] for each episode.
    """
    rewards, steps, successes = [], [], []
    for ep in range(episodes):
        state = env.reset()
        total_reward, step_count, done = 0, 0, False

        while not done and step_count < max_steps:
            action = agent.select_action(state, ep)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward
            step_count += 1

        rewards.append(total_reward)
        steps.append(step_count)
        successes.append(1 if done else 0)

    return rewards, steps, successes

def experiment():
    episodes = 500
    repetitions = 500
    seeds = list(range(repetitions))

    strategies = {
        "epsilon_greedy": EpsilonGreedyStrategy(epsilon=0.1),
        "decay_epsilon": DecayEpsilonStrategy(initial_epsilon=1.0, decay_rate=0.005),
        "softmax": SoftmaxStrategy(initial_temp=1.0, decay_rate=0.005)
    }
    environments = {
        "static": StaticMaze,
        "dynamic": DynamicMaze
    }

    # Prepare folders
    os.makedirs("results/aggregate", exist_ok=True)

    all_timing = []

    for strat_name, strat_obj in strategies.items():
        for env_name, env_cls in environments.items():
            # containers for all runs
            R = np.zeros((repetitions, episodes))
            S = np.zeros((repetitions, episodes))
            U = np.zeros((repetitions, episodes))  # successes
            times = []

            for run_idx, seed in enumerate(seeds):
                # set reproducible seed
                np.random.seed(seed)
                random.seed(seed)

                env = env_cls()
                agent = QLearningAgent(strategy=strat_obj)

                start = time.time()
                r, st, succ = run_training(env, agent, episodes)
                duration = time.time() - start
                """
                # save individual-run data
                save_to_csv(
                    r, st, succ,
                   strat_name,
                   env_name,
                   run_id=run_idx,
                   folder="results/individual"
                )
                plot_metrics(
                 r, st, succ,
                  strat_name,
                   env_name,
                   run_id=run_idx,
                    folder="results/individual"
                )
                """
                # collect for aggregate
                R[run_idx] = r
                S[run_idx] = st
                U[run_idx] = succ
                times.append(duration)
                all_timing.append((strat_name, env_name, seed, duration))

            # compute means and stds
            mean_r = R.mean(axis=0)
            std_r  = R.std(axis=0)
            mean_s = S.mean(axis=0)
            std_s  = S.std(axis=0)
            mean_u = U.mean(axis=0)
            std_u  = U.std(axis=0)
            mean_t = np.mean(times)
            std_t  = np.std(times)

            # save aggregate CSV + plots
            save_mean_results(
                episodes,
                mean_r, std_r,
                mean_s, std_s,
                mean_u, std_u,
                mean_t, std_t,
                strat_name, env_name,
                folder="results/aggregate"
            )
            plot_mean_metrics(
                mean_r, std_r,
                mean_s, std_s,
                mean_u, std_u,
                strat_name, env_name,
                folder="results/aggregate"
            )
            
            plot_time_runs(times, strat_name, env_name, folder="results/aggregate")
    # Save full timing table and bar plot
    save_timing_info(all_timing, filename="results/aggregate/computation_times.csv")
   

if __name__ == "__main__":
    experiment()
