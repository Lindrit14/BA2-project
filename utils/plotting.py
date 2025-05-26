# utils/plotting.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_metrics(rewards, steps, successes,
                 strategy_name, env_name,
                 folder="results"):
    os.makedirs(folder, exist_ok=True)

    # Reward
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.title(f"{strategy_name} on {env_name}: Reward")
    plt.savefig(f"{folder}/{strategy_name}_{env_name}_reward.png")
    plt.close()

    # Steps
    plt.figure()
    plt.plot(steps)
    plt.xlabel("Episode"); plt.ylabel("Steps")
    plt.title(f"{strategy_name} on {env_name}: Steps")
    plt.savefig(f"{folder}/{strategy_name}_{env_name}_steps.png")
    plt.close()

    # Success rate
    plt.figure()
    rate = np.cumsum(successes) / np.arange(1, len(successes)+1)
    plt.plot(rate)
    plt.xlabel("Episode"); plt.ylabel("Success Rate")
    plt.title(f"{strategy_name} on {env_name}: Success Rate")
    plt.savefig(f"{folder}/{strategy_name}_{env_name}_success.png")
    plt.close()

def plot_mean_metrics(mean_r, std_r,
                      mean_s, std_s,
                      mean_u, std_u,
                      strategy_name, env_name,
                      folder="results"):
    os.makedirs(folder, exist_ok=True)
    episodes = len(mean_r)
    x = np.arange(1, episodes+1)

    # Mean ± std reward
    plt.figure()
    plt.plot(x, mean_r, label="Mean")
    plt.fill_between(x, mean_r - std_r, mean_r + std_r, alpha=0.2)
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.title(f"{strategy_name} on {env_name}: Reward ± STD")
    plt.savefig(f"{folder}/{strategy_name}_{env_name}_reward_agg.png")
    plt.close()

    # Mean ± std steps
    plt.figure()
    plt.plot(x, mean_s, label="Mean")
    plt.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.2)
    plt.xlabel("Episode"); plt.ylabel("Steps")
    plt.title(f"{strategy_name} on {env_name}: Steps ± STD")
    plt.savefig(f"{folder}/{strategy_name}_{env_name}_steps_agg.png")
    plt.close()

    # Mean ± std success
    plt.figure()
    plt.plot(x, mean_u, label="Mean")
    plt.fill_between(x, mean_u - std_u, mean_u + std_u, alpha=0.2)
    plt.xlabel("Episode"); plt.ylabel("Success Rate")
    plt.title(f"{strategy_name} on {env_name}: Success ± STD")
    plt.savefig(f"{folder}/{strategy_name}_{env_name}_success_agg.png")
    plt.close()



def plot_time_runs(times, strategy_name, env_name, folder="results/aggregate", x_interval=100):
    """
    times: list of floats (one per run)
    Plots a scatter of each run’s duration and a horizontal line for the mean.
    """
    os.makedirs(folder, exist_ok=True)
    num_runs = len(times)
    runs = list(range(1, num_runs + 1))
    mean_time = sum(times) / num_runs

    plt.figure(figsize=(6, 4))
    # scatter of individual runs
    plt.scatter(runs, times, label="Run time", zorder=2)
    # horizontal mean line
    plt.hlines(mean_time, xmin=1, xmax=len(times), colors="red",
               linestyles="dashed", label=f"Mean = {mean_time:.3f}s", zorder=1)
    
    
    # Compute tick positions: every x_interval plus the last run
    ticks = list(range(1, num_runs + 1, x_interval))
    if num_runs not in ticks:
        ticks.append(num_runs)
    plt.xticks(ticks)


    plt.xlabel("Run #")
    plt.ylabel("Time (s)")
    plt.title(f"{strategy_name} on {env_name} — run times")
    plt.legend()
    plt.tight_layout()

    filename = f"{folder}/{strategy_name}_{env_name}_time_runs.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved per-run time plot: {filename}")