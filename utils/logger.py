# utils/logger.py
import pandas as pd
import os

def save_to_csv(rewards, steps, successes,
                strategy_name, env_name,
                folder="results"):
    """
    Saves one run's metrics to CSV.
    """
    df = pd.DataFrame({
        "Episode": range(1, len(rewards) + 1),
        "Reward": rewards,
        "Steps": steps,
        "Success": successes
    })
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/{strategy_name}_{env_name}_run.csv"
    df.to_csv(filename, index=False)
    print(f"Saved individual run: {filename}")

def save_mean_results(episodes,
                      mean_r, std_r,
                      mean_s, std_s,
                      mean_u, std_u,
                      mean_t, std_t,
                      strategy_name, env_name,
                      folder="results"):
    """
    Saves aggregate (mean ± std) metrics to CSV.
    """
    df = pd.DataFrame({
        "Episode": range(1, episodes + 1),
        "MeanReward": mean_r,
        "StdReward": std_r,
        "MeanSteps": mean_s,
        "StdSteps": std_s,
        "MeanSuccess": mean_u,
        "StdSuccess": std_u
    })
    os.makedirs(folder, exist_ok=True)
    results_csv = f"{folder}/{strategy_name}_{env_name}_aggregate.csv"
    df.to_csv(results_csv, index=False)
    print(f"Saved aggregate CSV: {results_csv}")

    # Also save timing summary
    timing_df = pd.DataFrame([{
        "Strategy": strategy_name,
        "Environment": env_name,
        "MeanTime": mean_t,
        "StdTime": std_t
    }])
    timing_csv = f"{folder}/{strategy_name}_{env_name}_time.csv"
    timing_df.to_csv(timing_csv, index=False)
    print(f"Saved timing summary: {timing_csv}")

def save_timing_info(timing_list, filename="results/computation_times.csv"):
    """
    Saves full timing info: one row per run (with seed).
    timing_list entries: (strategy, env, seed, time)
    """
    df = pd.DataFrame(timing_list,
                      columns=["Strategy", "Environment", "Seed", "TimeSeconds"])
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Saved all run timings: {filename}")
