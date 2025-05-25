# save_results.py
import pandas as pd
import os

def save_to_csv(rewards, success_list, steps_list, strategy, env_type):
    """
    Saves the Q-learning metrics (rewards, success, steps) per episode to a CSV file.
    """
    # Create a dataframe from the lists
    data = {
        'Episode': range(1, len(rewards)+1),
        'Reward': rewards,
        'Success': success_list,
        'Steps': steps_list
    }
    df = pd.DataFrame(data)

    # Ensure the strategy subfolder exists
    os.makedirs(strategy, exist_ok=True)

    # Construct a filename that includes the strategy and env type
    filename = os.path.join(strategy, f"{strategy}_{env_type}_results.csv")

    # Save to CSV without the default pandas index column
    df.to_csv(filename, index=False)
    print(f"Saved results to {filename}")
