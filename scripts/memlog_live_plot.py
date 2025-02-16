import os

import pandas as pd
import seaborn as sns
import typer
from matplotlib import pyplot as plt

sns.set_style("whitegrid")


def main(log_path: str):
    if not os.path.exists(log_path):
        print("Invalid log path")
        raise typer.Exit(-1)

    fig, ax = plt.subplots(1, 1)
    n_data_points = 0
    n_retries = 0
    max_retries = 5
    while True:
        try:
            data = pd.read_csv(log_path)
        except Exception as e:
            print(f"Unable to load data: {e}")
            raise typer.Exit(-1)
        if "datetime" not in data.columns or "ram_used_gb" not in data.columns:
            print("Bad log file")
            raise typer.Exit(-1)

        if len(data) == n_data_points:
            n_retries += 1
        else:
            n_data_points = len(data)
            n_retries = 0
        if n_retries == max_retries:
            print("No more data found")
            raise typer.Exit()

        data["datetime"] = pd.to_datetime(data["datetime"])
        ax.clear()
        sns.lineplot(data, x="datetime", y="ram_used_gb", ax=ax)
        ax.set_title("Simulation RAM Usage", size=16, weight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("RAM Used (GB)")
        fig.canvas.draw()
        plt.pause(1)


if __name__ == "__main__":
    typer.run(main)
