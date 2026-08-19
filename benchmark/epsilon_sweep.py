import matplotlib.pyplot as plt
import pandas as pd


def plot_epsilon_sweep(csv_path: str, save_path: str = "./results/ASR_vs_epsilon.png"):
    df = pd.read_csv(csv_path)
    sweep = df.groupby("epsilon")["evaded"].mean() * 100

    plt.figure(figsize=(8, 5))
    plt.plot(sweep.index, sweep.values, marker='o', color='red', linewidth=2)
    plt.title("Attack Success Rate (ASR) vs Epsilon Threshold")
    plt.xlabel("Epsilon (L_inf bound)")
    plt.ylabel("ASR (%)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()