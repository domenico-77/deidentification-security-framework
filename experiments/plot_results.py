import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_plots(csv_path="./results/benchmark_results.csv", output_dir="./results"):
    if not os.path.exists(csv_path):
        print(f"File non trovato: {csv_path}. Esegui prima il benchmark!")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Calcola statistiche aggregate per ogni epsilon
    grouped = df.groupby("epsilon").agg({
        "evaded": "mean",          # ASR (Attack Success Rate)
        "psnr": "mean",            # PSNR medio
        "pipeline_mse": "mean",    # MSE medio
        "iterations": "mean"       # Iterazioni medie
    }).reset_index()

    grouped["asr_percent"] = grouped["evaded"] * 100.0

    print("Statistiche aggregate per Epsilon:")
    print(grouped)

    # --- GRAFICO 1: ASR vs Epsilon ---
    plt.figure(figsize=(8, 5))
    plt.plot(grouped["epsilon"], grouped["asr_percent"], marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
    plt.title("Attack Success Rate (ASR) vs Epsilon ($L_\\infty$)", fontsize=12, fontweight='bold')
    plt.xlabel("Epsilon ($\epsilon$)", fontsize=10)
    plt.ylabel("ASR (%)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-5, 105)
    
    plot1_path = os.path.join(output_dir, "asr_vs_epsilon.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafico ASR salvato in: {plot1_path}")

    # --- GRAFICO 2: PSNR vs Epsilon (Trade-off Qualità/Robustezza) ---
    plt.figure(figsize=(8, 5))
    plt.plot(grouped["epsilon"], grouped["psnr"], marker='s', linestyle='-', color='r', linewidth=2, markersize=6)
    plt.title("Qualità Percettiva (PSNR) vs Epsilon", fontsize=12, fontweight='bold')
    plt.xlabel("Epsilon ($\epsilon$)", fontsize=10)
    plt.ylabel("PSNR medio (dB)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot2_path = os.path.join(output_dir, "psnr_vs_epsilon.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafico PSNR salvato in: {plot2_path}")

if __name__ == "__main__":
    generate_plots()
