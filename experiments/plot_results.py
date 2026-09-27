import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def print_aggregated_stats():
    base_dir = "/kaggle/working/deidentification-security-framework/results_baseline_uap"
    std_csv = os.path.join(base_dir, "standard", "benchmark_results.csv")
    ded_csv = os.path.join(base_dir, "dedicated_per_epsilon", "benchmark_results_dedicated.csv")

    def _process_csv(csv_path, title):
        if not os.path.exists(csv_path):
            print(f"File non trovato: {csv_path}")
            return
            
        df = pd.read_csv(csv_path)
        df['evaded_num'] = df['evaded'].astype(float)
        
        agg_df = df.groupby('epsilon').agg(
            evaded=('evaded_num', 'mean'),
            psnr=('psnr', 'mean'),
            pipeline_mse=('pipeline_mse', 'mean'),
            iterations=('iterations', 'mean'),
            asr_percent=('evaded_num', lambda x: x.mean() * 100)
        ).reset_index()
        
        print(f"\nStatistiche aggregate per Epsilon ({title}):")
        print(agg_df.to_string(index=True, formatters={
            'evaded': '{:.2f}'.format,
            'psnr': '{:.6f}'.format,
            'pipeline_mse': '{:.6f}'.format,
            'iterations': '{:.1f}'.format,
            'asr_percent': '{:.1f}'.format
        }))

    _process_csv(std_csv, "Standard")
    _process_csv(ded_csv, "Dedicati per Epsilon")

def generate_plots(csv_path="./results/benchmark_results.csv", output_dir="./results"):
    if not os.path.exists(csv_path):
        print(f"File non trovato: {csv_path}. Esegui prima il benchmark!")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Calcola statistiche aggregate per ogni epsilon
    grouped = df.groupby("epsilon").agg({
        "evaded": "mean",       # ASR (Attack Success Rate)
        "psnr": "mean",           # PSNR medio
        "pipeline_mse": "mean",   # MSE medio
        "iterations": "mean"      # Iterazioni medie
    }).reset_index()

    grouped["asr_percent"] = grouped["evaded"] * 100.0

    print("Statistiche aggregate per Epsilon:")
    print(grouped)

    # --- GRAFICO 1: ASR vs Epsilon ---
    plt.figure(figsize=(8, 5))
    plt.plot(grouped["epsilon"], grouped["asr_percent"], marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
    plt.title("Attack Success Rate (ASR) vs Epsilon ($L_\\infty$)", fontsize=12, fontweight='bold')
    plt.xlabel(r"Epsilon ($\epsilon$)", fontsize=10)
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
    plt.xlabel(r"Epsilon ($\epsilon$)", fontsize=10)
    plt.ylabel("PSNR medio (dB)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot2_path = os.path.join(output_dir, "psnr_vs_epsilon.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafico PSNR salvato in: {plot2_path}")

    # --- GRAFICO 3: Distribuzione delle Norme L2 ---
    if "l2" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(df["l2"].dropna(), bins=15, color='teal', edgecolor='black', alpha=0.8)
        mean_l2 = df["l2"].mean()
        plt.axvline(mean_l2, color='red', linestyle='dashed', linewidth=2, label=f'Media L2: {mean_l2:.2f}')
        plt.title("Distribuzione delle Norme L2 delle Perturbazioni", fontsize=12, fontweight='bold')
        plt.xlabel("Distanza L2", fontsize=10)
        plt.ylabel("Frequenza Campioni", fontsize=10)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plot3_path = os.path.join(output_dir, "l2_distribution.png")
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Istogramma distribuzione L2 salvato in: {plot3_path}")

    # --- GRAFICO 4: Efficienza delle Iterazioni al Successo ---
    plt.figure(figsize=(8, 5))
    plt.plot(grouped["epsilon"], grouped["iterations"], marker='^', linestyle='-', color='darkgreen', linewidth=2, markersize=6)
    plt.title("Numero Medio di Iterazioni PGD al Successo", fontsize=12, fontweight='bold')
    plt.xlabel(r"Epsilon ($\epsilon$)", fontsize=10)
    plt.ylabel("Iterazioni medie", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot4_path = os.path.join(output_dir, "iterations_vs_epsilon.png")
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafico iterazioni medie salvato in: {plot4_path}")

    # --- GRAFICO 5: Curva di Convergenza/MSE Pipeline per Epsilon ---
    plt.figure(figsize=(8, 5))
    plt.plot(grouped["epsilon"], grouped["pipeline_mse"], marker='d', linestyle='-', color='purple', linewidth=2, markersize=6)
    plt.title("MSE della Pipeline vs Epsilon", fontsize=12, fontweight='bold')
    plt.xlabel(r"Epsilon ($\epsilon$)", fontsize=10)
    plt.ylabel("Pipeline MSE medio", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot5_path = os.path.join(output_dir, "pipeline_mse_vs_epsilon.png")
    plt.savefig(plot5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafico Pipeline MSE salvato in: {plot5_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gestione plotting e statistiche UAP")
    parser.add_argument("--uap", action="store_true", help="Mostra le statistiche aggregate formattate per standard e dedicati")
    parser.add_argument("--csv_path", type=str, default="./results/benchmark_results.csv")
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    if args.stats_only:
        print_aggregated_stats()
    else:
        generate_plots(csv_path=args.csv_path, output_dir=args.output_dir)
