import sys
from pathlib import Path

# Aggiunge la directory root del progetto a sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset

from data_loaders.lfw import LFWDataset
from targets.deeprivacy2 import DeepPrivacy2Target
from detectors.dsfd import DSFDDetector
from attacks.uap import UAPAttack
from benchmark.benchmark_runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Run UAP Benchmark against DeepPrivacy2")
    parser.add_argument("--config", type=str, default="configs/attacks_dp2.yaml")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--train_samples", type=int, default=50, help="Numero di immagini da usare per calcolare la UAP")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Inizializzazione Modelli
    target = DeepPrivacy2Target(models_dir=config.get('models_dir', None))
    detector = DSFDDetector(target.pipeline)

    # Estrazione dei componenti dal target/pipeline di DeepPrivacy2
    anonymizer = target.pipeline
    if hasattr(target, "anonymizer"):
        anonymizer = target.anonymizer

    detector_wrapper = anonymizer.detector
    dsfd_net = detector_wrapper.face_detector.net.to(device).eval()
    mean_tensor = detector_wrapper.face_mean.to(device).float().flatten().view(1, 3, 1, 1)
    
    # Istanziazione di UAPAttack
    attack = UAPAttack(
        detector_wrapper=detector_wrapper,
        dsfd_net=dsfd_net,
        mean_tensor=mean_tensor,
        device=device
    )

    # Caricamento del dataset completo
    full_dataset = LFWDataset(root_dir=args.dataset_path)

    print(f"Preparazione del sottoinsieme di training per la UAP ({args.train_samples} campioni)...")
    train_indices = list(range(min(args.train_samples, len(full_dataset))))
    train_subset = Subset(full_dataset, train_indices)
    
    # Creazione del DataLoader per il fitting della UAP
    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)

    os.makedirs(args.output_dir, exist_ok=True)
    epsilons_list = [2.0, 4.0, 8.0, 16.0, 24.0, 32.0]

    # =========================================================================
    # FASE 1: Esecuzione Standard Originale (UAP unica a epsilon=16.0)
    # =========================================================================
    print("\n=================================================================")
    print(">>> FASE 1: Esecuzione Standard (UAP unica con epsilon=16.0)")
    print("=================================================================")
    
    epsilon_uap = 16.0
    print("Avvio calcolo della Universal Adversarial Perturbation (UAP) standard...")
    attack.fit(dataloader=train_loader, epsilon=epsilon_uap, alpha=2, epochs=8, max_iter_per_img=15)

    print("\nAvvio Benchmark UAP standard sul dataset di test...")
    standard_output_dir = os.path.join(args.output_dir, "standard")
    os.makedirs(standard_output_dir, exist_ok=True)
    
    runner = BenchmarkRunner(target, attack, full_dataset)
    runner.run_benchmark(epsilons=epsilons_list, num_samples=100, output_dir=standard_output_dir)

    # Grafico standard
    csv_path_std = os.path.join(standard_output_dir, "benchmark_results.csv")
    plot_path_std = os.path.join(standard_output_dir, "comparison_plot.png")
    if os.path.exists(csv_path_std):
        runner.visualize_best_attack(csv_path=csv_path_std, save_path=plot_path_std)

    # =========================================================================
    # FASE 2: Plus Aggiuntivo (Ricalcolo UAP dedicata per ciascun epsilon)
    # =========================================================================
    print("\n=================================================================")
    print(">>> FASE 2: Plus Aggiuntivo - UAP Ricalcolata Dedicata per ogni Epsilon")
    print("=================================================================")
    
    dedicated_output_dir = os.path.join(args.output_dir, "dedicated_per_epsilon")
    os.makedirs(dedicated_output_dir, exist_ok=True)
    
    all_dedicated_results = []

    for eps in epsilons_list:
        print(f"\n--- Ricalcolo UAP dedicata per epsilon = {eps} ---")
        # Ricalcola la UAP vincolata esattamente a questo epsilon
        attack.fit(dataloader=train_loader, epsilon=eps, alpha=2, epochs=5, max_iter_per_img=12)

        print(f"Valutazione sul test set con UAP dedicata (epsilon = {eps})...")
        runner_ded = BenchmarkRunner(target, attack, full_dataset)
        runner_ded.run_benchmark(epsilons=[eps], num_samples=100, output_dir=dedicated_output_dir)
        
        # Legge il risultato parziale e lo raccoglie
        par_csv = os.path.join(dedicated_output_dir, "benchmark_results.csv")
        if os.path.exists(par_csv):
            df_part = pd.read_csv(par_csv)
            all_dedicated_results.append(df_part)

    # Unisce e salva i risultati completi della fase dedicata
    if all_dedicated_results:
        final_dedicated_df = pd.concat(all_dedicated_results, ignore_index=True)
        final_csv_path = os.path.join(dedicated_output_dir, "benchmark_results_dedicated.csv")
        final_dedicated_df.to_csv(final_csv_path, index=False)
        
        final_plot_path = os.path.join(dedicated_output_dir, "comparison_plot_dedicated.png")
        runner.visualize_best_attack(csv_path=final_csv_path, save_path=final_plot_path)
        print(f"\n[INFO] Plus aggiuntivo completato! Risultati dedicati salvati in {dedicated_output_dir}")

    print(f"\nBenchmark UAP completo terminato con successo. Directory principale: {args.output_dir}")


if __name__ == "__main__":
    main()
