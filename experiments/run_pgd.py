import sys
from pathlib import Path
import random
import numpy as np
import torch
# Aggiunge la directory padre di 'experiments' (la root del progetto) a sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
import yaml
import torch

from data_loaders.lfw import LFWDataset
from targets.deeprivacy2 import DeepPrivacy2Target
from detectors.dsfd import DSFDDetector
from attacks.pgd import PGDAttack
from benchmark.benchmark_runner import BenchmarkRunner

def set_seed(seed_value=42):
    """Fissa i seed per garantire la riproducibilità degli esperimenti."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="Run PGD Benchmark against DeepPrivacy2")
    parser.add_argument("--config", type=str, default="configs/pgd.yaml")
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Inizializzazione Modelli
    target = DeepPrivacy2Target(models_dir=config['target']['models_dir'])
    detector = DSFDDetector(target.pipeline)

    # Estrazione corretta dei componenti dal target/pipeline di DeepPrivacy2
    anonymizer = target.pipeline  # oppure target.anonymizer a seconda di come è strutturato il target
    if hasattr(target, "anonymizer"):
        anonymizer = target.anonymizer

    detector_wrapper = anonymizer.detector
    dsfd_net = detector_wrapper.face_detector.net.to(device).eval()
    mean_tensor = detector_wrapper.face_mean.to(device).float().flatten().view(1, 3, 1, 1)
    
    # Istanziazione corretta della classe PGDAttack
    attack = PGDAttack(
        detector_wrapper=detector_wrapper,
        dsfd_net=dsfd_net,
        mean_tensor=mean_tensor,
        device=device
    )

    dataset = LFWDataset(root_dir=args.dataset_path)
    runner = BenchmarkRunner(target, attack, dataset)

    print("Avvio attacco PGD...")
    output_dir = "./results"
    
    # 1. Esecuzione del benchmark (con log in tempo reale grazie al nuovo logger)
    runner.run_benchmark(epsilons=[2.0, 4.0, 8.0, 16.0, 24.0, 32.0], num_samples=100, output_dir=output_dir)
    #runner.run_benchmark(epsilons=[8.0], num_samples=5, output_dir=output_dir)

    # 2. Generazione automatica del grafico di confronto (Post-analisi)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    plot_path = os.path.join(output_dir, "comparison_plot.png")
    
    print("\nGenerazione del grafico di confronto per il miglior attacco...")
    runner.visualize_best_attack(csv_path=csv_path, save_path=plot_path)

    print(f"Benchmark completato con successo. Risultati e grafici salvati in {output_dir}")


if __name__ == "__main__":
    main()
