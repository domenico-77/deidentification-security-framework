import sys
from pathlib import Path

# Aggiunge la directory root del progetto a sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
import yaml
import torch
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
    parser.add_argument("--train_samples", type=int, default=100, help="Numero di immagini da usare per calcolare la UAP")
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

    # 1. FASE DI FIT: Calcolo della Universal Adversarial Perturbation
    # Usiamo un epsilon di default (es. 16.0) per la palla L-inf della UAP
    epsilon_uap = 8.0
    print("Avvio calcolo della Universal Adversarial Perturbation (UAP)...")
    attack.fit(dataloader=train_loader, epsilon=epsilon_uap, alpha=2, epochs=8, max_iter_per_img=10)

    # 2. FASE DI BENCHMARK: Valutazione delle performance su diverse epsilon
    print("\nAvvio Benchmark UAP sul dataset di test...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    runner = BenchmarkRunner(target, attack, full_dataset)
    
    # Esegue il benchmark testando i diversi valori di epsilon consentiti dalla UAP
    runner.run_benchmark(epsilons=[4.0, 8.0, 16.0, 32.0], num_samples=100, output_dir=args.output_dir)

    # Generazione automatica del grafico di confronto
    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    plot_path = os.path.join(args.output_dir, "comparison_plot.png")
    
    print("\nGenerazione del grafico di confronto per UAP...")
    if os.path.exists(csv_path):
        runner.visualize_best_attack(csv_path=csv_path, save_path=plot_path)

    print(f"Benchmark UAP completato. Risultati salvati in {args.output_dir}")


if __name__ == "__main__":
    main()
