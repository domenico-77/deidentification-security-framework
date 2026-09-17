import sys
from pathlib import Path

# Aggiunge la directory root del progetto a sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
import yaml
import torch

from data_loaders.lfw import LFWDataset
from targets.deeprivacy2 import DeepPrivacy2Target
from detectors.dsfd import DSFDDetector
from attacks.deepfool import DeepFoolAttack
from benchmark.benchmark_runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Run DeepFool Benchmark against DeepPrivacy2")
    parser.add_argument("--config", type=str, default="configs/pgd.yaml")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
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
    
    # Istanziazione di DeepFoolAttack
    attack = DeepFoolAttack(
        detector_wrapper=detector_wrapper,
        dsfd_net=dsfd_net,
        mean_tensor=mean_tensor,
        device=device
    )

    dataset = LFWDataset(root_dir=args.dataset_path)
    runner = BenchmarkRunner(target, attack, dataset)

    print("Avvio del Benchmark DeepFool...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # DeepFool calcola la perturbazione minima dinamicamente (non usa epsilons fissi)
    # Passiamo una lista fittizia o gestiamo il runner in base alla struttura del tuo BenchmarkRunner
    runner.run_benchmark(epsilons=[0.0], num_samples=5, output_dir=args.output_dir)

    # Generazione automatica del grafico di confronto
    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    plot_path = os.path.join(args.output_dir, "comparison_plot.png")
    
    print("\nGenerazione del grafico di confronto per DeepFool...")
    if os.path.exists(csv_path):
        runner.visualize_best_attack(csv_path=csv_path, save_path=plot_path)

    print(f"Benchmark DeepFool completato. Risultati salvati in {args.output_dir}")


if __name__ == "__main__":
    main()
