import sys
from pathlib import Path
import random
import numpy as np
import torch
import os
import argparse
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loaders.lfw import LFWDataset
from targets.deeprivacy2 import DeepPrivacy2Target
from detectors.dsfd import DSFDDetector
from attacks.uap import UAPAttack
from benchmark.benchmark_runner import BenchmarkRunner

def set_seed(seed_value=42):
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
    parser = argparse.ArgumentParser(description="Run UAP Benchmark across multiple epsilons")
    parser.add_argument("--config", type=str, default="configs/attacks_dp2.yaml")
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Inizializzazione Modelli
    target = DeepPrivacy2Target(models_dir=config['target']['models_dir'])
    anonymizer = target.pipeline
    if hasattr(target, "anonymizer"):
        anonymizer = target.anonymizer

    detector_wrapper = anonymizer.detector
    dsfd_net = detector_wrapper.face_detector.net.to(device).eval()
    mean_tensor = detector_wrapper.face_mean.to(device).float().flatten().view(1, 3, 1, 1)

    # Istanziazione UAP Attack
    attack = UAPAttack(
        detector_wrapper=detector_wrapper,
        dsfd_net=dsfd_net,
        mean_tensor=mean_tensor,
        device=device
    )

    dataset = LFWDataset(root_dir=args.dataset_path)
    
    # Creiamo un sottoinsieme di training dedicato (es. 100 campioni)
    train_indices = list(range(100))
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)

    epsilons_list = [2.0, 4.0, 8.0, 16.0, 24.0, 32.0]
    output_dir = "./results_uap"
    os.makedirs(output_dir, exist_ok=True)

    print("=== Avvio Benchmark UAP Multi-Epsilon ===")
    
    # Iteriamo su ciascun epsilon ricalcolando la UAP dedicata
    for eps in epsilons_list:
        print(f"\n--- Fase 1: Ottimizzazione UAP per epsilon = {eps} ---")
        attack.fit(
            dataloader=train_loader, 
            epsilon=eps, 
            alpha=2.0, 
            epochs=5,           # Epoche per singolo epsilon (regolabili)
            max_iter_per_img=10
        )

        print(f"--- Fase 2: Valutazione sul Test Set con epsilon = {eps} ---")
        # Eseguiamo il benchmark filtrando specificamente per l'epsilon corrente
        runner = BenchmarkRunner(target, attack, dataset)
        runner.run_benchmark(epsilons=[eps], num_samples=100, output_dir=output_dir)

    # Generazione finale del grafico riassuntivo complessivo
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    plot_path = os.path.join(output_dir, "uap_comparison_plot.png")
    
    print("\nGenerazione del grafico riassuntivo UAP...")
    if os.path.exists(csv_path):
        runner.visualize_best_attack(csv_path=csv_path, save_path=plot_path)

    print(f"Benchmark UAP completato con successo. Risultati salvati in {output_dir}")

if __name__ == "__main__":
    main()
