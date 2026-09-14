import sys
from pathlib import Path

# Aggiunge la directory padre di 'experiments' (la root del progetto) a sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import yaml
import torch

from data_loaders.lfw import LFWDataset
from targets.deeprivacy2 import DeepPrivacy2Target
from detectors.dsfd import DSFDDetector
from attacks.pgd import PGDAttack
from benchmark.benchmark_runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Run PGD Benchmark against DeepPrivacy2")
    parser.add_argument("--config", type=str, default="configs/pgd.yaml")
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Inizializzazione Modelli
    target = DeepPrivacy2Target(models_dir=config['target']['models_dir'])
    detector = DSFDDetector(target.pipeline)
    # Estrazione dei componenti necessari per PGDAttack
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

    print("Avvio del Benchmark PGD...")
    #runner.run_benchmark(epsilons=[2.0, 4.0, 8.0, 16.0, 24.0, 32.0], num_samples=100)
    runner.run_benchmark(epsilons=[8.0], num_samples=5)
    print("Benchmark completato con successo. Risultati salvati in ./results")


if __name__ == "__main__":
    main()
