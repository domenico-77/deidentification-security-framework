import argparse
import yaml
import torch
from datasets.lfw import LFWDataset
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
    attack = PGDAttack(detector)

    dataset = LFWDataset(root_dir=args.dataset_path)
    runner = BenchmarkRunner(target, attack, dataset)

    print("Avvio del Benchmark PGD...")
    runner.run_benchmark(epsilons=[2.0, 4.0, 8.0, 16.0, 24.0, 32.0], num_samples=100)
    print("Benchmark completato con successo. Risultati salvati in ./results")


if __name__ == "__main__":
    main()