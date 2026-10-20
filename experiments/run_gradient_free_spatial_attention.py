import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import inspect

_old_getfile = inspect.getfile
def _safe_getfile(object):
    try:
        f = _old_getfile(object)
        if not isinstance(f, (str, bytes, os.PathLike)):
            return __file__
        return f
    except Exception:
        return __file__
inspect.getfile = _safe_getfile

_old_getsourcefile = inspect.getsourcefile
def _safe_getsourcefile(object):
    try:
        return _old_getsourcefile(object)
    except Exception:
        return None
inspect.getsourcefile = _safe_getsourcefile

import argparse
import yaml
import torch
import warnings

from targets.ciagan import CIAGANTarget
from attacks.blackbox.gradient_free_spatial_attention_attack import GradientFreeSpatialAttentionAttack
from benchmark.benchmark_runner import BenchmarkRunner
from data_loaders.lfw import LFWDataset

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Run Gradient-Free Spatial Attention Benchmark against CIAGAN")
    parser.add_argument("--config", type=str, default="configs/attacks_ciagan.yaml")
    parser.add_argument("--dataset_path", type=str, required=True, help="Percorso del dataset LFW")
    parser.add_argument("--output_dir", type=str, default="./results_spatial_attention_attack")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo di esecuzione: {device}")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("Caricamento del target CIAGAN...")
    target = CIAGANTarget(models_dir=config['target']['models_dir'], device=device)

    print("Avvio Gradient-Free Spatial Attention Attack...")
    attack = GradientFreeSpatialAttentionAttack(
        target=target,
        detector_wrapper=target.detector_wrapper,
        device=device,
        num_iter=200
    )

    dataset = LFWDataset(root_dir=args.dataset_path)

    runner = BenchmarkRunner(
        target=target,
        attack=attack,
        dataset=dataset,
        device=device
    )

    os.makedirs(args.output_dir, exist_ok=True)
    runner.run_benchmark(
        epsilons=[16.0, 32.0, 64.0, 128.0],
        num_samples=20,
        output_dir=args.output_dir
    )

    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    plot_path = os.path.join(args.output_dir, "spatial_attention_comparison_plot.png")
    runner.visualize_best_attack(csv_path=csv_path, save_path=plot_path)

    print(f"Benchmark Gradient-Free Spatial Attention completato con successo. Risultati salvati in {args.output_dir}")

if __name__ == "__main__":
    main()
