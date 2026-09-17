import sys
from pathlib import Path

# Aggiunge la root del progetto a sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import inspect

# --- PATCH DI SICUREZZA ROBUSTA PER INSPECT (PYTHON 3.12) ---
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
# -------------------------------------------------------------

import argparse
import yaml
import torch
import warnings

from targets.deeprivacy2 import DeepPrivacy2Target
from attacks.fgsm import FGSMAttack
from benchmark.benchmark_runner import BenchmarkRunner
from data_loaders.lfw import LFWDataset

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Run FGSM Benchmark against DeepPrivacy2")
    parser.add_argument("--config", type=str, default="configs/pgd.yaml")
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo di esecuzione: {device}")

    # 1. Inizializzazione del Target
    print("Caricamento del target DeepPrivacy2...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    target = DeepPrivacy2Target(models_dir=config['target']['models_dir'])

    # 2. Estrazione sicura dei componenti della pipeline per FGSM
    anonymizer = target.pipeline if hasattr(target, "pipeline") else target.anonymizer
    if hasattr(target, "anonymizer"):
        anonymizer = target.anonymizer

    detector_wrapper = anonymizer.detector
    dsfd_net = detector_wrapper.face_detector.net.to(device).eval()
    mean_tensor = detector_wrapper.face_mean.to(device).float().flatten().view(1, 3, 1, 1)

    # 3. Inizializzazione dell'attacco FGSM
    print("Inizializzazione dell'attacco FGSM...")
    attack = FGSMAttack(
        detector_wrapper=detector_wrapper,
        dsfd_net=dsfd_net,
        mean_tensor=mean_tensor,
        device=device
    )

    # 4. Caricamento del dataset LFW
    print("Caricamento del dataset LFW...")
    dataset = LFWDataset(root_dir=args.dataset_path)

    # 5. Configurazione del BenchmarkRunner
    runner = BenchmarkRunner(
        target=target,
        attack=attack,
        dataset=dataset,
        device=device
    )

    # 6. Esecuzione del benchmark
    output_dir = "./results_fgsm"
    runner.run_benchmark(
        epsilons=[2.0, 4.0, 8.0, 16.0, 32.0],
        num_samples=5,
        output_dir=output_dir
    )

    # 7. Generazione del grafico di visualizzazione
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    plot_path = os.path.join(output_dir, "fgsm_comparison_plot.png")
    runner.visualize_best_attack(csv_path=csv_path, save_path=plot_path)

    print(f"Benchmark FGSM completato con successo. Risultati salvati in {output_dir}")

if __name__ == "__main__":
    main()
