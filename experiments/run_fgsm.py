import os
import torch
import warnings
from torch.utils.data import DataLoader
from torchvision import transforms

from targets.deeprivacy2 import DeepPrivacy2Target
from attacks.fgsm import FGSMAttack
from benchmark.benchmark_runner import BenchmarkRunner
from dataset.lfw import LFWDataset  # Modifica con il percorso del tuo loader LFW se differente

warnings.filterwarnings("ignore")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo di esecuzione: {device}")

    # 1. Inizializzazione del Target (DeepPrivacy2)
    print("Caricamento del target DeepPrivacy2...")
    target = DeepPrivacy2Target(device=device)

    # 2. Inizializzazione dell'attacco FGSM
    print("Inizializzazione dell'attacco FGSM...")
    attack = FGSMAttack(
        detector_wrapper=target.anonymizer.detector,
        dsfd_net=target.anonymizer.detector.face_detector.net.to(device).eval(),
        mean_tensor=target.anonymizer.detector.face_mean.to(device).float().flatten().view(1, 3, 1, 1),
        device=device
    )

    # 3. Caricamento del dataset LFW
    print("Caricamento del dataset LFW...")
    dataset = LFWDataset(root_dir="/kaggle/input/lfw-dataset/lfw_funneled") # Sostituisci con il tuo path dataset reale

    # 4. Configurazione del BenchmarkRunner
    runner = BenchmarkRunner(
        target=target,
        attack=attack,
        dataset=dataset,
        device=device
    )

    # 5. Esecuzione del benchmark (es. test con epsilon fissi)
    output_dir = "./results_fgsm"
    df = runner.run_benchmark(
        epsilons=[2.0, 4.0, 8.0, 16.0, 32.0],
        num_samples=50,
        output_dir=output_dir
    )

    # 6. Generazione del grafico di visualizzazione del miglior attacco riuscito
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    plot_path = os.path.join(output_dir, "fgsm_comparison_plot.png")
    runner.visualize_best_attack(csv_path=csv_path, save_path=plot_path)

if __name__ == "__main__":
    main()
