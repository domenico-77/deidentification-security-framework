import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from metrics.perturbation import calculate_perturbation_metrics
from utils.logger import print_run_header, print_attack_results

class BenchmarkRunner:
    def __init__(self, target, attack, dataset, device="cuda"):
        self.target = target
        self.attack = attack
        self.dataset = dataset
        self.device = device
        
        # Recupera il wrapper del detector dal target o dall'anonymizer
        anonymizer = target.anonymizer if hasattr(target, "anonymizer") else target.pipeline
        self.detector_wrapper = anonymizer.detector
        self.dsfd_net = self.detector_wrapper.face_detector.net.to(device).eval()
        self.mean_tensor = self.detector_wrapper.face_mean.to(device).float().flatten().view(1, 3, 1, 1)

    def count_faces(self, img_tensor):
        detector_input = img_tensor.detach().byte().float()
        with torch.no_grad():
            detections = self.detector_wrapper(detector_input)
        if len(detections) > 0 and detections[0] is not None:
            return len(detections[0])
        return 0

    def run_benchmark(self, epsilons=[2.0, 4.0, 8.0, 16.0, 24.0, 32.0], num_samples=100, output_dir="./results"):
        os.makedirs(output_dir, exist_ok=True)
        results = []

        print("Avvio del Benchmark PGD...")
        max_samples = min(num_samples, len(self.dataset)) if hasattr(self.dataset, "__len__") else num_samples
        total_expected = len(epsilons) * max_samples
        current_run_count = 0

        for eps in epsilons:
            print(f"\nInizio test con epsilon = {eps}...")
            
            for idx in tqdm(range(max_samples)):
                start_time = time.time()
                
                batch = self.dataset[idx]
                img_orig_tensor = batch[0].float().to(self.device) if isinstance(batch, (list, tuple)) else batch.float().to(self.device)
                image_name = f"sample_{idx}.jpg"
                
                clean_faces = self.count_faces(img_orig_tensor)
                if clean_faces == 0:
                    continue  # Salta immagini senza volti rilevati in partenza

                # Esecuzione dell'attacco PGD tramite l'istanza della classe
                img_adv, success, succ_iter = self.attack.perturb(
                    img_orig_tensor=img_orig_tensor, 
                    epsilon=eps
                )

                adv_faces = self.count_faces(img_adv)
                evaded = (adv_faces == 0)

                # Calcolo metriche di perturbazione
                metrics = calculate_perturbation_metrics(img_orig_tensor, img_adv)
                elapsed_seconds = time.time() - start_time
                current_run_count += 1

                # Stampa dei log dettagliati in tempo reale
                print_run_header(idx + 1, max_samples, image_name, eps)
                print_attack_results(
                    detector_evasion=evaded,
                    pipeline_bypass=True,
                    attack_success=evaded,
                    success_iteration=succ_iter,
                    metrics=metrics,
                    pipeline_mse=metrics["mse"],
                    elapsed_seconds=elapsed_seconds,
                    current_run_idx=current_run_count,
                    total_expected=total_expected
                )

                results.append({
                    "image": image_name,
                    "epsilon": eps,
                    "evaded": evaded,
                    "detector_evasion": evaded,
                    "pipeline_mse": metrics["mse"],
                    "l2": metrics["l2"],
                    "linf": metrics["linf"],
                    "psnr": metrics["psnr"],
                    "iterations": succ_iter if succ_iter is not None else 150
                })

        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "benchmark_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nBenchmark completato con successo. Risultati salvati in {csv_path}")
        return df

    def visualize_best_attack(self, csv_path, save_path=None):
        """Metodo interno per la gestione dei risultati e dei plot finali."""
        if not os.path.exists(csv_path):
            print(f"[WARNING] File CSV non trovato: {csv_path}")
            return
            
        df = pd.read_csv(csv_path)
        successful = df[df["evaded"] == True] if "evaded" in df.columns else df.iloc[:0]
        
        if successful.empty:
            print("[WARNING] Nessun attacco riuscito trovato nel CSV per la visualizzazione.")
            return
        
        best_run = successful.sort_values(by="epsilon").iloc[0]
        print(f"\nGenerazione visualizzazione per la migliore run: Epsilon={best_run['epsilon']}")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].set_title(f"Target (Eps: {best_run['epsilon']})")
        axes[0].axis("off")
        axes[1].set_title("DeepPrivacy2 (Baseline)")
        axes[1].axis("off")
        axes[2].set_title("DeepPrivacy2 (Adversarial)")
        axes[2].axis("off")
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Grafico salvato in: {save_path}")
        plt.show()
        plt.close()
