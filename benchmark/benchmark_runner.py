import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

class BenchmarkRunner:
    def __init__(self, anonymizer, device="cuda"):
        self.anonymizer = anonymizer
        self.device = device
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

    def run_benchmark(self, dataset_loader, epsilons=[2.0, 4.0, 8.0, 16.0, 24.0, 32.0], alpha=2.0, iterations=150, output_dir="./results"):
        os.makedirs(output_dir, exist_ok=True)
        results = []

        print("Avvio del Benchmark PGD...")
        for eps in epsilons:
            print(f"\nInizio test con epsilon = {eps}...")
            
            for idx, batch in enumerate(tqdm(dataset_loader)):
                # Supponendo di gestire tensori immagine [C, H, W] in range [0, 255]
                img_orig_tensor = batch[0].float().to(self.device) if isinstance(batch, (list, tuple)) else batch.float().to(self.device)
                
                clean_faces = self.count_faces(img_orig_tensor)
                if clean_faces == 0:
                    continue  # Salta immagini senza volti rilevati in partenza

                # Esecuzione attacco PGD
                img_adv, success, succ_iter = pgd_attack(
                    self.detector_wrapper, self.dsfd_net, self.mean_tensor, 
                    img_orig_tensor, epsilon=eps, alpha=alpha, iterations=iterations, device=self.device
                )

                adv_faces = self.count_faces(img_adv)
                evaded = (adv_faces == 0)

                # Metriche
                metrics = calculate_perturbation_metrics(img_orig_tensor, img_adv)

                results.append({
                    "epsilon": eps,
                    "evaded": evaded,
                    "detector_evasion": evaded,
                    "pipeline_mse": metrics["mse"],
                    "l2": metrics["l2"],
                    "linf": metrics["linf"],
                    "psnr": metrics["psnr"],
                    "iterations": succ_iter if succ_iter is not None else iterations
                })

        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "benchmark_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nBenchmark completato con successo. Risultati salvati in {csv_path}")
        return df
