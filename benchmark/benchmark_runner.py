import os
import time
import pandas as pd
import torch
from metrics.perturbation import calculate_perturbation_metrics
from metrics.image_metrics import compute_pipeline_mse


class BenchmarkRunner:
    def __init__(self, target, attack, dataset, output_dir="./results"):
        self.target = target
        self.attack = attack
        self.dataset = dataset
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_benchmark(self, epsilons=[2.0, 4.0, 8.0, 16.0, 24.0, 32.0], num_samples=100):
        checkpoint_path = os.path.join(self.output_dir, "benchmark_checkpoint.csv")
        results = []

        for idx in range(min(num_samples, len(self.dataset))):
            img_tensor, path = self.dataset[idx]

            # Baseline Check
            orig_detections = self.attack.detector.count_detections(img_tensor)
            if orig_detections != 1:
                continue  # Considera solo immagini single-face valide

            for eps in epsilons:
                start_time = time.time()
                adv_tensor = self.attack.attack(img_tensor, epsilon=eps)
                elapsed_time = time.time() - start_time

                adv_detections = self.attack.detector.count_detections(adv_tensor)
                evaded = (adv_detections == 0)

                # De-identification pipeline test
                anon_out = self.target.process_image(adv_tensor)
                p_mse = compute_pipeline_mse(adv_tensor.cpu().numpy().transpose(1, 2, 0), anon_out)

                p_metrics = calculate_perturbation_metrics(img_tensor, adv_tensor)

                res = {
                    "image": os.path.basename(path),
                    "epsilon": eps,
                    "evaded": evaded,
                    "pipeline_mse": p_mse,
                    "execution_time": elapsed_time,
                    **p_metrics
                }
                results.append(res)

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.output_dir, "full_results.csv"), index=False)
        return df