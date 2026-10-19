import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eval_reidentification import BaseEvaluator
from attacks.blackbox.gradient_free_spatial_attention_attack import GradientFreeSpatialAttentionAttack
from data_loaders.lfw import LFWDataset

class GradientFreeSpatialAttentionBenchmark(BaseEvaluator):
    def __init__(self):
        super().__init__()

    def run_benchmark(self):
        parser = argparse.ArgumentParser(description="Benchmark per Gradient-Free Spatial Attention Attack")
        parser.add_argument("--epsilons", type=float, nargs="+", default=[8.0, 12.0, 16.0, 24.0], help="Lista di valori di epsilon da testare")
        parser.add_argument("--num_samples", type=int, default=10, help="Numero di campioni del dataset da valutare")
        parser.add_argument("--dataset_path", type=str, default="/kaggle/input/datasets/jessicali9530/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled", help="Percorso del dataset LFW")
        parser.add_argument("--output_dir", type=str, default="./results_spatial_attention_attack", help="Directory di output per i risultati e i plot")
        args = parser.parse_args()

        print(f"\n[INFO] Avvio Benchmark Gradient-Free Spatial Attention Attack con Epsilon: {args.epsilons}")
        
        target, anonymizer, detector_wrapper, dsfd_net, mean_tensor = self.setup_target("deeprivacy2")
        dataset = LFWDataset(root_dir=args.dataset_path)
        
        attack = GradientFreeSpatialAttentionAttack(target=target, detector_wrapper=detector_wrapper, device=self.device)
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        all_results = []
        
        for eps in args.epsilons:
            print(f"\n--- Test in corso con Epsilon = {eps} ---")
            for idx in range(min(args.num_samples, len(dataset))):
                sample = dataset[idx]
                img_orig = sample[0] if isinstance(sample, (list, tuple)) else sample
                
                if img_orig.ndim == 3 and img_orig.shape[2] == 3:
                    img_orig = img_orig.permute(2, 0, 1)
                    
                img_tensor = img_orig.clone().detach().to(self.device).float()
                if img_tensor.ndim == 3:
                    img_tensor = img_tensor.unsqueeze(0)

                # Generazione immagine avversaria
                img_adv, success, queries = attack.perturb(img_tensor[0], epsilon=eps, num_iter=50)

                # Passaggio attraverso la pipeline di anonimizzazione
                with torch.no_grad():
                    img_single = img_tensor[0] if img_tensor.ndim == 4 else img_tensor
                    if img_single.max() <= 1.0:
                        img_single = img_single * 255.0
                    img_single_uint8 = img_single.detach().byte().to(self.device)
                    
                    img_anonymized = target.anonymize(img_single_uint8) if hasattr(target, "anonymize") else anonymizer(img_single_uint8)
                    
                    img_adv_single = img_adv[0] if img_adv.ndim == 4 else img_adv
                    if img_adv_single.max() <= 1.0:
                        img_adv_single = img_adv_single * 255.0
                    img_adv_uint8 = img_adv_single.detach().byte().to(self.device)
                    
                    img_pipeline_adv = target.anonymize(img_adv_uint8) if hasattr(target, "anonymize") else anonymizer(img_adv_uint8)

                # Calcolo metriche visive (PSNR e MSE)
                orig_np = img_single.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
                adv_np = img_adv_single.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
                
                mse = np.mean((orig_np - adv_np) ** 2)
                psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0

                # Salvataggio immagine di confronto per il primo campione di ogni epsilon
                if idx == 0:
                    orig_img_cv = orig_np.astype(np.uint8)
                    anon_img_cv = np.array(img_anonymized)
                    adv_pipe_cv = np.array(img_pipeline_adv)
                    
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(orig_img_cv)
                    axes[0].set_title("Volto Originale")
                    axes[0].axis('off')
                    
                    axes[1].imshow(anon_img_cv)
                    axes[1].set_title("Target (Baseline Anonimizzata)")
                    axes[1].axis('off')
                    
                    axes[2].imshow(adv_pipe_cv)
                    axes[2].set_title(f"Target (Spatial Att., eps={eps})")
                    axes[2].axis('off')
                    
                    plt.suptitle(f"Gradient-Free Spatial Attention - Epsilon {eps} (Evaded: {success})")
                    plot_path = os.path.join(args.output_dir, f"spatial_attention_comparison_eps{eps}.png")
                    plt.savefig(plot_path, bbox_inches='tight')
                    plt.close()

                all_results.append({
                    "sample_id": idx,
                    "epsilon": eps,
                    "detector_evaded": success,
                    "queries": queries,
                    "psnr": round(psnr, 2),
                    "mse": round(mse, 2)
                })

        df = pd.DataFrame(all_results)
        csv_path = os.path.join(args.output_dir, "spatial_attention_high_epsilon_summary.csv")
        df.to_csv(csv_path, index=False)
        
        print("\n" + "="*80)
        print(" RISULTATI AGGREGATI - GRADIENT-FREE SPATIAL ATTENTION")
        print("="*80)
        summary = df.groupby("epsilon").agg(
            asr_percent=("detector_evaded", lambda x: f"{x.mean()*100:.1f}%"),
            avg_queries=("queries", "mean"),
            avg_psnr=("psnr", "mean"),
            avg_mse=("mse", "mean")
        ).reset_index()
        print(summary.to_string(index=False))
        print("="*80)
        print(f"[SUCCESSO] Report CSV e plot salvati in: {args.output_dir}")

if __name__ == "__main__":
    benchmark = GradientFreeSpatialAttentionBenchmark()
    benchmark.run_benchmark()
