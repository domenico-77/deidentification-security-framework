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
                image_name = self._get_image_name(idx)
                
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
        """Seleziona l'attacco riuscito, individua il volto e genera il plot con i ritagli comparativi."""
        if not os.path.exists(csv_path):
            print(f"[WARNING] File CSV non trovato: {csv_path}")
            return
            
        df = pd.read_csv(csv_path)
        successful = df[df["evaded"] == True] if "evaded" in df.columns else df.iloc[:0]
        
        if successful.empty:
            print("[WARNING] Nessun attacco riuscito trovato nel CSV per la visualizzazione.")
            return
        
        # Seleziona l'attacco riuscito con l'epsilon minore
        best_run = successful.sort_values(by="epsilon").iloc[0]
        eps = best_run['epsilon']
        image_name = best_run['image']
        
        try:
            idx = int(image_name.split("_")[1].split(".")[0])
        except Exception:
            idx = 0
            
        print(f"\nGenerazione visualizzazione ritagli di volto per: {image_name} | Epsilon={eps}")
        
        # 1. Recupera l'immagine originale dal dataset
        batch = self.dataset[idx]
        img_orig_tensor = batch[0].float().to(self.device) if isinstance(batch, (list, tuple)) else batch.float().to(self.device)
        
        # 2. Esegue l'attacco PGD per ottenere l'immagine adversarial
        img_adv, _, _ = self.attack.perturb(img_orig_tensor=img_orig_tensor, epsilon=eps)
        
        # 3. Rileva la bounding box del volto e processa le pipeline DeepPrivacy2
        with torch.no_grad():
            det_input = img_orig_tensor.detach().byte().float()  # Forma corretta (3, H, W) per il detector
            detections = self.detector_wrapper(det_input)
            
            if hasattr(self.target, "anonymize"):
                pipeline_out_orig = self.target.anonymize(img_orig_tensor.unsqueeze(0))
                pipeline_out_adv = self.target.anonymize(img_adv.unsqueeze(0))
            else:
                pipeline_out_orig = self.target(img_orig_tensor.unsqueeze(0))
                pipeline_out_adv = self.target(img_adv.unsqueeze(0))

        def tensor_to_numpy(t):
            if t.dim() == 4:
                t = t.squeeze(0)
            t = t.detach().cpu().permute(1, 2, 0).numpy()
            t = np.clip(t, 0, 255).astype(np.uint8)
            return t

        orig_np = tensor_to_numpy(img_orig_tensor)
        orig_pipeline_np = tensor_to_numpy(pipeline_out_orig)
        adv_pipeline_np = tensor_to_numpy(pipeline_out_adv)

        # 4. Estrae le coordinate del volto tramite il detector
        if len(detections) > 0 and detections[0] is not None and len(detections[0]) > 0:
            box = detections[0][0][:4].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            h, w = orig_np.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Ritaglio mirato sul volto
            face_orig = orig_np[y1:y2, x1:x2]
            face_orig_pipe = orig_pipeline_np[y1:y2, x1:x2]
            face_adv_pipe = adv_pipeline_np[y1:y2, x1:x2]
        else:
            # Fallimento di sicurezza nel caso il detector non restituisca coordinate valide
            face_orig, face_orig_pipe, face_adv_pipe = orig_np, orig_pipeline_np, adv_pipeline_np

        # 5. Generazione del plot a 3 pannelli focalizzato sui volti
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(face_orig)
        axes[0].set_title(f"Volto Originale\n(Eps: {eps})")
        axes[0].axis("off")
        
        axes[1].imshow(face_orig_pipe)
        axes[1].set_title("DeepPrivacy2 (Baseline)\n[Volto Anon. / Sostituito]")
        axes[1].axis("off")
        
        axes[2].imshow(face_adv_pipe)
        axes[2].set_title("DeepPrivacy2 (Adversarial)\n[Evasione / Inalterato]")
        axes[2].axis("off")
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Grafico dei volti salvato in: {save_path}")
            
        plt.show()
        plt.close()
