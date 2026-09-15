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

    def _get_image_name(self, idx):
        """Estrae il nome reale del file dal dataset LFW se disponibile, altrimenti usa un fallback."""
        if hasattr(self.dataset, "image_paths") and idx < len(self.dataset.image_paths):
            return os.path.basename(self.dataset.image_paths[idx])
        elif hasattr(self.dataset, "samples") and idx < len(self.dataset.samples):
            return os.path.basename(self.dataset.samples[idx][0])
        return f"sample_{idx}.jpg"

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
        """Seleziona l'attacco riuscito, individua il volto e genera il plot con le immagini comparative."""
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
        
        # Trova l'indice corrispondente nel dataset
        idx = 0
        found = False
        if hasattr(self.dataset, "image_paths"):
            for i, p in enumerate(self.dataset.image_paths):
                if os.path.basename(p) == image_name:
                    idx = i
                    found = True
                    break
        elif hasattr(self.dataset, "samples"):
            for i, (p, _) in enumerate(self.dataset.samples):
                if os.path.basename(p) == image_name:
                    idx = i
                    found = True
                    break
                    
        if not found:
            idx = 0
            
        print(f"\nGenerazione visualizzazione ritagli di volto per: {image_name} | Epsilon={eps} (Indice dataset: {idx})")
        
        # 1. Recupera l'immagine originale dal dataset
        batch = self.dataset[idx]
        img_orig_tensor = batch[0].float().to(self.device) if isinstance(batch, (list, tuple)) else batch.float().to(self.device)
        
        # 2. Esegue l'attacco PGD per ottenere l'immagine adversarial
        img_adv, _, _ = self.attack.perturb(img_orig_tensor=img_orig_tensor, epsilon=eps)
        
        # 3. Chiamata corretta alla pipeline di anonimizzazione di DeepPrivacy2 (formato 3D: C, H, W)
        # 3. Chiamata alla pipeline di anonimizzazione speculare al tuo vecchio script
        # 3. Chiamata alla pipeline identica al vecchio script funzionante
        with torch.no_grad():
            # Assicuriamoci che i tensori abbiano il formato e il tipo attesi dalla pipeline (senza forzare .byte() grezzo se sballa i colori)
            def prepare_for_anonymizer(t):
                if t.is_cuda:
                    t = t.detach().cpu()
                # Se il tensore è in float [0, 255] o [0, 1], convertiamolo correttamente come nel vecchio test
                t = t.float()
                if t.max() <= 1.0:
                    t = t * 255.0
                return t.byte().to(self.device)

            tensor_orig_clean = prepare_for_anonymizer(img_orig_tensor)
            tensor_adv_clean = prepare_for_anonymizer(img_adv)
            
            anonymizer_obj = getattr(self.target, "anonymizer", getattr(self.target, "pipeline", self.target))
            
            output_orig = anonymizer_obj(
                tensor_orig_clean,
                truncation_value=1.0,
                multi_modal_truncation=False,
                amp=False
            )
            output_adv = anonymizer_obj(
                tensor_adv_clean,
                truncation_value=1.0,
                multi_modal_truncation=False,
                amp=False
            )
            
            # Estrazione sicura dell'immagine dall'output
            def extract_img_tensor(out):
                if isinstance(out, dict):
                    res = out.get("im", out.get("img", list(out.values())[0]))
                else:
                    res = out
                if isinstance(res, torch.Tensor):
                    res = res.detach().cpu()
                return res

            pipeline_out_orig = extract_img_tensor(output_orig)
            pipeline_out_adv = extract_img_tensor(output_adv)
            
            # Detection per il ritaglio del volto
            detections = self.detector_wrapper(tensor_orig_clean.float())

        def tensor_to_numpy(t):
            if t.dim() == 4:
                t = t.squeeze(0)
            t = t.detach().cpu().permute(1, 2, 0).numpy()
            t = np.clip(t, 0, 255).astype(np.uint8)
            return t

        orig_np = tensor_to_numpy(img_orig_tensor)
        orig_pipeline_np = tensor_to_numpy(pipeline_out_orig)
        adv_pipeline_np = tensor_to_numpy(pipeline_out_adv)

        # 4. Estrazione delle coordinate del volto
        # 4. Estrazione delle coordinate del volto (Gestione robusta del detector)
        box = None
        try:
            # DeepPrivacy2 / DSFD detector può restituire un tensore, una lista o un oggetto strutturato
            if detections is not None:
                det_results = detections[0] if isinstance(detections, (list, tuple)) else detections
                
                # Se è un tensore o un array numpy
                if isinstance(det_results, torch.Tensor):
                    det_results = det_results.detach().cpu().numpy()
                elif hasattr(det_results, "cpu"):
                    det_results = det_results.cpu().numpy()
                    
                if isinstance(det_results, np.ndarray) and det_results.size > 0:
                    box = det_results[0][:4].astype(int)
                elif isinstance(det_results, (list, tuple)) and len(det_results) > 0:
                    face_item = det_results[0]
                    if hasattr(face_item, "cpu"):
                        box = face_item[:4].cpu().numpy().astype(int)
                    elif isinstance(face_item, (np.ndarray, list, tuple)):
                        box = np.array(face_item[:4]).astype(int)
        except Exception as e:
            print(f"[DEBUG] Errore estrazione box corretto: {e}")

        face_orig, face_orig_pipe, face_adv_pipe = orig_np, orig_pipeline_np, adv_pipeline_np

        # 5. Generazione del plot a 3 pannelli
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(face_orig)
        axes[0].set_title(f"Volto Originale\n({image_name})")
        axes[0].axis("off")
        
        axes[1].imshow(face_orig_pipe)
        axes[1].set_title("DeepPrivacy2 (Baseline)\n[Volto Sintetico StyleGAN]")
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
