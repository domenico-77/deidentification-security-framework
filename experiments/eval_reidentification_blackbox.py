%%writefile experiments/eval_reidentification_blackbox.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from eval_reidentification import BaseEvaluator
from attacks.blackbox.square_attack import SquareAttack
from attacks.blackbox.simba_attack import SimBA
from attacks.blackbox.boundary_attack import BoundaryAttack
from attacks.blackbox.adversarialpatch_attack import AdversarialPatchAttack
from attacks.blackbox.spsa_attack import SPSAAttack
from attacks.blackbox.landmark_spoofing_attack import LandmarkSpoofingAttack
from attacks.blackbox.gradient_free_spatial_attention_attack import GradientFreeSpatialAttentionAttack

from data_loaders.lfw import LFWDataset
from targets.ciagan import CIAGANTarget

class BlackBoxEvaluator(BaseEvaluator):
    """
    Evaluator specializzato per Black-Box Attacks (Score-based & Decision-based).
    Eredita l'intero motore di valutazione e gestisce in modo uniforme 
    sia DeepPrivacy2 che CIAGAN tramite il metodo nativo setup_target.
    """
    def __init__(self):
        super().__init__()

    def setup_target(self, model_name):
        model_name = model_name.lower()
        if model_name == "deeprivacy2":
            return super().setup_target("deeprivacy2")
        elif model_name == "ciagan":
            print("[INFO] Caricamento del target CIAGAN...")
            models_dir = "/kaggle/input/datasets/domenicovicenti/ciagan-models"
            target = CIAGANTarget(models_dir=models_dir, device=self.device)
            detector_wrapper = getattr(target, 'detector_wrapper', None)
            
            def ciagan_anonymizer(img_tensor_uint8):
                if img_tensor_uint8.ndim == 3 and img_tensor_uint8.shape[0] == 3:
                    img_input = img_tensor_uint8.unsqueeze(0).float()
                else:
                    img_input = img_tensor_uint8.float()
                if img_input.max() > 1.0:
                    img_input = img_input / 255.0
                with torch.no_grad():
                    out = target.anonymizer(img_input)
                if isinstance(out, torch.Tensor):
                    out = out.squeeze(0)
                    if out.max() <= 1.0:
                        out = out * 255.0
                    return out.byte()
                return out

            return target, ciagan_anonymizer, detector_wrapper, None, None
        else:
            return super().setup_target(model_name)

    def get_attack_instance(self, attack_name, detector_wrapper, target, anonymizer, epsilon):
        attack_name = attack_name.lower()
        if attack_name == "square":
            return SquareAttack(detector_wrapper=detector_wrapper, device=self.device, n_queries=1000)
        elif attack_name == "simba":
            return SimBA(detector_wrapper=detector_wrapper, device=self.device, n_queries=1000)
        elif attack_name == "boundary":
            return BoundaryAttack(detector_wrapper=detector_wrapper, device=self.device, max_iter=200)
        elif attack_name == "advpatch":
            return AdversarialPatchAttack(detector_wrapper=detector_wrapper, device=self.device)
        elif attack_name == "spsa":
            anonymizer_fn = target.anonymize if hasattr(target, "anonymize") else (target.anonymizer if hasattr(target, "anonymizer") else anonymizer)
            return SPSAAttack(
                detector_wrapper=detector_wrapper,
                anonymizer_fn=anonymizer_fn,
                extract_emb_fn=self.extract_embedding,
                compute_sim_fn=self.compute_cosine_similarity,
                device=self.device,
                n_iters=50,
                lr=2.0,
                delta=1e-2
            )
        elif attack_name == "landmark_spoofing":
            return LandmarkSpoofingAttack(
                target=target,
                detector_wrapper=detector_wrapper,
                device=self.device
            )
        elif attack_name == "gfsa":
            return GradientFreeSpatialAttentionAttack(
                target=target,
                detector_wrapper=detector_wrapper,
                device=self.device
            )
        else:
            raise ValueError(f"Attacco black-box non supportato o non valido: {attack_name}")

    @staticmethod
    def _ensure_chw_tensor(img_obj):
        if isinstance(img_obj, dict):
            for k in ['image', 'img', 'fake_image', 'output', 'anonymized', 'result', 'im']:
                if k in img_obj:
                    img_obj = img_obj[k]
                    break
            else:
                for v in img_obj.values():
                    if isinstance(v, (torch.Tensor, np.ndarray)):
                        img_obj = v
                        break

        if isinstance(img_obj, torch.Tensor):
            t = img_obj.detach().cpu().float()
            if t.dim() == 4:
                t = t.squeeze(0)
            if t.shape[0] != 3 and t.shape[-1] == 3:
                t = t.permute(2, 0, 1)
            return t
        
        if hasattr(img_obj, "convert"):
            img_obj = np.array(img_obj.convert("RGB"))

        if isinstance(img_obj, (list, tuple)):
            img_obj = np.array(img_obj)

        if isinstance(img_obj, np.ndarray):
            if img_obj.dtype == object:
                try:
                    img_obj = np.stack([np.array(x, dtype=np.uint8) for x in img_obj])
                except Exception:
                    img_obj = np.array(img_obj.tolist(), dtype=np.uint8)
            arr = img_obj
        else:
            try:
                arr = np.array(img_obj, dtype=np.uint8)
            except Exception:
                arr = np.array(list(img_obj), dtype=np.uint8)

        if arr.ndim == 4:
            arr = arr[0]

        t = torch.from_numpy(arr.astype(np.float32))
        if t.dim() == 3 and t.shape[0] != 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        elif t.dim() == 2:
            t = t.unsqueeze(0).repeat(3, 1, 1)
        return t
            
    def run_blackbox(self):
        parser = argparse.ArgumentParser(description="Valutazione Identity Leakage & Re-identification (Black-Box Mode)")
        parser.add_argument("--attack", type=str, default="square", choices=["square", "simba", "boundary", "advpatch", "spsa", "landmark_spoofing", "gfsa"])
        parser.add_argument("--model", type=str, default="deeprivacy2", choices=["deeprivacy2", "ciagan"])
        parser.add_argument("--epsilon", type=float, default=32.0)
        parser.add_argument("--num_samples", type=int, default=20)
        
        args = parser.parse_args()

        print(f"\n[INFO] Avvio Identity Leakage Analysis & Re-identification (BLACK-BOX MODE)")
        print(f" -> Attacco: {args.attack.upper()} (Epsilon: {args.epsilon})")
        print(f" -> Target Anonimizzazione: {args.model.upper()}")
        print(f" -> Dispositivo: {self.device}")

        target, anonymizer, detector_wrapper, dsfd_net, mean_tensor = self.setup_target(args.model)

        attack = self.get_attack_instance(
            args.attack, 
            detector_wrapper=detector_wrapper, 
            target=target, 
            anonymizer=anonymizer, 
            epsilon=args.epsilon
        )

        dataset = LFWDataset(root_dir="/kaggle/input/datasets/jessicali9530/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled")
        
        output_dir = f"./results_identity_{args.model}_blackbox"
        success_img_dir = os.path.join(output_dir, "successful_evasions")
        os.makedirs(success_img_dir, exist_ok=True)

        results = []
        print(f"[INFO] Elaborazione di {min(args.num_samples, len(dataset))} campioni con analisi di similarità (Black-Box)...")

        for idx in tqdm(range(min(args.num_samples, len(dataset)))):
            sample = dataset[idx]
            img_orig = sample[0] if isinstance(sample, (list, tuple)) else sample
            
            if img_orig.ndim == 3 and img_orig.shape[2] == 3:
                img_orig = img_orig.permute(2, 0, 1)
                
            img_tensor = img_orig.clone().detach().to(self.device).float()
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)

            if hasattr(attack, "perturb"):
                try:
                    img_adv, success, queries = attack.perturb(img_tensor[0], epsilon=args.epsilon)
                except TypeError:
                    img_adv, success, queries = attack.perturb(img_tensor[0])
            else:
                raise AttributeError("L'oggetto attacco non possiede un metodo .perturb() valido.")

            with torch.no_grad():
                img_single = img_tensor[0] if img_tensor.ndim == 4 else img_tensor
                if img_single.max() <= 1.0:
                    img_single = img_single * 255.0
                img_single_uint8 = img_single.detach().byte().to(self.device)
                
                if hasattr(target, "anonymize"):
                    img_anonymized = target.anonymize(img_single_uint8)
                elif hasattr(target, "anonymizer"):
                    img_anonymized = target.anonymizer(img_single_uint8)
                elif hasattr(anonymizer, "__call__"):
                    img_anonymized = anonymizer(img_single_uint8)
                else:
                    raise AttributeError("Il target non possiede un metodo di anonimizzazione valido.")
                
                img_adv_single = img_adv[0] if img_adv.ndim == 4 else img_adv
                if img_adv_single.max() <= 1.0:
                    img_adv_single = img_adv_single * 255.0
                img_adv_uint8 = img_adv_single.detach().byte().to(self.device)
                
                if hasattr(target, "anonymize"):
                    img_pipeline_adv = target.anonymize(img_adv_uint8)
                elif hasattr(target, "anonymizer"):
                    img_pipeline_adv = target.anonymizer(img_adv_uint8)
                elif hasattr(anonymizer, "__call__"):
                    img_pipeline_adv = anonymizer(img_adv_uint8)
                else:
                    raise AttributeError("Il target non possiede un metodo di anonimizzazione valido.")

            sim_orig_anon = 0.0
            sim_orig_adv = 0.0

            if self.recognizer is not None:
                img_anon_chw = self._ensure_chw_tensor(img_anonymized)
                img_adv_chw = self._ensure_chw_tensor(img_pipeline_adv)

                emb_orig = self.extract_embedding(img_single.detach().cpu())
                emb_anon = self.extract_embedding(img_anon_chw)
                emb_adv_pipe = self.extract_embedding(img_adv_chw)

                if emb_orig is not None and emb_anon is not None:
                    sim_orig_anon = self.compute_cosine_similarity(emb_orig, emb_anon)
                if emb_orig is not None and emb_adv_pipe is not None:
                    sim_orig_adv = self.compute_cosine_similarity(emb_orig, emb_adv_pipe)

            # Salvataggio automatico del plot se l'attacco ha eluso il detector e ha alta similarità
            if success and sim_orig_adv >= 0.5:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                orig_np = img_single.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                anon_np = img_anon_chw.permute(1, 2, 0).numpy().astype(np.uint8)
                adv_np = img_adv_chw.permute(1, 2, 0).numpy().astype(np.uint8)

                axes[0].imshow(orig_np)
                axes[0].set_title(f"Volto Originale\n(Sample ID: {idx})")
                axes[0].axis('off')

                axes[1].imshow(anon_np)
                axes[1].set_title(f"Target (Baseline)\nSim: {sim_orig_anon:.4f}")
                axes[1].axis('off')

                axes[2].imshow(adv_np)
                axes[2].set_title(f"Target (Adversarial Bypass)\nSim: {sim_orig_adv:.4f}")
                axes[2].axis('off')

                plt.tight_layout()
                plot_filename = os.path.join(success_img_dir, f"successful_evasion_sample_{idx}_sim_{sim_orig_adv:.2f}.png")
                plt.savefig(plot_filename, dpi=150)
                plt.close(fig)

            results.append({
                "sample_id": idx,
                "attack": f"bb_{args.attack}",
                "epsilon": args.epsilon,
                "detector_evaded": success,
                "queries_used": queries,
                "pipeline_behavior": "Bypassed (Original face retained)" if success else "Anonymized (Face replaced)",
                "sim_orig_vs_anonymized": round(sim_orig_anon, 4),
                "sim_orig_vs_adversarial_pipe": round(sim_orig_adv, 4)
            })

        df = pd.DataFrame(results)
        output_csv = os.path.join(output_dir, f"identity_leakage_similarity_{args.attack}_eps{args.epsilon}.csv")
        df.to_csv(output_csv, index=False)
        print(f"\n[SUCCESSO] Analisi di similarità black-box completata. Report salvato in: {output_csv}")
        print(f"[INFO] Immagini delle evasioni riuscite salvate in: {success_img_dir}")

        print("\n" + "="*85)
        print(f" TABELLA RIASSUNTIVA BLACK-BOX ({args.model.upper()})")
        print("="*85)
        print(df.to_string(index=False))
        
        print("\n" + "="*85)
        print(" STATISTICHE DESCRITTIVE DELLE SIMILARITÀ BLACK-BOX & QUERY EFFICIENCY")
        print("="*85)
        print(df[["queries_used", "sim_orig_vs_anonymized", "sim_orig_vs_adversarial_pipe"]].describe().to_string())
        print("="*85)

if __name__ == "__main__":
    evaluator = BlackBoxEvaluator()
    evaluator.run_blackbox()
