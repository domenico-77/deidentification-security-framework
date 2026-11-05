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


# Soglia biometrica per leakage (allineata al resto del codice)
LEAKAGE_THRESHOLD = 0.55


class BlackBoxEvaluator(BaseEvaluator):
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

    def get_attack_instance(self, attack_name, detector_wrapper, target, anonymizer, epsilon,
                            loss_mode="evasion", region_mode="face", n_queries=None,
                            patch_size=4):
        attack_name = attack_name.lower()
        if n_queries is None:
            n_queries = 100 if loss_mode == "leakage" else 500

        if attack_name == "square":
            return SquareAttack(
                detector_wrapper=detector_wrapper, device=self.device, n_queries=n_queries,
                loss_mode=loss_mode, region_mode=region_mode,
                anonymizer_fn=anonymizer,
                extract_emb_fn=self.extract_embedding,
                compute_sim_fn=self.compute_cosine_similarity,
            )
        elif attack_name == "simba":
            return SimBA(
                detector_wrapper=detector_wrapper, device=self.device, n_queries=n_queries,
                step_size=2.0, patch_size=patch_size,
                loss_mode=loss_mode, region_mode=region_mode,
                anonymizer_fn=anonymizer,
                extract_emb_fn=self.extract_embedding,
                compute_sim_fn=self.compute_cosine_similarity,
            )
        elif attack_name == "boundary":
            return BoundaryAttack(detector_wrapper=detector_wrapper, device=self.device, max_iter=200)
        elif attack_name == "advpatch":
            return AdversarialPatchAttack(detector_wrapper=detector_wrapper, device=self.device)
        elif attack_name == "spsa":
            n_iters = max(50, (n_queries or 200) // 2)
            return SPSAAttack(
                detector_wrapper=detector_wrapper, anonymizer_fn=anonymizer,
                extract_emb_fn=self.extract_embedding,
                compute_sim_fn=self.compute_cosine_similarity,
                device=self.device, n_iters=n_iters, lr=2.0, delta=4.0,
                region_mode=region_mode, loss_mode=loss_mode,
            )
        elif attack_name == "landmark_spoofing":
            return LandmarkSpoofingAttack(
                target=target, detector_wrapper=detector_wrapper, device=self.device
            )
        elif attack_name == "gfsa":
            return GradientFreeSpatialAttentionAttack(
                target=target, detector_wrapper=detector_wrapper, device=self.device
            )
        else:
            raise ValueError(f"Attacco black-box non supportato: {attack_name}")

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

    def _check_stochasticity(self, anonymizer, img_uint8, n_trials=3):
        sims = []
        try:
            emb_orig = self.extract_embedding(img_uint8.float().cpu())
            if emb_orig is None:
                return None, None
            for _ in range(n_trials):
                out = anonymizer(img_uint8)
                out_t = self._ensure_chw_tensor(out)
                emb = self.extract_embedding(out_t)
                if emb is not None:
                    sims.append(self.compute_cosine_similarity(emb_orig, emb))
            if not sims:
                return None, None
            return float(np.mean(sims)), float(np.std(sims))
        except Exception:
            return None, None

    def run_blackbox(self):
        parser = argparse.ArgumentParser(description="Valutazione Identity Leakage & Re-identification (Black-Box Mode)")
        parser.add_argument("--attack", type=str, default="square",
                            choices=["square", "simba", "boundary", "advpatch", "spsa",
                                     "landmark_spoofing", "gfsa"])
        parser.add_argument("--model", type=str, default="deeprivacy2",
                            choices=["deeprivacy2", "ciagan"])
        parser.add_argument("--epsilon", type=float, default=32.0)
        parser.add_argument("--num_samples", type=int, default=20)
        parser.add_argument("--loss_mode", type=str, default="evasion",
                            choices=["evasion", "leakage"])
        parser.add_argument("--n_queries", type=int, default=None)
        parser.add_argument("--patch_size", type=int, default=4)
        args = parser.parse_args()

        attacks_without_patch = {"square", "boundary", "advpatch", "spsa",
                                 "landmark_spoofing", "gfsa"}
        if args.attack.lower() in attacks_without_patch and args.patch_size != 4:
            print(f"[INFO] --patch_size={args.patch_size} ignorato: "
                  f"l'attacco '{args.attack}' non usa patch.")

        print(f"\n[INFO] Avvio Identity Leakage Analysis & Re-identification (BLACK-BOX MODE)")
        print(f" -> Attacco: {args.attack.upper()} (Epsilon: {args.epsilon})")
        print(f" -> Target Anonimizzazione: {args.model.upper()}")
        print(f" -> Loss mode: {args.loss_mode.upper()}")
        print(f" -> Dispositivo: {self.device}")

        target, anonymizer, detector_wrapper, dsfd_net, mean_tensor = self.setup_target(args.model)

        region_mode = "ring" if args.model == "ciagan" else "face"
        print(f" -> Region mode: {region_mode.upper()}")

        std_sim = None
        if args.loss_mode == "leakage":
            ds_probe = LFWDataset(root_dir="/kaggle/input/datasets/jessicali9530/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled")
            probe_img = ds_probe[0][0]
            if probe_img.max() <= 1.0:
                probe_img = probe_img * 255.0
            probe_uint8 = probe_img.byte().to(self.device)
            print("\n[PRE-CHECK] Verifica stocasticità della pipeline (3 chiamate)...")
            mean_sim, std_sim = self._check_stochasticity(anonymizer, probe_uint8, n_trials=3)
            if mean_sim is not None:
                print(f"  sim media: {mean_sim:.4f}, std: {std_sim:.4f}")

        attack = self.get_attack_instance(
            args.attack, detector_wrapper=detector_wrapper, target=target,
            anonymizer=anonymizer, epsilon=args.epsilon,
            loss_mode=args.loss_mode, region_mode=region_mode,
            n_queries=args.n_queries, patch_size=args.patch_size,
        )
        print(f" -> n_queries: {getattr(attack, 'n_queries', 'N/A')}")
        if hasattr(attack, "n_iters"):
            print(f" -> n_iters: {attack.n_iters}")
        if hasattr(attack, "patch_size"):
            print(f" -> patch_size: {attack.patch_size}")

        dataset = LFWDataset(root_dir="/kaggle/input/datasets/jessicali9530/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled")

        # STRUTTURA: ./results_blackbox/{attack}/{model}/
        base_results_dir = "./results_blackbox"
        output_dir = os.path.join(base_results_dir, args.attack.lower(), args.model.lower())
        success_img_dir = os.path.join(output_dir, "successful_evasions")
        os.makedirs(success_img_dir, exist_ok=True)

        results = []
        print(f"[INFO] Elaborazione di {min(args.num_samples, len(dataset))} campioni...")

        for idx in tqdm(range(min(args.num_samples, len(dataset)))):
            sample = dataset[idx]
            img_orig = sample[0] if isinstance(sample, (list, tuple)) else sample
            if img_orig.ndim == 3 and img_orig.shape[2] == 3:
                img_orig = img_orig.permute(2, 0, 1)

            img_tensor = img_orig.clone().detach().to(self.device).float()
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)

            try:
                img_adv, success, queries = attack.perturb(img_tensor[0], epsilon=args.epsilon)
            except TypeError:
                img_adv, success, queries = attack.perturb(img_tensor[0])

            delta_linf = (img_adv.float() - img_tensor[0].float()).abs().max().item()

            with torch.no_grad():
                img_single = img_tensor[0] if img_tensor.ndim == 4 else img_tensor
                if img_single.max() <= 1.0:
                    img_single = img_single * 255.0
                img_single_uint8 = img_single.detach().byte().to(self.device)

                raw_anon = anonymizer(img_single_uint8)
                img_anonymized = raw_anon

                img_adv_single = img_adv[0] if img_adv.ndim == 4 else img_adv
                if img_adv_single.max() <= 1.0:
                    img_adv_single = img_adv_single * 255.0
                img_adv_uint8 = img_adv_single.detach().byte().to(self.device)

                raw_adv = anonymizer(img_adv_uint8)
                img_pipeline_adv = raw_adv

            # Uso interno: pixel_diff per calcolare pipeline_bypassed (non stampato)
            with torch.no_grad():
                img_adv_f = img_adv_uint8.detach().cpu().float()
                img_pipe_adv_chw = self._ensure_chw_tensor(img_pipeline_adv).float()
                if img_pipe_adv_chw.max() <= 1.0:
                    img_pipe_adv_chw = img_pipe_adv_chw * 255.0

                _pixel_diff = (img_adv_f - img_pipe_adv_chw).abs().mean().item()
                pipeline_bypassed = bool(success and _pixel_diff < 1.0)

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

            # ============================================================
            # CLASSIFICAZIONE OUTCOME (solo per leakage)
            #   - BYPASS: pipeline_bypassed=True (volto originale intatto)
            #   - LEAKAGE: pipeline anonimizza MA sim_adv >= soglia
            #   - FAIL: altrimenti
            # ============================================================
            if args.loss_mode == "leakage":
                leakage_detected = bool((not pipeline_bypassed) and (sim_orig_adv >= LEAKAGE_THRESHOLD))
            else:
                leakage_detected = False  # non applicabile

            # Salvataggio PNG silenzioso: includiamo anche leakage_detected
            save_plot = pipeline_bypassed or leakage_detected or (success and sim_orig_adv >= 0.5)
            if save_plot:
                try:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    orig_np = img_single.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                    anon_np = img_anon_chw.permute(1, 2, 0).numpy().astype(np.uint8)
                    adv_np = img_adv_chw.permute(1, 2, 0).numpy().astype(np.uint8)

                    axes[0].imshow(orig_np); axes[0].set_title(f"Originale (ID: {idx})"); axes[0].axis('off')
                    axes[1].imshow(anon_np); axes[1].set_title(f"Anonimizzata (sim={sim_orig_anon:.4f})"); axes[1].axis('off')
                    axes[2].imshow(adv_np)
                    axes[2].set_title(f"De-anonimizzata (sim={sim_orig_adv:.4f})"); axes[2].axis('off')

                    plt.tight_layout()
                    tag = "bypass" if pipeline_bypassed else ("leak" if leakage_detected else "highsim")
                    fname = os.path.join(success_img_dir,
                                         f"sample_{idx}_{tag}_sim{sim_orig_adv:.2f}.png")
                    plt.savefig(fname, dpi=150)
                    plt.close(fig)
                except Exception:
                    pass

            # Costruzione riga condizionale
            row = {
                "sample_id": idx,
                "attack": f"bb_{args.attack}",
                "model": args.model,
                "loss_mode": args.loss_mode,
                "region_mode": region_mode,
                "epsilon": args.epsilon,
                "delta_linf": round(delta_linf, 4),
                "queries_used": queries,
                "sim_orig_vs_anonymized": round(sim_orig_anon, 4),
                "sim_orig_vs_adversarial_pipe": round(sim_orig_adv, 4),
            }
            if args.loss_mode == "evasion":
                row["detector_evaded"] = success
            else:  # leakage
                row["pipeline_bypassed"] = pipeline_bypassed
                row["leakage_detected"] = leakage_detected

            results.append(row)

        df = pd.DataFrame(results)
        output_csv = os.path.join(output_dir, f"{args.loss_mode}_eps{args.epsilon}.csv")
        df.to_csv(output_csv, index=False)
        print(f"\n[SUCCESSO] Report in: {output_csv}")

        print("\n" + "=" * 100)
        print(f" TABELLA RIASSUNTIVA ({args.model.upper()}, {args.loss_mode.upper()})")
        print("=" * 100)
        print(df.to_string(index=False))

        print("\n" + "=" * 100)
        print(" STATISTICHE")
        print("=" * 100)
        base_cols = ["delta_linf", "queries_used",
                     "sim_orig_vs_anonymized", "sim_orig_vs_adversarial_pipe"]
        stats_cols = [c for c in base_cols if c in df.columns]
        print(df[stats_cols].describe().to_string())
        print("=" * 100)

        print(f"Risultati salvati in: {output_dir}")


if __name__ == "__main__":
    evaluator = BlackBoxEvaluator()
    evaluator.run_blackbox()
