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
                            loss_mode="evasion", region_mode="face", n_queries=None):
        attack_name = attack_name.lower()

        # n_queries adattivo: DP2 è ~0.5s/query, CIAGAN è ~0.05s/query
        if n_queries is None:
            n_queries = 100 if loss_mode == "leakage" else 500

        if attack_name == "square":
            return SquareAttack(
                detector_wrapper=detector_wrapper,
                device=self.device,
                n_queries=n_queries,
                loss_mode=loss_mode,
                region_mode=region_mode,
                anonymizer_fn=anonymizer,
                extract_emb_fn=self.extract_embedding,
                compute_sim_fn=self.compute_cosine_similarity,
            )
        elif attack_name == "simba":
            return SimBA(detector_wrapper=detector_wrapper, device=self.device, n_queries=n_queries)
        elif attack_name == "boundary":
            return BoundaryAttack(detector_wrapper=detector_wrapper, device=self.device, max_iter=200)
        elif attack_name == "advpatch":
            return AdversarialPatchAttack(detector_wrapper=detector_wrapper, device=self.device)
        elif attack_name == "spsa":
            return SPSAAttack(
                detector_wrapper=detector_wrapper,
                anonymizer_fn=anonymizer,
                extract_emb_fn=self.extract_embedding,
                compute_sim_fn=self.compute_cosine_similarity,
                device=self.device,
                n_iters=50,
                lr=1.0,
                delta=2.0,
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
        """Verifica se la pipeline è stocastica (utile prima di attacchi leakage)."""
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
        except Exception as e:
            print(f"[WARN] stochasticity check failed: {e}")
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
        parser.add_argument("--n_queries", type=int, default=None,
                            help="Numero di query (default: 100 leakage / 500 evasion)")
        args = parser.parse_args()

        print(f"\n[INFO] Avvio Identity Leakage Analysis & Re-identification (BLACK-BOX MODE)")
        print(f" -> Attacco: {args.attack.upper()} (Epsilon: {args.epsilon})")
        print(f" -> Target Anonimizzazione: {args.model.upper()}")
        print(f" -> Loss mode: {args.loss_mode.upper()}")
        print(f" -> Dispositivo: {self.device}")

        target, anonymizer, detector_wrapper, dsfd_net, mean_tensor = self.setup_target(args.model)

        region_mode = "ring" if args.model == "ciagan" else "face"
        print(f" -> Region mode: {region_mode.upper()}")

        # PRE-CHECK: stocasticità della pipeline (solo per leakage)
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
                if std_sim > 0.05:
                    print(f"  [WARN] Pipeline STOCASTICA (std={std_sim:.4f}). "
                          f"Il rumore della GAN domina il segnale biometrico.")
                    print(f"         Un attacco leakage potrebbe essere inefficace "
                          f"indipendentemente dalla logica.")
                else:
                    print(f"  [OK] Pipeline deterministica/stable (std={std_sim:.4f}).")
            else:
                print("  [WARN] Impossibile misurare la stocasticità.")

        attack = self.get_attack_instance(
            args.attack,
            detector_wrapper=detector_wrapper,
            target=target,
            anonymizer=anonymizer,
            epsilon=args.epsilon,
            loss_mode=args.loss_mode,
            region_mode=region_mode,
            n_queries=args.n_queries,
        )
        print(f" -> n_queries: {getattr(attack, 'n_queries', 'N/A')}")

        dataset = LFWDataset(root_dir="/kaggle/input/datasets/jessicali9530/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled")

        output_dir = f"./results_identity_{args.model}_blackbox"
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

            if success and sim_orig_adv >= 0.5:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                orig_np = img_single.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                anon_np = img_anon_chw.permute(1, 2, 0).numpy().astype(np.uint8)
                adv_np = img_adv_chw.permute(1, 2, 0).numpy().astype(np.uint8)

                axes[0].imshow(orig_np)
                axes[0].set_title(f"Originale (ID: {idx})")
                axes[0].axis('off')
                axes[1].imshow(anon_np)
                axes[1].set_title(f"Baseline Sim: {sim_orig_anon:.4f}")
                axes[1].axis('off')
                axes[2].imshow(adv_np)
                axes[2].set_title(f"Adversarial Sim: {sim_orig_adv:.4f}")
                axes[2].axis('off')

                plt.tight_layout()
                fname = os.path.join(success_img_dir, f"sample_{idx}_sim_{sim_orig_adv:.2f}.png")
                plt.savefig(fname, dpi=150)
                plt.close(fig)

            results.append({
                "sample_id": idx,
                "attack": f"bb_{args.attack}",
                "loss_mode": args.loss_mode,
                "region_mode": region_mode,
                "epsilon": args.epsilon,
                "delta_linf": round(delta_linf, 4),
                "detector_evaded": success,
                "queries_used": queries,
                "sim_orig_vs_anonymized": round(sim_orig_anon, 4),
                "sim_orig_vs_adversarial_pipe": round(sim_orig_adv, 4),
            })

        df = pd.DataFrame(results)
        output_csv = os.path.join(
            output_dir,
            f"identity_leakage_similarity_{args.attack}_{args.loss_mode}_eps{args.epsilon}.csv"
        )
        df.to_csv(output_csv, index=False)
        print(f"\n[SUCCESSO] Report in: {output_csv}")

        print("\n" + "=" * 85)
        print(f" TABELLA RIASSUNTIVA ({args.model.upper()}, {args.loss_mode.upper()})")
        print("=" * 85)
        print(df.to_string(index=False))

        print("\n" + "=" * 85)
        print(" STATISTICHE")
        print("=" * 85)
        print(df[["delta_linf", "queries_used",
                  "sim_orig_vs_anonymized", "sim_orig_vs_adversarial_pipe"]].describe().to_string())
        print("=" * 85)

        if df["delta_linf"].mean() < 0.01:
            print("\n[WARN] delta_linf ≈ 0: l'attacco NON ha modificato l'immagine.")
        diff = (df["sim_orig_vs_adversarial_pipe"] - df["sim_orig_vs_anonymized"]).mean()
        print(f"\n[DIAG] Δ medio (adv - anon) = {diff:+.4f}")
        if abs(diff) < 0.05:
            print("[WARN] Δ ≈ 0: l'attacco non ha spostato la similarità biometrica.")
            print("       Cause: loss mal segnata, region sbagliata, pipeline stocastica, o attacco inerte.")


if __name__ == "__main__":
    evaluator = BlackBoxEvaluator()
    evaluator.run_blackbox()
