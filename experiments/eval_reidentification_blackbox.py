import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
import argparse
from math import comb
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# IPython display (per mostrare inline in Jupyter/Kaggle)
try:
    from IPython.display import display, Image as IPyImage, HTML
    HAS_IPYTHON_DISPLAY = True
except Exception:
    HAS_IPYTHON_DISPLAY = False

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
        parser.add_argument("--n_queries", type=int, default=None)
        parser.add_argument("--patch_size", type=int, default=4)

        # --- Plot inline ---
        parser.add_argument("--show_plots", dest="show_plots", action="store_true",
                            default=True, help="Mostra plot inline dei sample evasi/leakati (default: True)")
        parser.add_argument("--no_show_plots", dest="show_plots", action="store_false",
                            help="Disabilita plot inline")
        parser.add_argument("--max_plots", type=int, default=20,
                            help="Numero massimo di plot inline (default: 20)")
        args = parser.parse_args()

        attacks_without_patch = {"square", "boundary", "advpatch", "spsa",
                                 "landmark_spoofing", "gfsa"}
        if args.attack.lower() in attacks_without_patch and args.patch_size != 4:
            print(f"[INFO] --patch_size={args.patch_size} ignorato: "
                  f"l'attacco '{args.attack}' non usa patch. Valido solo per SimBA.")

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
                if std_sim > 0.05:
                    print(f"  [ERROR] Pipeline MOLTO stocastica (std={std_sim:.4f}). "
                          f"Leakage impossibile oltre Δ={std_sim*2:.3f}.")
                elif std_sim > 0.03:
                    print(f"  [WARN] Pipeline stocastica (std={std_sim:.4f}). "
                          f"Soglia |Δ| > {std_sim*2:.3f}.")
                else:
                    print(f"  [OK] Pipeline deterministica (std={std_sim:.4f}).")

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

        output_dir = f"./results_identity_{args.model}_blackbox"
        success_img_dir = os.path.join(output_dir, "successful_evasions")
        os.makedirs(success_img_dir, exist_ok=True)

        results = []
        successful_plots = []  # per mostrare inline al termine
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

            # Pixel-level: adv vs pipeline(adv)
            with torch.no_grad():
                img_adv_f = img_adv_uint8.detach().cpu().float()
                img_pipe_adv_chw = self._ensure_chw_tensor(img_pipeline_adv).float()
                if img_pipe_adv_chw.max() <= 1.0:
                    img_pipe_adv_chw = img_pipe_adv_chw * 255.0

                pixel_diff_pipeline = (img_adv_f - img_pipe_adv_chw).abs().mean().item()
                pipeline_bypassed = bool(pixel_diff_pipeline < 5.0)

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

            # Salva plot + registra per mostra inline finale
            is_interesting = pipeline_bypassed or (success and sim_orig_adv >= 0.5)
            if is_interesting:
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
                    fname = os.path.join(success_img_dir,
                                         f"sample_{idx}_bypass{int(pipeline_bypassed)}_sim{sim_orig_adv:.2f}.png")
                    plt.savefig(fname, dpi=150)
                    plt.close(fig)

                    # Etichetta motivo
                    reasons = []
                    if pipeline_bypassed:
                        reasons.append("BYPASS pipeline")
                    if success and sim_orig_adv >= 0.5:
                        reasons.append(f"leakage (sim={sim_orig_adv:.3f})")
                    reason = " + ".join(reasons) if reasons else "success"

                    successful_plots.append({
                        "sample_id": idx,
                        "path": fname,
                        "reason": reason,
                        "pipeline_bypassed": pipeline_bypassed,
                        "delta_linf": delta_linf,
                        "pixel_diff": pixel_diff_pipeline,
                        "sim_anon": sim_orig_anon,
                        "sim_adv": sim_orig_adv,
                        "queries": queries,
                    })
                except Exception as e:
                    print(f"[WARN] plot sample {idx} failed: {e}")

            results.append({
                "sample_id": idx,
                "attack": f"bb_{args.attack}",
                "loss_mode": args.loss_mode,
                "region_mode": region_mode,
                "epsilon": args.epsilon,
                "delta_linf": round(delta_linf, 4),
                "detector_evaded": success,
                "pipeline_bypassed": pipeline_bypassed,
                "pixel_diff_pipeline": round(pixel_diff_pipeline, 2),
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

        print("\n" + "=" * 100)
        print(f" TABELLA RIASSUNTIVA ({args.model.upper()}, {args.loss_mode.upper()})")
        print("=" * 100)
        print(df.to_string(index=False))

        print("\n" + "=" * 100)
        print(" STATISTICHE")
        print("=" * 100)
        stats_cols = ["delta_linf", "pixel_diff_pipeline", "queries_used",
                      "sim_orig_vs_anonymized", "sim_orig_vs_adversarial_pipe"]
        print(df[stats_cols].describe().to_string())
        print("=" * 100)

        # ============================================================
        # DIAGNOSTIC TARGET-AWARE
        # ============================================================
        if df["delta_linf"].mean() < 0.01:
            print("\n[WARN] delta_linf ≈ 0: l'attacco NON ha modificato l'immagine.")

        n_bypassed = int(df["pipeline_bypassed"].sum())
        n_tot = len(df)
        print(f"\n[DIAG] Pipeline bypass: {n_bypassed}/{n_tot} "
              f"({100*n_bypassed/max(n_tot,1):.1f}%)")

        if args.loss_mode == "leakage":
            diff = (df["sim_orig_vs_adversarial_pipe"] - df["sim_orig_vs_anonymized"]).mean()
            diff_med = (df["sim_orig_vs_adversarial_pipe"] - df["sim_orig_vs_anonymized"]).median()
            n_pos = int((df["sim_orig_vs_adversarial_pipe"] > df["sim_orig_vs_anonymized"]).sum())

            noise_floor = 2.0 * std_sim if std_sim is not None else 0.05
            p_value = sum(comb(n_tot, k) for k in range(n_pos, n_tot + 1)) / (2 ** n_tot)

            print(f"[DIAG] Δ medio: {diff:+.4f} | Δ mediano: {diff_med:+.4f}")
            print(f"[DIAG] Sign test: {n_pos}/{n_tot} positivi, p = {p_value:.4f}")
            print(f"[DIAG] Noise floor (2×std): {noise_floor:.4f}")

            if n_bypassed >= n_tot * 0.5:
                print(f"\n[INFO] {n_bypassed}/{n_tot} campioni hanno BYPASSATO la pipeline.")
                print(f"       Il Δ metrico è NEGATIVO perché anche InsightFace è stato evaso.")
                print(f"       Il BYPASS è confermato da pixel_diff_pipeline ≈ 0.")
                measurable = df[df["sim_orig_vs_adversarial_pipe"].abs() > 0.01]
                if len(measurable) > 0:
                    print(f"\n[DIAG] Sample MISURABILI ({len(measurable)}/{n_tot}):")
                    for _, row in measurable.iterrows():
                        d = row["sim_orig_vs_adversarial_pipe"] - row["sim_orig_vs_anonymized"]
                        print(f"  sample {int(row['sample_id'])}: "
                              f"anon={row['sim_orig_vs_anonymized']:.4f} → "
                              f"adv={row['sim_orig_vs_adversarial_pipe']:.4f} (Δ={d:+.4f})")
            else:
                if diff > noise_floor and p_value < 0.05:
                    print(f"[OK] ATTACCO EFFICACE: Δ={diff:+.4f} > soglia {noise_floor:.4f}, p={p_value:.4f}")
                elif p_value < 0.05:
                    print(f"[INFO] Segnale significativo (p={p_value:.4f}) ma Δ={diff:+.4f} < {noise_floor:.4f}. Debole.")
                else:
                    print(f"[WARN] Δ={diff:+.4f} non significativo (p={p_value:.4f}, soglia {noise_floor:.4f}).")
        else:
            n_evaded = int(df["detector_evaded"].sum())
            print(f"\n[DIAG] Evasion detector: {n_evaded}/{n_tot} ({100*n_evaded/max(n_tot,1):.1f}%)")
            if n_bypassed > 0:
                print(f"[OK] ATTACCO FUNZIONA: {n_bypassed}/{n_tot} campioni hanno "
                      f"bypassato la pipeline (volto originale preservato).")
            elif n_evaded > 0:
                print(f"[INFO] Detector evade su {n_evaded}/{n_tot} ma pipeline anonimizza ancora.")
            else:
                print(f"[WARN] Nessuna evasione e nessun bypass.")

        print("\n" + "=" * 100)

        # ============================================================
        # PLOT INLINE DEI SAMPLE EVASI / LEAKATI
        # ============================================================
        if args.show_plots and len(successful_plots) > 0:
            n_to_show = min(len(successful_plots), args.max_plots)
            print(f"\n{'=' * 100}")
            print(f" PLOT INLINE — {n_to_show}/{len(successful_plots)} sample evasi/leakati")
            print(f"{'=' * 100}")
            print(f" Ogni plot mostra: Originale | Anonimizzata (baseline) | De-anonimizzata (pipeline su adv)")
            print(f" File salvati in: {success_img_dir}")
            print(f"{'=' * 100}\n")

            for entry in successful_plots[:n_to_show]:
                header_html = (
                    f"<div style='border:1px solid #ccc; padding:10px; margin:10px 0; "
                    f"background-color:#f9f9f9;'>"
                    f"<h3 style='margin:0;'>Sample {entry['sample_id']} — {entry['reason']}</h3>"
                    f"<p style='margin:5px 0; font-size:13px; color:#333;'>"
                    f"<b>ε = {args.epsilon:.0f}</b> | "
                    f"<b>Δlinf = {entry['delta_linf']:.1f}</b> | "
                    f"<b>queries = {entry['queries']}</b> | "
                    f"<b>pixel_diff = {entry['pixel_diff']:.2f}</b> | "
                    f"<b>pipeline_bypassed = {entry['pipeline_bypassed']}</b><br>"
                    f"<b>sim_anon (baseline) = {entry['sim_anon']:.4f}</b> → "
                    f"<b>sim_adv (de-anonimizzata) = {entry['sim_adv']:.4f}</b> "
                    f"(Δ = {entry['sim_adv'] - entry['sim_anon']:+.4f})"
                    f"</p></div>"
                )

                if HAS_IPYTHON_DISPLAY:
                    display(HTML(header_html))
                    display(IPyImage(filename=entry['path'], width=1200))
                else:
                    # Fallback: matplotlib
                    print(f"\n=== Sample {entry['sample_id']} — {entry['reason']} ===")
                    print(f"    Δlinf={entry['delta_linf']:.1f} | queries={entry['queries']} | "
                          f"sim_anon={entry['sim_anon']:.4f} → sim_adv={entry['sim_adv']:.4f}")
                    try:
                        import matplotlib.image as mpimg
                        img = mpimg.imread(entry['path'])
                        plt.figure(figsize=(15, 5))
                        plt.imshow(img)
                        plt.axis('off')
                        plt.tight_layout()
                        plt.show()
                        plt.close()
                    except Exception as e:
                        print(f"    [WARN] impossibile mostrare {entry['path']}: {e}")

            if len(successful_plots) > n_to_show:
                print(f"\n[INFO] {len(successful_plots) - n_to_show} plot non mostrati "
                      f"(max_plots={args.max_plots}). Tutti salvati in: {success_img_dir}")

            print(f"\n{'=' * 100}")
            print(f" FINE PLOT INLINE — {len(successful_plots)} sample totali")
            print(f"{'=' * 100}\n")
        elif args.show_plots and len(successful_plots) == 0:
            print(f"\n[INFO] Nessun sample evaso/leakato da mostrare (nessun bypass, nessuna sim alta).")


if __name__ == "__main__":
    evaluator = BlackBoxEvaluator()
    evaluator.run_blackbox()
