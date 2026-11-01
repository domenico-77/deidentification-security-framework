import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
import contextlib
import io
import yaml
import torch
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Subset

from targets.ciagan import CIAGANTarget
from attacks.transfer.ciagan_identity_leakage_attack import CiaganIdentityLeakageAttack
from benchmark.benchmark_runner import BenchmarkRunner
from data_loaders.lfw import LFWDataset

warnings.filterwarnings("ignore")


# =========================================================
# Pre-flight
# =========================================================
def _check_torch_health():
    print("\n---- Torch/CUDA health check ----")
    print(f"  torch:      {torch.__version__}")
    try:
        import numpy as _np
        print(f"  numpy:      {_np.__version__}")
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            _ = torch.zeros(1, device="cuda") + 1
            print(f"  cuda:       OK (device={torch.cuda.get_device_name(0)})")
        except Exception as e:
            print(f"  cuda:       FAIL — {type(e).__name__}: {e}")
            raise RuntimeError("CUDA non operativo — riavvia il runtime.")
    else:
        print("  cuda:       non disponibile (CPU mode)")
    print("---- end health check ----\n")


def _probe_attack(attack: CiaganIdentityLeakageAttack, target, device: str):
    print("\n================ PRE-FLIGHT CHECK ================")
    ok = True

    oracolo = attack._id_surrogate
    ttype = attack._surrogate_type
    if oracolo is None:
        print("[FAIL] Nessun oracolo biometrico disponibile.")
        ok = False
    else:
        print(f"[ OK ] Oracolo: {type(oracolo).__name__} (backend={ttype})")

    if attack.netG is None:
        print("[FAIL] netG non caricato.")
        ok = False
    else:
        print(f"[ OK ] netG caricato: {sum(p.numel() for p in attack.netG.parameters()):,} params")

    if attack.predictor is None:
        print("[WARN] dlib shape predictor non trovato: fallback ellipse.")
    else:
        print("[ OK ] dlib shape predictor caricato")

    print("==================================================\n")
    if not ok:
        raise RuntimeError("Pre-flight fallito.")


# =========================================================
# Utility
# =========================================================
def _tensor_to_numpy(img_tensor):
    if isinstance(img_tensor, (tuple, list)):
        img_tensor = img_tensor[0]
    if not isinstance(img_tensor, torch.Tensor):
        img_tensor = torch.tensor(img_tensor)
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    arr = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _process_with_target(target, img_tensor):
    with torch.no_grad():
        if hasattr(target, "run"):
            out = target.run(img_tensor)
        elif hasattr(target, "anonymize"):
            out = target.anonymize(img_tensor)
        elif hasattr(target, "forward"):
            out = target.forward(img_tensor)
        else:
            out = target(img_tensor)

        if isinstance(out, dict):
            for k in ("im", "image", "output", "result", "images", "faces"):
                if k in out:
                    out = out[k]
                    break
            else:
                out = list(out.values())[0]

        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


# =========================================================
# Plot (invariato)
# =========================================================
def generate_all_plots(attack, target, dataset, device, output_dir, epsilon=16.0):
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"\n[INFO] Generazione plot per tutti i campioni in {plots_dir}...")

    for idx in range(len(dataset)):
        sample = dataset[idx]
        if isinstance(sample, (tuple, list)):
            img_orig, img_path = sample[0], sample[1]
        else:
            img_orig, img_path = sample, f"sample_{idx}.jpg"
        sample_name = os.path.basename(str(img_path))

        if not isinstance(img_orig, torch.Tensor):
            img_orig = torch.tensor(img_orig)
        img_orig = img_orig.unsqueeze(0).to(device).float()
        if img_orig.max() <= 1.5:
            img_orig = img_orig * 255.0

        with contextlib.redirect_stdout(io.StringIO()):
            img_adv, success, _ = attack.perturb(img_orig, epsilon=epsilon)

        img_base_out = _process_with_target(target, img_orig)
        img_adv_out = _process_with_target(target, img_adv)

        orig_np = _tensor_to_numpy(img_orig)
        base_np = _tensor_to_numpy(img_base_out)
        adv_np = _tensor_to_numpy(img_adv_out)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_np)
        axes[0].set_title(f"Volto Originale\n({sample_name})")
        axes[0].axis('off')
        axes[1].imshow(base_np)
        axes[1].set_title("Target (Baseline)\n[Volto Anonimizzato]")
        axes[1].axis('off')
        axes[2].imshow(adv_np)
        axes[2].set_title(f"Target (Adversarial)\n[Evasione: {success}]")
        axes[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(plots_dir, f"plot_{idx:02d}_{Path(sample_name).stem}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> Salvato: {save_path} (success={success})")

    print("[INFO] Generazione plot completata.\n")


# =========================================================
# NUOVO: metriche biometriche (tabella separata)
# =========================================================
def generate_biometric_table(attack, dataset, device, epsilons,
                             suppress_logs=True):
    """Ricalcola le metriche biometriche per ciascun sample × epsilon.

    Sfrutta gli attributi `_last_*` popolati internamente da `attack.perturb()`:
      - _last_oracle_sim          = cos(e_orig, e_CIAGAN(adv))
      - _last_baseline_orig_sim   = cos(e_orig, e_CIAGAN(clean))
      - _last_baseline_sim        = cos(e_CIAGAN(clean), e_CIAGAN(adv))
    """
    results = []

    for eps in epsilons:
        print(f"\n[INFO] Calcolo metriche biometriche per epsilon={eps} "
              f"({len(dataset)} campioni)...")
        for idx in tqdm(range(len(dataset)), desc=f"Bio eps={eps}"):
            sample = dataset[idx]
            if isinstance(sample, (tuple, list)):
                img_orig = sample[0]
                sample_name = os.path.basename(str(sample[1])) if len(sample) > 1 else f"sample_{idx}"
            else:
                img_orig = sample
                sample_name = f"sample_{idx}"

            img_tensor = img_orig.to(device).float()
            if img_tensor.max() <= 1.5:
                img_tensor = img_tensor * 255.0

            if suppress_logs:
                with contextlib.redirect_stdout(io.StringIO()):
                    _, success, _ = attack.perturb(img_tensor, epsilon=eps)
            else:
                _, success, _ = attack.perturb(img_tensor, epsilon=eps)

            final_sim = float(getattr(attack, "_last_oracle_sim", 0.0))
            baseline_orig_sim = float(getattr(attack, "_last_baseline_orig_sim", 0.0))
            baseline_vs_adv = float(getattr(attack, "_last_baseline_sim", 0.0))
            improvement = final_sim - baseline_orig_sim

            results.append({
                "sample": sample_name,
                "epsilon": eps,
                "sim_orig_vs_adv": round(final_sim, 4),
                "sim_orig_vs_baseline": round(baseline_orig_sim, 4),
                "improvement": round(improvement, 4),
                "baseline_vs_adv": round(baseline_vs_adv, 4),
                "success": bool(success),
            })

    return pd.DataFrame(results)


def _print_biometric_summary(bio_df, epsilons):
    # ---- Tabella completa ----
    print("\n" + "=" * 100)
    print(" TABELLA — METRICHE BIOMETRICHE")
    print("=" * 100)
    print(bio_df.to_string(index=False))

    # ---- Legenda ----
    print("\n" + "=" * 100)
    print(" LEGENDA")
    print("=" * 100)
    print("  sim_orig_vs_adv      = cos(e_orig, e_CIAGAN(adv))")
    print("                          → alta = il riconoscitore dice che sono la stessa persona")
    print()
    print("  sim_orig_vs_baseline = cos(e_orig, e_CIAGAN(clean))")
    print("                          → bassa = CIAGAN anonimizza bene l'originale")
    print()
    print("  improvement          = sim_orig_vs_adv - sim_orig_vs_baseline")
    print("                          → alta = il rumore sposta la similarità verso l'originale")
    print()
    print("  baseline_vs_adv      = cos(e_CIAGAN(clean), e_CIAGAN(adv))")
    print("                          → bassa = i due output di CIAGAN sono volti diversi, eppure")
    print("                            il riconoscitore li confonde entrambi con l'originale")
    print("                            (questo è il caso più forte: rumore efficace nonostante")
    print("                            la pipeline abbia generato volti visivamente diversi)")
    print("=" * 100)

    # ---- Statistiche per epsilon ----
    print("\n" + "=" * 100)
    print(" STATISTICHE AGGREGATE PER EPSILON")
    print("=" * 100)
    for eps in epsilons:
        sub = bio_df[bio_df["epsilon"] == eps]
        if len(sub) == 0:
            continue
        print(f"\nε = {eps}")
        cols = ["sim_orig_vs_adv", "sim_orig_vs_baseline", "improvement", "baseline_vs_adv"]
        stats = sub[cols].describe().loc[["mean", "std", "min", "max"]]
        print(stats.to_string())

    # ---- Verdetto automatico ----
    print("\n" + "=" * 100)
    print(" VERDETTO — Interpretazione automatica")
    print("=" * 100)
    for eps in epsilons:
        sub = bio_df[bio_df["epsilon"] == eps]
        if len(sub) == 0:
            continue

        m_adv = sub["sim_orig_vs_adv"].mean()
        m_base = sub["sim_orig_vs_baseline"].mean()
        m_imp = sub["improvement"].mean()
        m_bvsa = sub["baseline_vs_adv"].mean()
        n_eff = int(((sub["sim_orig_vs_adv"] > 0.55) &
                     (sub["improvement"] > 0.20)).sum())
        total = len(sub)

        print(f"\nε = {eps}")
        print(f"  sim_orig_vs_baseline (mean): {m_base:.4f}   — CIAGAN anonimizza: "
              f"{'sì' if m_base < 0.4 else 'parzialmente' if m_base < 0.6 else 'NO'}")
        print(f"  sim_orig_vs_adv (mean):      {m_adv:.4f}   — Riconoscitore confuso: "
              f"{'sì' if m_adv > 0.55 else 'no'}")
        print(f"  improvement (mean):          {m_imp:+.4f}  — Rumore efficace: "
              f"{'sì' if m_imp > 0.20 else 'parzialmente' if m_imp > 0.10 else 'no'}")
        print(f"  baseline_vs_adv (mean):      {m_bvsa:.4f}   — Volti generati diversi: "
              f"{'sì' if m_bvsa < 0.85 else 'NO (stesso volto)'}")
        print(f"  Casi efficaci: {n_eff}/{total}")

        if m_adv > 0.55 and m_imp > 0.20 and m_bvsa < 0.85:
            print("  → ✅ ATTACCO EFFICACE (metrico): rumore inganna il riconoscitore, "
                  "i volti generati sono diversi")
        elif m_imp > 0.10:
            print("  → ⚠️  PARZIALMENTE EFFICACE: il rumore sposta la similarità, "
                  "ma sotto la soglia di verifica")
        else:
            print("  → ❌ NON EFFICACE: il rumore non inganna il riconoscitore")
    print("\n" + "=" * 100)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Run CIAGAN Identity Leakage Attack Benchmark")
    parser.add_argument("--config", type=str, default="configs/attacks_ciagan.yaml")
    parser.add_argument("--dataset_path", type=str, required=True, help="Percorso del dataset LFW")
    parser.add_argument("--output_dir", type=str, default="./results_ciagan_identity_leakage")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--skip_preflight", action="store_true")
    parser.add_argument("--bio_epsilons", type=float, nargs="+",
                        default=[16.0, 32.0, 64.0, 128.0],
                        help="Epsilon per cui calcolare le metriche biometriche")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo di esecuzione: {device}")

    _check_torch_health()

    models_dir = "/kaggle/input/datasets/domenicovicenti/ciagan-models"
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            models_dir = config.get("target", {}).get("models_dir", models_dir)

    print("Caricamento del target CIAGAN...")
    target = CIAGANTarget(models_dir=models_dir, device=device)

    arcface_model = None
    for attr in ("arcface", "id_model", "recognizer", "face_recognizer",
                 "identity_model", "embedder", "id_net", "recog",
                 "oracle", "surrogate", "face_net",
                 "recognition_model", "feature_extractor"):
        cand = getattr(target, attr, None)
        if cand is not None and not isinstance(cand, str):
            arcface_model = cand
            break

    print(f"[INFO] Modello ArcFace associato all'attacco: {arcface_model is not None}")

    print("Avvio CIAGAN Identity Leakage Attack...")
    attack = CiaganIdentityLeakageAttack(
        target=target,
        device=device,
        arcface_model=arcface_model,
    )

    if not args.skip_preflight:
        _probe_attack(attack, target, device)

    base_dataset = LFWDataset(root_dir=args.dataset_path)
    num_samples_to_use = min(args.num_samples, len(base_dataset))
    dataset = Subset(base_dataset, range(num_samples_to_use))
    print(f"[INFO] Dataset LFW: {len(base_dataset)} totali, uso {num_samples_to_use} campioni")

    # ---- 1. BENCHMARK STANDARD (CSV con metriche pixel-space) ----
    runner = BenchmarkRunner(
        target=target,
        attack=attack,
        dataset=dataset,
        device=device,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    runner.run_benchmark(
        epsilons=[16.0, 32.0, 64.0, 128.0],
        num_samples=num_samples_to_use,
        output_dir=args.output_dir,
    )
    print(f"\nBenchmark completato. Risultati in {args.output_dir}")

    # ---- 2. PLOT PER TUTTI I CAMPIONI ----
    generate_all_plots(
        attack=attack, target=target, dataset=dataset,
        device=device, output_dir=args.output_dir, epsilon=16.0,
    )

    # ---- 3. SUMMARY STANDARD (dal CSV) ----
    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            print("\n=== RISULTATI AGGREGATI PER EPSILON (metriche pixel-space) ===")
            print(df.to_string(index=False))
            if "evaded" in df.columns:
                n_total = len(df)
                n_success = int(df["evaded"].sum())
                print(f"\n[INFO] Success rate (evaded): {n_success}/{n_total} "
                      f"({100.0 * n_success / max(n_total, 1):.1f}%)")
        except Exception as e:
            print(f"[WARN] Post-run summary failed: {e}")

    # ---- 4. METRICHE BIOMETRICHE (tabella separata) ----
    print("\n" + "=" * 100)
    print(" METRICHE BIOMETRICHE — effetto dell'attacco sul riconoscitore")
    print("=" * 100)
    oracolo_name = type(attack._id_surrogate).__name__ if attack._id_surrogate is not None else "N/A"
    print(f" Oracolo usato:  {oracolo_name}")
    print(f" Epsilon:        {args.bio_epsilons}")
    print("=" * 100)

    bio_df = generate_biometric_table(
        attack, dataset, device,
        epsilons=args.bio_epsilons,
        suppress_logs=True,
    )

    bio_csv = os.path.join(args.output_dir, "biometric_metrics.csv")
    bio_df.to_csv(bio_csv, index=False)
    print(f"\n[SUCCESSO] Metriche biometriche salvate in: {bio_csv}\n")

    _print_biometric_summary(bio_df, args.bio_epsilons)


if __name__ == "__main__":
    main()
