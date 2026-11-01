import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RETINAFACE_REPO_PATH = "/kaggle/input/datasets/domenicovicenti/pytorch-retinaface-repo"

import os
if os.path.isdir(RETINAFACE_REPO_PATH):
    if RETINAFACE_REPO_PATH not in sys.path:
        sys.path.insert(0, RETINAFACE_REPO_PATH)
    print(f"[INFO] RetinaFace repo path aggiunto al sys.path: {RETINAFACE_REPO_PATH}")
else:
    print(f"[WARN] {RETINAFACE_REPO_PATH} non esiste. "
          f"Se usi --surrogate retinaface, fallirà con ModuleNotFoundError: 'models'.")
    print(f"       Controlla il nome del dataset con: !ls /kaggle/input/")
# =========================================================================

import argparse
import inspect
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from eval_reidentification import BaseEvaluator
from attacks.transfer.fgsm_transfer import TransferFGSMAttack
from attacks.transfer.bim_transfer import TransferBIMAttack
from attacks.transfer.pgd_transfer import TransferPGDAttack
from attacks.transfer.deepfool_transfer import TransferDeepFoolAttack
from attacks.transfer.uap_transfer import TransferUAPAttack
from attacks.transfer.ciagan_identity_leakage_attack import CiaganIdentityLeakageAttack
from data_loaders.lfw import LFWDataset


class TransferEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _extract_image_tensor(output):
        """Estrae il tensore immagine da output che può essere tensor, dict, tuple.
        Necessario per CIAGAN che restituisce {"im": tensor}."""
        if isinstance(output, dict):
            for k in ("im", "image", "output", "result", "images", "faces"):
                if k in output:
                    return output[k]
            for v in output.values():
                if isinstance(v, torch.Tensor):
                    return v
            raise ValueError(f"dict senza tensore: chiavi={list(output.keys())}")
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    @staticmethod
    def _to_chw_tensor(x):
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu()
            if t.dim() == 4:
                t = t.squeeze(0)
            if t.dtype == torch.uint8:
                t = t.float()
            return t
        return torch.from_numpy(np.array(x)).permute(2, 0, 1).float()

    @staticmethod
    def _call_perturb(attack, img, epsilon):
        """Chiama attack.perturb gestendo la firma in modo pulito (niente
        try/except TypeError che maschera errori veri)."""
        sig = inspect.signature(attack.perturb)
        if "epsilon" in sig.parameters:
            return attack.perturb(img, epsilon=epsilon)
        return attack.perturb(img)

    def get_attack_instance(self, attack_name, target, detector_wrapper, dsfd_net,
                            mean_tensor, epsilon, surrogate_net=None):
        attack_name = attack_name.lower()
        common = dict(
            detector=detector_wrapper,
            detector_wrapper=detector_wrapper,
            dsfd_net=dsfd_net,
            mean_tensor=mean_tensor,
            surrogate_net=surrogate_net,
            device=self.device,
        )

        if attack_name == "pgd":
            return TransferPGDAttack(**common, use_momentum=True, use_di=True)
        elif attack_name == "fgsm":
            return TransferFGSMAttack(**common)
        elif attack_name == "bim":
            return TransferBIMAttack(**common)
        elif attack_name == "deepfool":
            return TransferDeepFoolAttack(**common)
        elif attack_name == "uap":
            return TransferUAPAttack(
                detector_wrapper=detector_wrapper, dsfd_net=dsfd_net,
                mean_tensor=mean_tensor, surrogate_net=surrogate_net, device=self.device,
            )
        elif attack_name in ("ciagan_il", "ciagan_identity_leakage"):
            return CiaganIdentityLeakageAttack(
                target=target,
                detector_wrapper=detector_wrapper,
                device=self.device,
                mean_tensor=mean_tensor,
            )
        else:
            raise ValueError(f"Attacco transfer non supportato: {attack_name}")

    def _load_retinaface(self, weights_path):
        """Carica RetinaFace usando il wrapper del repo (RetinaFaceDetector).

        Il wrapper importa `models.retinaface.retinaface.RetinaFace`, che
        richiede che RETINAFACE_REPO_PATH sia nel sys.path (fatto in cima).
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Pesi RetinaFace non trovati: {weights_path}")

        try:
            from detectors.retinaface import RetinaFaceDetector
        except ImportError as ie:
            print(f"[ERROR] Modulo detectors.retinaface non trovato: {ie}")
            raise ie

        # RetinaFaceDetector.__init__ → _load_retinaface → import models.retinaface...
        # Se la cartella `models/` non è nel sys.path, dà ModuleNotFoundError.
        # Catturiamo per dare un messaggio chiaro all'utente.
        try:
            net = RetinaFaceDetector(weights_path=weights_path, device=self.device)
        except ModuleNotFoundError as e:
            if "models" in str(e):
                raise RuntimeError(
                    "RetinaFaceDetector non riesce a importare `models.retinaface.retinaface`.\n"
                    f"Sys.path contiene: {RETINAFACE_REPO_PATH} "
                    f"(esiste={os.path.isdir(RETINAFACE_REPO_PATH)})\n"
                    "Cause probabili:\n"
                    "  1) Il dataset Kaggle del repo RetinaFace non è collegato al notebook\n"
                    "  2) Il nome del dataset è diverso (controlla con `!ls /kaggle/input/`)\n"
                    "  3) La struttura del repo è diversa (es. c'è una sottocartella)\n"
                    f"Dettaglio originale: {e}"
                ) from e
            raise

        print(f"[SUCCESSO] RetinaFace caricato da: {weights_path}")
        return net

    def run_transfer(self):
        parser = argparse.ArgumentParser(description="Valutazione Identity Leakage & Re-identification (Transfer Mode)")
        parser.add_argument("--attack", type=str, default="pgd",
                            choices=["fgsm", "bim", "pgd", "uap", "deepfool", "ciagan_il"])
        parser.add_argument("--model", type=str, default="ciagan",
                            choices=["deeprivacy2", "ciagan"])
        parser.add_argument("--surrogate", type=str, default="",
                            choices=["", "retinaface", "yolo"])
        parser.add_argument("--surrogate_weights", type=str, default="")
        parser.add_argument("--epsilon", type=float, default=8.0)
        parser.add_argument("--num_samples", type=int, default=30)
        args = parser.parse_args()

        print(f"\n[INFO] Avvio Identity Leakage Analysis & Re-identification (TRANSFER MODE)")
        print(f" -> Attacco: {args.attack.upper()} (Epsilon: {args.epsilon})")
        print(f" -> Surrogato: {args.surrogate.upper() if args.surrogate else 'Nessuno'}")
        print(f" -> Target Anonimizzazione: {args.model}")
        print(f" -> Dispositivo: {self.device}")

        target, anonymizer, detector_wrapper, dsfd_net, mean_tensor = self.setup_target(args.model)

        if mean_tensor is None:
            print("[WARN] mean_tensor=None dal target — uso zero-tensor come fallback")
            mean_tensor = torch.zeros(1, 3, 1, 1, device=self.device)

        # --- Surrogato ---
        surrogate_net = None
        if args.surrogate:
            print(f"[INFO] Caricamento modello surrogato ({args.surrogate})...")
            if args.surrogate == "yolo":
                from detectors.yolo import YoloFaceDetector
                weights_path = args.surrogate_weights if args.surrogate_weights else "yolov8n-face.pt"
                surrogate_net = YoloFaceDetector(weights_path=weights_path, device=self.device)
                print(f"[SUCCESSO] YOLO caricato da: {weights_path}")
            elif args.surrogate == "retinaface":
                weights_path = args.surrogate_weights if args.surrogate_weights else \
                    "/kaggle/input/datasets/domenicovicenti/retinaface-weights/mobilenet0.25_Final.pth"
                surrogate_net = self._load_retinaface(weights_path)
        else:
            print("[INFO] Nessun surrogato: l'attacco userà dsfd_net come surrogate differenziabile.")

        # Avviso se non c'è alcun detector differenziabile
        if surrogate_net is None and dsfd_net is None and args.attack.lower() in ("fgsm", "bim", "pgd", "deepfool"):
            print("[WARN] Nessun detector differenziabile (surrogate_net=None, dsfd_net=None).")
            print("       FGSM/BIM/PGD/DeepFool non avranno gradienti → l'attacco sarà inerte.")

        attack = self.get_attack_instance(
            args.attack, target=target, detector_wrapper=detector_wrapper,
            dsfd_net=dsfd_net, mean_tensor=mean_tensor,
            epsilon=args.epsilon, surrogate_net=surrogate_net,
        )

        dataset = LFWDataset(root_dir="/kaggle/input/datasets/jessicali9530/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled")

        if args.attack.lower() == "uap":
            print("[INFO] UAP: avvio fit...")
            from torch.utils.data import DataLoader, Subset
            subset_indices = range(min(100, len(dataset)))
            uap_subset = Subset(dataset, subset_indices)
            train_loader = DataLoader(uap_subset, batch_size=4, shuffle=True)
            attack.fit(train_loader, epsilon=args.epsilon, alpha=2.0, epochs=8, max_iter_per_img=15)

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

            # A. Perturbazione
            try:
                img_adv, success, _ = self._call_perturb(attack, img_tensor[0], args.epsilon)
            except Exception as e:
                print(f"[ERROR] sample {idx} perturb fallito: {type(e).__name__}: {e}")
                raise

            # Diagnostica: l'attacco ha modificato l'immagine?
            delta_linf = (img_adv.float() - img_tensor[0].float()).abs().max().item()

            # B. Pipeline (robusto a dict di CIAGAN)
            with torch.no_grad():
                img_single = img_tensor[0] if img_tensor.ndim == 4 else img_tensor
                if img_single.max() <= 1.0:
                    img_single = img_single * 255.0
                img_single_uint8 = img_single.detach().byte().to(self.device)

                raw_anon = (target.anonymize(img_single_uint8)
                            if hasattr(target, "anonymize")
                            else anonymizer(img_single_uint8))
                img_anonymized = self._extract_image_tensor(raw_anon)

                img_adv_single = img_adv[0] if img_adv.ndim == 4 else img_adv
                if img_adv_single.max() <= 1.0:
                    img_adv_single = img_adv_single * 255.0
                img_adv_uint8 = img_adv_single.detach().byte().to(self.device)

                raw_adv = (target.anonymize(img_adv_uint8)
                           if hasattr(target, "anonymize")
                           else anonymizer(img_adv_uint8))
                img_pipeline_adv = self._extract_image_tensor(raw_adv)

            # C. Similarità (InsightFace)
            sim_orig_anon = 0.0
            sim_orig_adv = 0.0
            if self.recognizer is not None:
                img_anon_chw = self._to_chw_tensor(img_anonymized)
                img_adv_chw = self._to_chw_tensor(img_pipeline_adv)
                emb_orig = self.extract_embedding(img_single.detach().cpu())
                emb_anon = self.extract_embedding(img_anon_chw)
                emb_adv = self.extract_embedding(img_adv_chw)
                if emb_orig is not None and emb_anon is not None:
                    sim_orig_anon = self.compute_cosine_similarity(emb_orig, emb_anon)
                if emb_orig is not None and emb_adv is not None:
                    sim_orig_adv = self.compute_cosine_similarity(emb_orig, emb_adv)

            results.append({
                "sample_id": idx,
                "attack": f"transfer_{args.attack}",
                "surrogate": args.surrogate or "none",
                "epsilon": args.epsilon,
                "delta_linf": round(delta_linf, 4),     # prova che l'attacco ha agito
                "detector_evaded": success,
                "sim_orig_vs_anonymized": round(sim_orig_anon, 4),
                "sim_orig_vs_adversarial_pipe": round(sim_orig_adv, 4),
            })

        df = pd.DataFrame(results)
        output_dir = f"./results_identity_{args.model}_transfer"
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(
            output_dir,
            f"identity_leakage_similarity_{args.attack}_surrogate_{args.surrogate or 'none'}_eps{args.epsilon}.csv",
        )
        df.to_csv(output_csv, index=False)
        print(f"\n[SUCCESSO] Report in: {output_csv}")

        print("\n" + "=" * 100)
        print(" TABELLA RIASSUNTIVA TRANSFER")
        print("=" * 100)
        print(df.to_string(index=False))

        # Diagnostica aggregata delta_linf
        if "delta_linf" in df.columns:
            mean_delta = df["delta_linf"].mean()
            print(f"\n[DIAG] delta_linf medio = {mean_delta:.4f} (max = {df['delta_linf'].max():.4f})")
            if mean_delta < 0.01:
                print("[WARN] delta_linf ≈ 0: l'attacco NON ha modificato l'immagine.")
                print("       Cause: surrogate_net=None, dsfd_net=None, o attacco fallito silenziosamente.")

        print("\n" + "=" * 100)
        print(" STATISTICHE DESCRITTIVE (InsightFace buffalo_l)")
        print("=" * 100)
        print(df[["sim_orig_vs_anonymized", "sim_orig_vs_adversarial_pipe"]].describe().to_string())
        print("=" * 100)


if __name__ == "__main__":
    evaluator = TransferEvaluator()
    evaluator.run_transfer()
