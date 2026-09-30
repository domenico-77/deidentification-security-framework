import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import inspect
import types
import subprocess

# --- INSTALLAZIONE AUTOMATICA DIPENDENZE (INSIGHTFACE & ONNXRUNTIME) ---
try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    print("[INFO] InsightFace non trovato. Installazione automatica in corso...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "insightface", "onnxruntime-gpu"])
        import insightface
        from insightface.app import FaceAnalysis
        HAS_INSIGHTFACE = True
        print("[SUCCESSO] InsightFace installato correttamente.")
    except Exception as e:
        print(f"[WARNING] Impossibile installare automaticamente InsightFace: {e}")
        HAS_INSIGHTFACE = False

# --- PATCH DI SICUREZZA ROBUSTA PER PYTHON 3.12 (GETFILE & GETSOURCEFILE) ---
_old_getfile = inspect.getfile
def _safe_getfile(object):
    try:
        f = _old_getfile(object)
        if not isinstance(f, (str, bytes, os.PathLike)):
            return __file__
        return f
    except Exception:
        return __file__
inspect.getfile = _safe_getfile

_old_getsourcefile = inspect.getsourcefile
def _safe_getsourcefile(object):
    try:
        f = _old_getsourcefile(object)
        if f is not None and not isinstance(f, (str, bytes, os.PathLike)):
            return None
        return f
    except Exception:
        return None
inspect.getsourcefile = _safe_getsourcefile
# --------------------------------------------------------------------------

import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2

from targets.deeprivacy2 import DeepPrivacy2Target

from attacks.fgsm import FGSMAttack
from attacks.bim import BIMAttack
from attacks.pgd import PGDAttack
from attacks.uap import UAPAttack
from attacks.deepfool import DeepFoolAttack

from data_loaders.lfw import LFWDataset

def get_attack_instance(attack_name, detector_wrapper, dsfd_net, mean_tensor, device, epsilon):
    """Factory per selezionare l'attacco desiderato da riga di comando."""
    attack_name = attack_name.lower()
    if attack_name == "fgsm":
        return FGSMAttack(detector_wrapper=detector_wrapper, dsfd_net=dsfd_net, mean_tensor=mean_tensor, device=device)
    elif attack_name == "bim":
        return BIMAttack(detector_wrapper=detector_wrapper, dsfd_net=dsfd_net, mean_tensor=mean_tensor, device=device)
    elif attack_name == "pgd":
        return PGDAttack(detector_wrapper=detector_wrapper, dsfd_net=dsfd_net, mean_tensor=mean_tensor, device=device)
    elif attack_name == "deepfool":
        return DeepFoolAttack(detector_wrapper=detector_wrapper, dsfd_net=dsfd_net, mean_tensor=mean_tensor, device=device)
    elif attack_name == "uap":
        return UAPAttack(detector_wrapper=detector_wrapper, dsfd_net=dsfd_net, mean_tensor=mean_tensor, device=device)
    else:
        raise ValueError(f"Attacco non supportato o non valido: {attack_name}")

def compute_cosine_similarity(emb1, emb2):
    """Calcola la similarità coseno tra due vettori di embedding."""
    if emb1 is None or emb2 is None:
        return 0.0
    emb1 = torch.tensor(emb1) if not isinstance(emb1, torch.Tensor) else emb1
    emb2 = torch.tensor(emb2) if not isinstance(emb2, torch.Tensor) else emb2
    emb1 = F.normalize(emb1.float().flatten().unsqueeze(0), p=2, dim=1)
    emb2 = F.normalize(emb2.float().flatten().unsqueeze(0), p=2, dim=1)
    return F.cosine_similarity(emb1, emb2).item()

def main():
    parser = argparse.ArgumentParser(description="Valutazione Identity Leakage & Re-identification Confronto")
    parser.add_argument("--attack", type=str, default="bim", choices=["fgsm", "bim", "pgd", "uap", "deepfool"], help="Tipo di attacco da testare")
    parser.add_argument("--model", type=str, default="deeprivacy2", choices=["deeprivacy2"], help="Modello di anonimizzazione")
    parser.add_argument("--epsilon", type=float, default=8.0, help="Valore di epsilon per la perturbazione")
    parser.add_argument("--num_samples", type=int, default=30, help="Numero di campioni del dataset da valutare")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Avvio Identity Leakage Analysis & Re-identification")
    print(f" -> Attacco: {args.attack.upper()} (Epsilon: {args.epsilon})")
    print(f" -> Modello di anonimizzazione: {args.model}")
    print(f" -> Dispositivo: {device}")

    # Inizializzazione Recognizer Facciale (ArcFace / InsightFace)
    recognizer = None
    if HAS_INSIGHTFACE:
        try:
            recognizer = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider' if device=='cuda' else 'CPUExecutionProvider'])
            recognizer.prepare(ctx_id=0 if device=='cuda' else -1, det_size=(112, 112))
            print("[INFO] Modello ArcFace (InsightFace - antelopev2) caricato con successo.")
        except Exception as e:
            print(f"[WARNING] Impossibile inizializzare InsightFace con antelopev2, provo buffalo_l: {e}")
            try:
                recognizer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider' if device=='cuda' else 'CPUExecutionProvider'])
                recognizer.prepare(ctx_id=0 if device=='cuda' else -1, det_size=(112, 112))
                print("[INFO] Modello ArcFace (InsightFace - buffalo_l) caricato con successo.")
            except Exception as e2:
                print(f"[ERROR] Impossibile inizializzare alcun modello InsightFace: {e2}")

    # 1. Caricamento del target di anonimizzazione
    if args.model == "deeprivacy2":
        target = DeepPrivacy2Target(models_dir="/kaggle/input/datasets/domenicovicenti/deep-privacy2-models")
        anonymizer = target.pipeline if hasattr(target, "pipeline") else target.anonymizer
        
        # PATCH DINAMICA AL VOLO per forward_G
        if hasattr(anonymizer, "forward_G"):
            _orig_forward_G = anonymizer.forward_G
            def _patched_forward_G(self, *args, multi_modal_truncation=False, amp=False, truncation_value=0.5, **kwargs):
                return _orig_forward_G(*args, multi_modal_truncation=multi_modal_truncation, amp=amp, truncation_value=truncation_value, **kwargs)
            anonymizer.forward_G = types.MethodType(_patched_forward_G, anonymizer)

        detector_wrapper = anonymizer.detector
        dsfd_net = detector_wrapper.face_detector.net.to(device).eval()
        mean_tensor = detector_wrapper.face_mean.to(device).float().flatten().view(1, 3, 1, 1)
    else:
        raise NotImplementedError(f"Modello {args.model} non ancora integrato.")

    # 2. Inizializzazione dell'attacco scelto
    attack = get_attack_instance(args.attack, detector_wrapper, dsfd_net, mean_tensor, device, args.epsilon)

    # 3. Caricamento dataset
    dataset = LFWDataset(root_dir="/kaggle/input/datasets/jessicali9530/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled")
    results = []

    print(f"[INFO] Elaborazione di {min(args.num_samples, len(dataset))} campioni con analisi di similarità...")

    def extract_embedding(img_tensor_chw):
        """Estrae l'embedding facciale usando ArcFace/InsightFace con supporto robusto per LFW."""
        if recognizer is None:
            return None
        try:
            if isinstance(img_tensor_chw, torch.Tensor):
                img_np = img_tensor_chw.detach().cpu().permute(1, 2, 0).numpy()
            else:
                img_np = np.array(img_tensor_chw)
                if img_np.ndim == 3 and img_np.shape[0] == 3:
                    img_np = np.transpose(img_np, (1, 2, 0))

            if img_np.max() <= 1.0:
                img_np = img_np * 255.0
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            # RGB -> BGR per OpenCV / InsightFace
            if img_np.shape[2] == 3:
                img_np = img_np[:, :, ::-1]

            # 1. Tentativo con il detector standard di InsightFace
            faces = recognizer.get(img_np)
            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                for attr in ['embedding', 'normed_embedding']:
                    if hasattr(faces[0], attr) and getattr(faces[0], attr) is not None:
                        emb = getattr(faces[0], attr)
                        if isinstance(emb, np.ndarray):
                            return emb

            # 2. Fallback diretto sul modello di riconoscimento se il volto in LFW è già centrato
            if hasattr(recognizer, 'models') and 'recognition' in recognizer.models:
                resized = cv2.resize(img_np, (112, 112))
                emb = recognizer.models['recognition'].get(resized)
                if emb is not None:
                    return emb
        except Exception:
            pass
        return None

    for idx in tqdm(range(min(args.num_samples, len(dataset)))):
        sample = dataset[idx]
        img_orig = sample[0] if isinstance(sample, (list, tuple)) else sample
        
        if img_orig.ndim == 3 and img_orig.shape[2] == 3:
            img_orig = img_orig.permute(2, 0, 1)
            
        img_tensor = img_orig.clone().detach().to(device).float()
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # A. Generazione immagine perturbata tramite l'attacco selezionato
        if hasattr(attack, "perturb"):
            try:
                img_adv, success, _ = attack.perturb(img_tensor[0], epsilon=args.epsilon)
            except TypeError:
                img_adv, success, _ = attack.perturb(img_tensor[0])
        else:
            raise AttributeError("L'oggetto attacco non possiede un metodo .perturb() valido.")

        # B. Passaggio attraverso la pipeline di anonimizzazione
        with torch.no_grad():
            img_single = img_tensor[0] if img_tensor.ndim == 4 else img_tensor
            if img_single.max() <= 1.0:
                img_single = img_single * 255.0
            img_single_uint8 = img_single.detach().byte().to(device)
            
            # 1. Immagine originale -> Anonimizzata (GAN)
            if hasattr(target, "anonymize"):
                img_anonymized = target.anonymize(img_single_uint8)
            elif hasattr(anonymizer, "__call__"):
                img_anonymized = anonymizer(img_single_uint8)
            else:
                raise AttributeError("Il target DeepPrivacy2 non possiede un metodo di anonimizzazione valido.")
            
            # 2. Immagine perturbata -> Pipeline
            img_adv_single = img_adv[0] if img_adv.ndim == 4 else img_adv
            if img_adv_single.max() <= 1.0:
                img_adv_single = img_adv_single * 255.0
            img_adv_uint8 = img_adv_single.detach().byte().to(device)
            
            if hasattr(target, "anonymize"):
                img_pipeline_adv = target.anonymize(img_adv_uint8)
            elif hasattr(anonymizer, "__call__"):
                img_pipeline_adv = anonymizer(img_adv_uint8)
            else:
                raise AttributeError("Il target DeepPrivacy2 non possiede un metodo di anonimizzazione valido.")

        # C. Estrazione Embedding e Calcolo Similarità (ArcFace)
        sim_orig_anon = 0.0
        sim_orig_adv = 0.0

        if recognizer is not None:
            if isinstance(img_anonymized, torch.Tensor):
                img_anon_chw = img_anonymized.detach().cpu()
                if img_anon_chw.dim() == 4: img_anon_chw = img_anon_chw.squeeze(0)
            else:
                img_anon_chw = torch.from_numpy(np.array(img_anonymized)).permute(2, 0, 1)

            if isinstance(img_pipeline_adv, torch.Tensor):
                img_adv_chw = img_pipeline_adv.detach().cpu()
                if img_adv_chw.dim() == 4: img_adv_chw = img_adv_chw.squeeze(0)
            else:
                img_adv_chw = torch.from_numpy(np.array(img_pipeline_adv)).permute(2, 0, 1)

            emb_orig = extract_embedding(img_single.detach().cpu())
            emb_anon = extract_embedding(img_anon_chw)
            emb_adv_pipe = extract_embedding(img_adv_chw)

            if emb_orig is not None and emb_anon is not None:
                sim_orig_anon = compute_cosine_similarity(emb_orig, emb_anon)
            if emb_orig is not None and emb_adv_pipe is not None:
                sim_orig_adv = compute_cosine_similarity(emb_orig, emb_adv_pipe)

        # D. Registrazione metriche estese
        results.append({
            "sample_id": idx,
            "attack": args.attack,
            "epsilon": args.epsilon,
            "detector_evaded": success,
            "pipeline_behavior": "Bypassed (Original face retained)" if success else "Anonymized (Face replaced)",
            "sim_orig_vs_anonymized": round(sim_orig_anon, 4),
            "sim_orig_vs_adversarial_pipe": round(sim_orig_adv, 4)
        })

    # Salvataggio risultati
    df = pd.DataFrame(results)
    output_dir = f"./results_identity_{args.model}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_csv = os.path.join(output_dir, f"identity_leakage_similarity_{args.attack}_eps{args.epsilon}.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n[SUCCESSO] Analisi di similarità completata. Report salvato in: {output_csv}")

    # STAMPA TABELLA RIASSUNTIVA E STATISTICHE A TERMINALE
    print("\n" + "="*80)
    print(" TABELLA RIASSUNTIVA DEI RISULTATI (Campioni elaborati)")
    print("="*80)
    print(df.to_string(index=False))
    
    print("\n" + "="*80)
    print(" STATISTICHE DESCRITTIVE DELLE SIMILARITÀ (ArcFace)")
    print("="*80)
    print(df[["sim_orig_vs_anonymized", "sim_orig_vs_adversarial_pipe"]].describe().to_string())
    print("="*80)

if __name__ == "__main__":
    main()
