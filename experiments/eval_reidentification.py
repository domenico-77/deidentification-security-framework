import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from targets.deeprivacy2 import DeepPrivacy2Target
from attacks.fgsm import FGSMAttack
from attacks.bim import BIMAttack
from attacks.pgd import PGDAttack
from attacks.uap import UAPAttack
from data_loaders.lfw import LFWDataset

def get_attack_instance(attack_name, detector_wrapper, dsfd_net, mean_tensor, device, epsilon):
    """Factory per selezionare l'attacco desiderato da riga di comando."""
    attack_name = attack_name.lower()
    if attack_name == "fgsm":
        return FGSMAttack(detector_wrapper, dsfd_net, mean_tensor, device=device)
    elif attack_name == "bim":
        return BIMAttack(detector_wrapper, dsfd_net, mean_tensor, device=device)
    elif attack_name == "pgd":
        return PGDAttack(detector_wrapper, dsfd_net, mean_tensor, device=device)
    else:
        raise ValueError(f"Attacco non supportato o non valido: {attack_name}")

def main():
    parser = argparse.ArgumentParser(description="Valutazione Identity Leakage & Re-identification Confronto")
    parser.add_argument("--attack", type=str, default="bim", choices=["fgsm", "bim", "pgd", "uap"], help="Tipo di attacco da testare")
    parser.add_argument("--model", type=str, default="deeprivacy2", choices=["deeprivacy2"], help="Modello di anonimizzazione")
    parser.add_argument("--epsilon", type=float, default=8.0, help="Valore di epsilon per la perturbazione")
    parser.add_argument("--num_samples", type=int, default=30, help="Numero di campioni del dataset da valutare")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Avvio Identity Leakage Analysis")
    print(f" -> Attacco: {args.attack.upper()} (Epsilon: {args.epsilon})")
    print(f" -> Modello di anonimizzazione: {args.model}")
    print(f" -> Dispositivo: {device}")

    # 1. Caricamento del target di anonimizzazione
    if args.model == "deeprivacy2":
        target = DeepPrivacy2Target(models_dir="/kaggle/input/datasets/domenicovicenti/deep-privacy2-models")
        anonymizer = target.pipeline if hasattr(target, "pipeline") else target.anonymizer
        detector_wrapper = anonymizer.detector
        dsfd_net = detector_wrapper.face_detector.net.to(device).eval()
        mean_tensor = detector_wrapper.face_mean.to(device).float().flatten().view(1, 3, 1, 1)
    else:
        raise NotImplementedError(f"Modello {args.model} non ancora integrato.")

    # 2. Inizializzazione dell'attacco scelto
    attack = get_attack_instance(args.attack, detector_wrapper, dsfd_net, mean_tensor, device, args.epsilon)

    # 3. Caricamento dataset
    dataset = LFWDataset(root_dir="/kaggle/input/datasets/domenicovicenti/lfw-dataset")
    results = []

    print(f"[INFO] Elaborazione di {min(args.num_samples, len(dataset))} campioni...")

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
            # Gestione firma metodo perturb a seconda dell'implementazione (alcuni richiedono epsilon come argomento)
            try:
                img_adv, success, _ = attack.perturb(img_tensor[0], epsilon=args.epsilon)
            except TypeError:
                img_adv, success, _ = attack.perturb(img_tensor[0])
        else:
            raise AttributeError("L'oggetto attacco non possiede un metodo .perturb() valido.")

        # B. Passaggio attraverso la pipeline di anonimizzazione
        with torch.no_grad():
            # 1. Immagine originale -> Anonimizzata (GAN)
            img_anonymized = target.anonymize(img_tensor) if hasattr(target, "anonymize") else target(img_tensor)
            
            # 2. Immagine perturbata -> Pipeline (se il detector fallisce, bypassa e tiene l'originale rumorosa)
            img_adv_tensor = img_adv.unsqueeze(0) if img_adv.ndim == 3 else img_adv
            img_pipeline_adv = target.anonymize(img_adv_tensor) if hasattr(target, "anonymize") else target(img_adv_tensor)

        # C. Registrazione metriche e comportamento
        results.append({
            "sample_id": idx,
            "attack": args.attack,
            "epsilon": args.epsilon,
            "detector_evaded": success,
            "pipeline_behavior": "Bypassed (Original face retained)" if success else "Anonymized (Face replaced)"
        })

    # Salvataggio risultati
    df = pd.DataFrame(results)
    output_dir = f"./results_identity_{args.model}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_csv = os.path.join(output_dir, f"identity_leakage_{args.attack}_eps{args.epsilon}.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n[SUCCESSO] Analisi completata. Report salvato in: {output_csv}")

if __name__ == "__main__":
    main()
