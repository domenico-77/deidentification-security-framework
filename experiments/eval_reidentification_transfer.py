import sys
from pathlib import Path
# Aggiungiamo la root del progetto al path per importare eval_reidentification e i moduli degli attacchi
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import torch
import pandas as pd
from tqdm import tqdm

from eval_reidentification import BaseEvaluator
from attacks.transfer.pgd_transfer import TransferPGDAttack
from data_loaders.lfw import LFWDataset

class TransferEvaluator(BaseEvaluator):
    """
    Evaluator specializzato per Transfer / Gray-Box Attacks.
    Eredita l'intero motore di valutazione e gestione InsightFace da BaseEvaluator.
    """
    def __init__(self):
        super().__init__()

    def get_attack_instance(self, attack_name, detector_wrapper, dsfd_net, mean_tensor, epsilon, surrogate_net=None):
        """Sovrascrive la factory per restituire gli attacchi di trasferimento."""
        attack_name = attack_name.lower()
        if attack_name == "pgd":
            return TransferPGDAttack(
                detector=detector_wrapper,          
                detector_wrapper=detector_wrapper,
                dsfd_net=dsfd_net,
                mean_tensor=mean_tensor,
                surrogate_net=surrogate_net,
                device=self.device,
                use_momentum=True,
                use_di=True
            )
        elif attack_name == "bim":
            # Posto per futuri attacchi transfer (es. BIM Transfer)
            raise NotImplementedError("BIM Transfer non ancora implementato.")
        elif attack_name == "uap":
            raise NotImplementedError("UAP Transfer non ancora implementato.")
        else:
            raise ValueError(f"Attacco transfer non supportato o non valido: {attack_name}")

    def run_transfer(self):
        parser = argparse.ArgumentParser(description="Valutazione Identity Leakage & Re-identification (Transfer Mode)")
        parser.add_argument("--attack", type=str, default="pgd", choices=["fgsm", "bim", "pgd", "uap", "deepfool"], help="Tipo di attacco transfer da testare")
        parser.add_argument("--model", type=str, default="deeprivacy2", choices=["deeprivacy2"], help="Modello di anonimizzazione target")
        parser.add_argument("--surrogate", type=str, default="retinaface", choices=["retinaface"], help="Modello surrogato per il calcolo dei gradienti")
        parser.add_argument("--surrogate_weights", type=str, default="", help="Percorso ai pesi del modello surrogato (es. .pt o .pth)")
        parser.add_argument("--epsilon", type=float, default=8.0, help="Valore di epsilon per la perturbazione")
        parser.add_argument("--num_samples", type=int, default=30, help="Numero di campioni del dataset da valutare")
        args = parser.parse_args()

        print(f"\n[INFO] Avvio Identity Leakage Analysis & Re-identification (TRANSFER MODE)")
        print(f" -> Attacco: {args.attack.upper()} (Epsilon: {args.epsilon})")
        print(f" -> Surrogato: {args.surrogate.upper()} (Pesi: {args.surrogate_weights or 'Default/None'})")
        print(f" -> Target Anonimizzazione: {args.model}")
        print(f" -> Dispositivo: {self.device}")

        # 1. Caricamento del target di anonimizzazione tramite metodo ereditato
        target, anonymizer, detector_wrapper, dsfd_net, mean_tensor = self.setup_target(args.model)

        # 2. Caricamento del Modello Surrogato
        surrogate_net = None
        print(f"[INFO] Caricamento modello surrogato ({args.surrogate})...")
        if args.surrogate == "yolo":
            try:
                from ultralytics import YOLO
                yolo_model = YOLO(args.surrogate_weights if args.surrogate_weights else "yolov8n-face.pt")
                surrogate_net = yolo_model.model.to(self.device).eval()
                print("[SUCCESSO] Modello surrogato YOLO caricato correttamente.")
            except Exception as e:
                print(f"[WARNING] Impossibile caricare YOLO nativo: {e}")
                
        elif args.surrogate == "retinaface":
            try:
                # Assicurati che il modulo retinaface sia disponibile nel path o nel progetto
                from detectors.retinaface import RetinaFaceDetector
                # Inizializzazione RetinaFace con configurazione MobileNet0.25 (come i tuoi pesi)
                cfg = {
                    'name': 'mobilenet0.25',
                    'min_sizes': [[16, 32], [64, 128], [256, 512]],
                    'steps': [8, 16, 32],
                    'variance': [0.1, 0.2],
                    'clip': False,
                    'loc_weight': 2.0,
                    'cls_weight': 1.0,
                    'landm_weight': 1.0,
                    'pretrain': False
                }
                surrogate_net = RetinaFace(cfg=cfg, phase='test')
                
                weights_path = args.surrogate_weights if args.surrogate_weights else "/kaggle/input/datasets/domenicovicenti/retinaface-weights/mobilenet0.25_Final.pth"
                checkpoint = torch.load(weights_path, map_location=self.device)
                
                if 'state_dict' in checkpoint:
                    checkpoint = checkpoint['state_dict']
                # Rimuove eventuali prefissi 'module.' salvati da DataParallel
                new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
                surrogate_net.load_state_dict(new_state_dict, strict=False)
                
                surrogate_net = surrogate_net.to(self.device).eval()
                print(f"[SUCCESSO] RetinaFace caricato correttamente da: {weights_path}")
            except Exception as e:
                print(f"[ERROR] Errore nel caricamento dei pesi di RetinaFace: {e}")
                raise e

        # 3. Istanziazione dell'attacco di trasferimento tramite factory personalizzata
        attack = self.get_attack_instance(
            args.attack, 
            detector_wrapper=detector_wrapper, 
            dsfd_net=dsfd_net, 
            mean_tensor=mean_tensor, 
            epsilon=args.epsilon, 
            surrogate_net=surrogate_net
        )

        # 4. Caricamento dataset LFW
        dataset = LFWDataset(root_dir="/kaggle/input/datasets/jessicali9530/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled")

        results = []
        print(f"[INFO] Elaborazione di {min(args.num_samples, len(dataset))} campioni con analisi di similarità (Transfer)...")

        for idx in tqdm(range(min(args.num_samples, len(dataset)))):
            sample = dataset[idx]
            img_orig = sample[0] if isinstance(sample, (list, tuple)) else sample
            
            if img_orig.ndim == 3 and img_orig.shape[2] == 3:
                img_orig = img_orig.permute(2, 0, 1)
                
            img_tensor = img_orig.clone().detach().to(self.device).float()
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)

            # A. Generazione immagine perturbata tramite l'attacco di trasferimento
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
                img_single_uint8 = img_single.detach().byte().to(self.device)
                
                if hasattr(target, "anonymize"):
                    img_anonymized = target.anonymize(img_single_uint8)
                elif hasattr(anonymizer, "__call__"):
                    img_anonymized = anonymizer(img_single_uint8)
                else:
                    raise AttributeError("Il target DeepPrivacy2 non possiede un metodo di anonimizzazione valido.")
                
                img_adv_single = img_adv[0] if img_adv.ndim == 4 else img_adv
                if img_adv_single.max() <= 1.0:
                    img_adv_single = img_adv_single * 255.0
                img_adv_uint8 = img_adv_single.detach().byte().to(self.device)
                
                if hasattr(target, "anonymize"):
                    img_pipeline_adv = target.anonymize(img_adv_uint8)
                elif hasattr(anonymizer, "__call__"):
                    img_pipeline_adv = anonymizer(img_adv_uint8)
                else:
                    raise AttributeError("Il target DeepPrivacy2 non possiede un metodo di anonimizzazione valido.")

            # C. Estrazione Embedding e Calcolo Similarità (tramite metodi ereditati)
            sim_orig_anon = 0.0
            sim_orig_adv = 0.0

            if self.recognizer is not None:
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

                emb_orig = self.extract_embedding(img_single.detach().cpu())
                emb_anon = self.extract_embedding(img_anon_chw)
                emb_adv_pipe = self.extract_embedding(img_adv_chw)

                if emb_orig is not None and emb_anon is not None:
                    sim_orig_anon = self.compute_cosine_similarity(emb_orig, emb_anon)
                if emb_orig is not None and emb_adv_pipe is not None:
                    sim_orig_adv = self.compute_cosine_similarity(emb_orig, emb_adv_pipe)

            # D. Registrazione metriche estese
            results.append({
                "sample_id": idx,
                "attack": f"transfer_{args.attack}",
                "surrogate": args.surrogate,
                "epsilon": args.epsilon,
                "detector_evaded": success,
                "pipeline_behavior": "Bypassed (Original face retained)" if success else "Anonymized (Face replaced)",
                "sim_orig_vs_anonymized": round(sim_orig_anon, 4),
                "sim_orig_vs_adversarial_pipe": round(sim_orig_adv, 4)
            })

        # Salvataggio risultati in CSV
        df = pd.DataFrame(results)
        output_dir = f"./results_identity_{args.model}_transfer"
        os.makedirs(output_dir, exist_ok=True)
        
        output_csv = os.path.join(output_dir, f"identity_leakage_similarity_{args.attack}_surrogate_{args.surrogate}_eps{args.epsilon}.csv")
        df.to_csv(output_csv, index=False)
        print(f"\n[SUCCESSO] Analisi di similarità transfer completata. Report salvato in: {output_csv}")

        # STAMPA TABELLA RIASSUNTIVA E STATISTICHE A TERMINALE
        print("\n" + "="*80)
        print(" TABELLA RIASSUNTIVA TRANSFER (Campioni elaborati)")
        print("="*80)
        print(df.to_string(index=False))
        
        print("\n" + "="*80)
        print(" STATISTICHE DESCRITTIVE DELLE SIMILARITÀ TRANSFER (ArcFace)")
        print("="*80)
        print(df[["sim_orig_vs_anonymized", "sim_orig_vs_adversarial_pipe"]].describe().to_string())
        print("="*80)

if __name__ == "__main__":
    evaluator = TransferEvaluator()
    evaluator.run_transfer()
