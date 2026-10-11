
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Root del progetto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval_reidentification import BaseEvaluator
from attacks.transfer.pgd_transfer import TransferPGDAttack
from attacks.transfer.fgsm_transfer import TransferFGSMAttack
from data_loaders.lfw import LFWDataset


class TransferEvaluator(BaseEvaluator):
    """
    Evaluazione degli attacchi transfer / gray-box contro DeepPrivacy2.

    Metriche:
    1. Evasione del modello surrogato (RetinaFace o YOLO).
    2. Evasione del detector DSFD utilizzato dalla pipeline target.
    3. Possibile fallimento della de-identificazione, valutato tramite
       similarità ArcFace tra identità originale e output avversario.
    """

    def __init__(self):
        super().__init__()

    # ---------------------------------------------------------
    # FACTORY DEGLI ATTACCHI
    # ---------------------------------------------------------

    def get_attack_instance(
        self,
        attack_name,
        detector_wrapper,
        dsfd_net,
        mean_tensor,
        epsilon,
        surrogate_net=None
    ):
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

        elif attack_name == "fgsm":
            return TransferFGSMAttack(
                detector=detector_wrapper,
                detector_wrapper=detector_wrapper,
                dsfd_net=dsfd_net,
                mean_tensor=mean_tensor,
                surrogate_net=surrogate_net,
                device=self.device
            )

        else:
            raise ValueError(
                f"Attacco '{attack_name}' non implementato in questa factory. "
                "Sono supportati: pgd, fgsm."
            )

    # ---------------------------------------------------------
    # CONTEGGIO DEI VOLTI
    # ---------------------------------------------------------

    def count_faces(self, detector, image_tensor):
        """
        Richiede che il detector esponga count_detections().

        ATTENZIONE:
        il valore restituito è affidabile come numero di volti soltanto
        se il wrapper implementa correttamente il post-processing,
        compresa la NMS, e usa il preprocessing previsto dal modello.
        """
        if detector is None:
            raise RuntimeError("Detector non inizializzato.")

        if not hasattr(detector, "count_detections"):
            raise NotImplementedError(
                f"{type(detector).__name__} non espone count_detections(). "
                "Implementare il conteggio dei volti per questo detector."
            )

        image_tensor = image_tensor.detach()

        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            count = detector.count_detections(image_tensor)

        if count is None:
            raise RuntimeError(
                f"{type(detector).__name__}.count_detections() ha restituito None."
            )

        return int(count)

    # ---------------------------------------------------------
    # CONVERSIONE OUTPUT ANONIMIZZATO
    # ---------------------------------------------------------

    @staticmethod
    def to_chw_tensor(image):
        """
        Converte un'immagine Tensor o NumPy in Tensor CHW su CPU.
        Presuppone che gli array NumPy siano in formato HWC.
        """
        if isinstance(image, torch.Tensor):
            result = image.detach().cpu()

            if result.ndim == 4:
                if result.shape[0] != 1:
                    raise ValueError(
                        "Atteso un singolo campione nell'output della pipeline."
                    )
                result = result.squeeze(0)

            if result.ndim != 3:
                raise ValueError(
                    f"Dimensioni Tensor inattese: {tuple(result.shape)}"
                )

            # Se è HWC, converte in CHW.
            if result.shape[-1] == 3 and result.shape[0] != 3:
                result = result.permute(2, 0, 1)

            return result.contiguous()

        array = np.asarray(image)

        if array.ndim != 3:
            raise ValueError(
                f"Dimensioni NumPy inattese: {array.shape}"
            )

        if array.shape[-1] == 3:
            array = np.transpose(array, (2, 0, 1))

        return torch.from_numpy(np.ascontiguousarray(array))

    # ---------------------------------------------------------
    # ESECUZIONE VALUTAZIONE
    # ---------------------------------------------------------

    def run_transfer(self):

        parser = argparse.ArgumentParser(
            description=(
                "Valutazione Identity Leakage e detector evasion "
                "in modalità Transfer"
            )
        )

        parser.add_argument(
            "--attack",
            type=str,
            default="pgd",
            choices=["fgsm", "pgd"],
            help="Attacco transfer da utilizzare"
        )

        parser.add_argument(
            "--model",
            type=str,
            default="deeprivacy2",
            choices=["deeprivacy2"],
            help="Modello di anonimizzazione target"
        )

        parser.add_argument(
            "--surrogate",
            type=str,
            default="retinaface",
            choices=["retinaface", "yolo"],
            help="Modello surrogato utilizzato per l'attacco"
        )

        parser.add_argument(
            "--surrogate_weights",
            type=str,
            default="",
            help="Percorso ai pesi del modello surrogato"
        )

        parser.add_argument(
            "--epsilon",
            type=float,
            default=8.0,
            help="Budget di perturbazione dell'attacco"
        )

        parser.add_argument(
            "--num_samples",
            type=int,
            default=30,
            help="Numero di campioni LFW da valutare"
        )

        parser.add_argument(
            "--identity_threshold",
            type=float,
            default=0.5,
            help=(
                "Soglia preliminare ArcFace per segnalare possibile "
                "perdita di identità; deve essere calibrata"
            )
        )

        args = parser.parse_args()

        # -----------------------------------------------------
        # INFORMAZIONI INIZIALI
        # -----------------------------------------------------

        print("\n" + "=" * 78)
        print(" VALUTAZIONE TRANSFER / GRAY-BOX")
        print("=" * 78)

        print(f"Attacco: {args.attack.upper()}")
        print(f"Epsilon: {args.epsilon}")
        print(f"Surrogato: {args.surrogate.upper()}")
        print(f"Target: {args.model}")
        print(f"Device: {self.device}")
        print(f"Soglia ArcFace preliminare: {args.identity_threshold}")

        # -----------------------------------------------------
        # 1. CARICAMENTO DEEPRIVACY2 E DSFD
        # -----------------------------------------------------

        print("\n[INFO] Caricamento target e detector DSFD...")

        target, anonymizer, detector_wrapper, dsfd_net, mean_tensor = (
            self.setup_target(args.model)
        )

        if detector_wrapper is None:
            raise RuntimeError(
                "setup_target() non ha restituito detector_wrapper."
            )

        print("[SUCCESSO] Target caricato.")

        # -----------------------------------------------------
        # 2. CARICAMENTO MODELLO SURROGATO
        # -----------------------------------------------------

        surrogate_net = None

        print(
            f"\n[INFO] Caricamento surrogato {args.surrogate.upper()}..."
        )

        if args.surrogate == "retinaface":

            from detectors.retinaface import RetinaFaceDetector

            default_weights = (
                "/kaggle/input/datasets/domenicovicenti/"
                "retinaface-weights/mobilenet0.25_Final.pth"
            )

            weights_path = (
                args.surrogate_weights
                if args.surrogate_weights
                else default_weights
            )

            if not os.path.isfile(weights_path):
                raise FileNotFoundError(
                    f"Pesi RetinaFace non trovati: {weights_path}"
                )

            surrogate_net = RetinaFaceDetector(
                weights_path=weights_path,
                device=self.device
            )

            print(
                f"[SUCCESSO] RetinaFace caricato da: {weights_path}"
            )

        elif args.surrogate == "yolo":

            from detectors.yolo import YoloFaceDetector

            default_weights = "yolov8n-face.pt"

            weights_path = (
                args.surrogate_weights
                if args.surrogate_weights
                else default_weights
            )

            if not os.path.isfile(weights_path):
                raise FileNotFoundError(
                    f"Pesi YOLO non trovati: {weights_path}"
                )

            surrogate_net = YoloFaceDetector(
                weights_path=weights_path,
                device=self.device
            )

            print(f"[SUCCESSO] YOLO caricato da: {weights_path}")

        # -----------------------------------------------------
        # 3. INIZIALIZZAZIONE ATTACCO
        # -----------------------------------------------------

        attack = self.get_attack_instance(
            attack_name=args.attack,
            detector_wrapper=detector_wrapper,
            dsfd_net=dsfd_net,
            mean_tensor=mean_tensor,
            epsilon=args.epsilon,
            surrogate_net=surrogate_net
        )

        # -----------------------------------------------------
        # 4. CARICAMENTO DATASET
        # -----------------------------------------------------

        dataset = LFWDataset(
            root_dir=(
                "/kaggle/input/datasets/jessicali9530/"
                "lfw-dataset/lfw-deepfunneled/lfw-deepfunneled"
            )
        )

        num_samples = min(args.num_samples, len(dataset))

        print(
            f"\n[INFO] Campioni da elaborare: {num_samples}"
        )

        results = []

        # Contatori metriche
        retinaface_evasion_count = 0
        dsfd_evasion_count = 0
        deidentification_failure_count = 0

        # Denominatori: campioni per cui la metrica è valutabile
        valid_retinaface_count = 0
        valid_dsfd_count = 0
        valid_identity_count = 0

        # -----------------------------------------------------
        # 5. CICLO PRINCIPALE
        # -----------------------------------------------------

        for idx in tqdm(range(num_samples)):

            sample = dataset[idx]
            img_orig = (
                sample[0]
                if isinstance(sample, (list, tuple))
                else sample
            )

            if not isinstance(img_orig, torch.Tensor):
                img_orig = torch.as_tensor(img_orig)

            # Dataset HWC -> CHW, se necessario.
            if (
                img_orig.ndim == 3
                and img_orig.shape[-1] == 3
                and img_orig.shape[0] != 3
            ):
                img_orig = img_orig.permute(2, 0, 1)

            img_tensor = (
                img_orig.clone()
                .detach()
                .to(self.device)
                .float()
            )

            if img_tensor.ndim != 3:
                print(
                    f"[WARN] Campione {idx} ignorato: "
                    f"shape inattesa {tuple(img_tensor.shape)}"
                )
                continue

            # Input batch per i detector.
            img_batch = img_tensor.unsqueeze(0)

            # -------------------------------------------------
            # 5A. GENERAZIONE IMMAGINE AVVERSARIA
            # -------------------------------------------------

            try:
                img_adv, attack_success, _ = attack.perturb(
                    img_tensor,
                    epsilon=args.epsilon
                )
            except TypeError:
                img_adv, attack_success, _ = attack.perturb(
                    img_tensor
                )

            if not isinstance(img_adv, torch.Tensor):
                img_adv = torch.as_tensor(img_adv)

            img_adv = img_adv.detach().to(self.device).float()

            if img_adv.ndim == 4:
                if img_adv.shape[0] != 1:
                    raise ValueError(
                        "L'attacco ha restituito più immagini avversarie "
                        "per un singolo campione."
                    )
                img_adv = img_adv.squeeze(0)

            if img_adv.ndim != 3:
                raise ValueError(
                    f"Shape immagine avversaria inattesa: "
                    f"{tuple(img_adv.shape)}"
                )

            img_adv_batch = img_adv.unsqueeze(0)

            # Inizializzazione valori per questo campione.
            retinaface_faces = None
            retinaface_evaded = None

            dsfd_faces = None
            dsfd_evaded = None

            identity_similarity_valid = False
            identity_failure = None

            sim_orig_anon = None
            sim_orig_adv = None

            emb_orig = None
            emb_anon = None
            emb_adv_pipe = None

            # -------------------------------------------------
            # METRICA 1: EVASIONE RETINAFACE / SURROGATO
            # -------------------------------------------------

            if args.surrogate == "retinaface":

                try:
                    retinaface_faces = self.count_faces(
                        surrogate_net,
                        img_adv_batch
                    )

                    retinaface_evaded = (
                        retinaface_faces == 0
                    )

                    valid_retinaface_count += 1

                    if retinaface_evaded:
                        retinaface_evasion_count += 1

                except (
                    RuntimeError,
                    NotImplementedError,
                    ValueError
                ) as exc:
                    print(
                        f"\n[WARN] Campione {idx}: "
                        f"valutazione RetinaFace non disponibile: {exc}"
                    )

            # -------------------------------------------------
            # METRICA 2: EVASIONE DSFD
            # -------------------------------------------------

            try:
                # Il wrapper deve applicare il preprocessing corretto
                # e restituire un conteggio di volti post-processing.
                dsfd_faces = self.count_faces(
                    detector_wrapper,
                    img_adv_batch
                )

                dsfd_evaded = (
                    dsfd_faces == 0
                )

                valid_dsfd_count += 1

                if dsfd_evaded:
                    dsfd_evasion_count += 1

            except (
                RuntimeError,
                NotImplementedError,
                ValueError
            ) as exc:
                print(
                    f"\n[WARN] Campione {idx}: "
                    f"valutazione DSFD non disponibile: {exc}"
                )

            # -------------------------------------------------
            # 5B. PIPELINE DEEPRIVACY2
            # -------------------------------------------------

            with torch.no_grad():

                img_single = img_tensor.detach().clone()

                img_single_uint8 = img_single

                if img_single_uint8.max() <= 1.0:
                    img_single_uint8 = img_single_uint8 * 255.0

                img_single_uint8 = (
                    img_single_uint8
                    .clamp(0, 255)
                    .to(torch.uint8)
                )

                img_adv_uint8 = img_adv.detach().clone()

                if img_adv_uint8.max() <= 1.0:
                    img_adv_uint8 = img_adv_uint8 * 255.0

                img_adv_uint8 = (
                    img_adv_uint8
                    .clamp(0, 255)
                    .to(torch.uint8)
                )

                if hasattr(target, "anonymize"):

                    img_anonymized = target.anonymize(
                        img_single_uint8
                    )

                    img_pipeline_adv = target.anonymize(
                        img_adv_uint8
                    )

                elif callable(anonymizer):

                    img_anonymized = anonymizer(
                        img_single_uint8
                    )

                    img_pipeline_adv = anonymizer(
                        img_adv_uint8
                    )

                else:
                    raise AttributeError(
                        "Target e anonymizer non espongono "
                        "un metodo di anonimizzazione utilizzabile."
                    )

            # -------------------------------------------------
            # 5C. ARC FACE: SIMILARITÀ DI IDENTITÀ
            # -------------------------------------------------

            if self.recognizer is not None:

                try:
                    img_anon_chw = self.to_chw_tensor(
                        img_anonymized
                    )

                    img_adv_chw = self.to_chw_tensor(
                        img_pipeline_adv
                    )

                    img_orig_chw = img_single.detach().cpu()

                    emb_orig = self.extract_embedding(
                        img_orig_chw
                    )

                    emb_anon = self.extract_embedding(
                        img_anon_chw
                    )

                    emb_adv_pipe = self.extract_embedding(
                        img_adv_chw
                    )

                    if (
                        emb_orig is not None
                        and emb_anon is not None
                    ):
                        sim_orig_anon = float(
                            self.compute_cosine_similarity(
                                emb_orig,
                                emb_anon
                            )
                        )

                    if (
                        emb_orig is not None
                        and emb_adv_pipe is not None
                    ):
                        sim_orig_adv = float(
                            self.compute_cosine_similarity(
                                emb_orig,
                                emb_adv_pipe
                            )
                        )

                        identity_similarity_valid = (
                            np.isfinite(sim_orig_adv)
                        )

                        if identity_similarity_valid:

                            valid_identity_count += 1

                            identity_failure = (
                                sim_orig_adv
                                >= args.identity_threshold
                            )

                            if identity_failure:
                                deidentification_failure_count += 1

                except Exception as exc:
                    print(
                        f"\n[WARN] Campione {idx}: "
                        f"errore nell'analisi ArcFace: {exc}"
                    )

            # -------------------------------------------------
            # 5D. RISULTATI PER CAMPIONE
            # -------------------------------------------------

            if retinaface_evaded is None:
                retinaface_text = "NON VALUTABILE"
            else:
                retinaface_text = (
                    f"{'SÌ' if retinaface_evaded else 'NO'} "
                    f"({retinaface_faces} volti)"
                )

            if dsfd_evaded is None:
                dsfd_text = "NON VALUTABILE"
            else:
                dsfd_text = (
                    f"{'SÌ' if dsfd_evaded else 'NO'} "
                    f"({dsfd_faces} volti)"
                )

            if identity_failure is None:
                identity_text = (
                    "NON VALUTABILE: embedding non disponibile"
                )
            else:
                identity_text = (
                    f"{'POSSIBILE FALLIMENTO' if identity_failure else 'NON SEGNALATO'} "
                    f"(similarità={sim_orig_adv:.4f}, "
                    f"soglia={args.identity_threshold:.2f})"
                )

            print(f"\n\n{'-' * 70}")
            print(f"CAMPIONE {idx}")
            print(f"{'-' * 70}")

            print(
                "[METRICA 1] Evasione RetinaFace: "
                f"{retinaface_text}"
            )

            print(
                "[METRICA 2] Evasione DSFD: "
                f"{dsfd_text}"
            )

            print(
                "[METRICA 3] Fallimento de-identificazione: "
                f"{identity_text}"
            )

            # -------------------------------------------------
            # 5E. REGISTRAZIONE CSV
            # -------------------------------------------------

            results.append({
                "sample_id": idx,
                "attack": f"transfer_{args.attack}",
                "surrogate": args.surrogate,
                "epsilon": args.epsilon,

                # Output originale dell'attacco.
                "attack_success": (
                    bool(attack_success)
                    if attack_success is not None
                    else None
                ),

                # Metrica 1.
                "retinaface_faces_detected": retinaface_faces,
                "retinaface_evaded": retinaface_evaded,

                # Metrica 2.
                "dsfd_faces_detected": dsfd_faces,
                "dsfd_evaded": dsfd_evaded,

                # Metrica 3.
                "identity_similarity_valid": identity_similarity_valid,
                "identity_threshold": args.identity_threshold,
                "deidentification_failure": identity_failure,

                # Similarità ArcFace.
                "sim_orig_vs_anonymized": (
                    round(sim_orig_anon, 4)
                    if sim_orig_anon is not None
                    else None
                ),

                "sim_orig_vs_adversarial_pipe": (
                    round(sim_orig_adv, 4)
                    if sim_orig_adv is not None
                    else None
                )
            })

        # -----------------------------------------------------
        # 6. SALVATAGGIO CSV
        # -----------------------------------------------------

        output_dir = (
            f"./results_identity_{args.model}_transfer"
        )

        os.makedirs(output_dir, exist_ok=True)

        output_csv = os.path.join(
            output_dir,
            (
                f"identity_leakage_similarity_{args.attack}"
                f"_surrogate_{args.surrogate}"
                f"_eps{args.epsilon}.csv"
            )
        )

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)

        print(f"\n[SUCCESSO] Report salvato in: {output_csv}")

        # -----------------------------------------------------
        # 7. TABELLA E STATISTICHE DESCRITTIVE
        # -----------------------------------------------------

        print("\n" + "=" * 78)
        print(" RISULTATI PER CAMPIONE")
        print("=" * 78)

        if not df.empty:
            print(df.to_string(index=False))

            similarity_columns = [
                "sim_orig_vs_anonymized",
                "sim_orig_vs_adversarial_pipe"
            ]

            print("\n" + "=" * 78)
            print(" STATISTICHE DESCRITTIVE ARCFACE")
            print("=" * 78)

            print(
                df[similarity_columns]
                .describe()
                .to_string()
            )

        # -----------------------------------------------------
        # 8. RIEPILOGO DELLE TRE METRICHE
        # -----------------------------------------------------

        def print_metric(name, successes, valid):
            if valid == 0:
                print(
                    f"{name}: NON VALUTABILE "
                    "(nessun campione valido)"
                )
                return

            rate = 100.0 * successes / valid

            print(
                f"{name}: {successes}/{valid} "
                f"({rate:.2f}%)"
            )

        print("\n" + "=" * 78)
        print(" RIEPILOGO DELLE TRE METRICHE DI SICUREZZA")
        print("=" * 78)

        print_metric(
            "1. Evasione RetinaFace",
            retinaface_evasion_count,
            valid_retinaface_count
        )

        print_metric(
            "2. Evasione DSFD",
            dsfd_evasion_count,
            valid_dsfd_count
        )

        print_metric(
            "3. Possibile fallimento della de-identificazione",
            deidentification_failure_count,
            valid_identity_count
        )

        print("-" * 78)
        print(f"Campioni elaborati: {len(df)}")
        print(f"CSV: {output_csv}")
        print("=" * 78)


if __name__ == "__main__":
    evaluator = TransferEvaluator()
    evaluator.run_transfer()
