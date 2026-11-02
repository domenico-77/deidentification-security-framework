import torch
import numpy as np
from abc import ABC, abstractmethod


class BaseBlackBoxAttack(ABC):
    """
    Classe base per tutti gli attacchi black-box.

    Due modalità di loss:
      - "evasion" → minimizza il numero di volti rilevati dal detector
      - "leakage" → massimizza la similarità biometrica originale ↔ output pipeline

    Due modalità di regione perturbabile:
      - "face" → rettangolo del volto (corretto per DeepPrivacy2)
      - "ring" → anello periferico attorno al volto (corretto per CIAGAN)
    """

    def __init__(
        self,
        detector_wrapper,
        device="cuda",
        n_queries=1000,
        loss_mode="evasion",
        region_mode="face",
        anonymizer_fn=None,
        extract_emb_fn=None,
        compute_sim_fn=None,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.detector_wrapper = detector_wrapper
        self.n_queries = n_queries

        self.loss_mode = loss_mode.lower()
        self.region_mode = region_mode.lower()
        self.anonymizer_fn = anonymizer_fn
        self.extract_emb_fn = extract_emb_fn
        self.compute_sim_fn = compute_sim_fn

        self._cached_emb_orig = None

        if self.loss_mode == "leakage":
            if not (anonymizer_fn and extract_emb_fn and compute_sim_fn):
                raise ValueError(
                    "loss_mode='leakage' richiede anonymizer_fn, extract_emb_fn, compute_sim_fn."
                )

    @staticmethod
    def _extract_image_tensor(output):
        """Unwrap dict/tuple/tensore. CIAGAN restituisce {"im": tensor}."""
        if isinstance(output, dict):
            for k in ("im", "image", "img", "output", "result", "images", "faces", "anonymized"):
                if k in output:
                    return output[k]
            for v in output.values():
                if isinstance(v, torch.Tensor):
                    return v
            raise ValueError(f"dict senza tensore: chiavi={list(output.keys())}")
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def _parse_detections(self, dets):
        """Normalizza in lista di righe [x1, y1, x2, y2, conf].

        Gestisce: None, [], [None], [tensor(N,5)], [tensor(5,)].
        """
        if dets is None:
            return []

        if isinstance(dets, (list, tuple)):
            if len(dets) == 0:
                return []
            first = dets[0]
            if first is None:
                return []
            if isinstance(first, torch.Tensor):
                if first.dim() == 2:
                    return first.detach().cpu().numpy().astype(float).tolist()
                if first.dim() == 1:
                    return [first.detach().cpu().numpy().astype(float).tolist()]
            if isinstance(first, np.ndarray):
                if first.ndim == 2:
                    return first.astype(float).tolist()
                if first.ndim == 1:
                    return [first.astype(float).tolist()]
            out = []
            for item in dets:
                if item is None:
                    continue
                if isinstance(item, (list, tuple, np.ndarray)):
                    out.append(list(map(float, item)))
                elif isinstance(item, torch.Tensor):
                    out.append(item.detach().cpu().numpy().astype(float).tolist())
            return out

        if isinstance(dets, torch.Tensor):
            arr = dets.detach().cpu().numpy()
            if arr.size == 0:
                return []
            if arr.ndim == 2:
                return arr.astype(float).tolist()
            if arr.ndim == 1:
                return [arr.astype(float).tolist()]
            return []

        if isinstance(dets, np.ndarray):
            if dets.size == 0:
                return []
            if dets.ndim == 2:
                return dets.astype(float).tolist()
            if dets.ndim == 1:
                return [dets.astype(float).tolist()]
            return []

        for attr in ["boxes", "detections", "data", "dets", "_boxes", "pred_boxes"]:
            if hasattr(dets, attr):
                val = getattr(dets, attr)
                if val is not None:
                    return self._parse_detections(val)

        return []

    def _get_face_roi(self, img_tensor):
        with torch.no_grad():
            eval_input = (img_tensor.squeeze(0).byte().float()
                          if img_tensor.ndim == 4 else img_tensor.byte().float())
            if eval_input.ndim == 4:
                eval_input = eval_input[0]

            dets = self.detector_wrapper(eval_input)
            boxes = self._parse_detections(dets)

            _, _, h, w = img_tensor.shape if img_tensor.ndim == 4 else (
                1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])

            if len(boxes) > 0:
                b = boxes[0]
                x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                pad_h = int((y2 - y1) * 0.2)
                pad_w = int((x2 - x1) * 0.2)
                ymin = max(0, int(y1) - pad_h)
                xmin = max(0, int(x1) - pad_w)
                ymax = min(h, int(y2) + pad_h)
                xmax = min(w, int(x2) + pad_w)
                return [ymin, xmin, ymax, xmax]

        return [int(h * 0.2), int(w * 0.2), int(h * 0.8), int(w * 0.8)]

    def _get_perturbation_mask(self, img_tensor):
        """Maschera (1, H, W) con 1.0 dove si può perturbare.

        - "face" → rettangolo del volto (DeepPrivacy2)
        - "ring" → anello attorno al volto, escluso il volto (CIAGAN)
        """
        _, _, h, w = img_tensor.shape if img_tensor.ndim == 4 else (
            1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])
        ymin, xmin, ymax, xmax = self._get_face_roi(img_tensor)

        mask = torch.zeros((1, h, w), device=self.device)
        if self.region_mode == "face":
            mask[0, ymin:ymax, xmin:xmax] = 1.0
        elif self.region_mode == "ring":
            face_box = torch.zeros((h, w), device=self.device)
            face_box[ymin:ymax, xmin:xmax] = 1.0
            pad_h = int((ymax - ymin) * 0.6)
            pad_w = int((xmax - xmin) * 0.6)
            yy1 = max(0, ymin - pad_h)
            xx1 = max(0, xmin - pad_w)
            yy2 = min(h, ymax + pad_h)
            xx2 = min(w, xmax + pad_w)
            outer = torch.zeros((h, w), device=self.device)
            outer[yy1:yy2, xx1:xx2] = 1.0
            mask[0] = (outer - face_box).clamp(0.0, 1.0)
        else:
            raise ValueError(f"region_mode non valido: {self.region_mode}")

        return mask

    def _get_loss(self, img_tensor, emb_orig=None):
        if self.loss_mode == "evasion":
            return self._get_detector_loss(img_tensor)
        elif self.loss_mode == "leakage":
            if emb_orig is None:
                emb_orig = self._embed_orig(img_tensor)
            return self._get_biometric_loss(img_tensor, emb_orig)
        else:
            raise ValueError(f"loss_mode non valido: {self.loss_mode}")

    def _get_detector_loss(self, img_tensor):
        with torch.no_grad():
            eval_input = (img_tensor.squeeze(0).byte().float()
                          if img_tensor.ndim == 4 else img_tensor.byte().float())
            if eval_input.ndim == 4:
                eval_input = eval_input[0]

            dets = self.detector_wrapper(eval_input)
            boxes = self._parse_detections(dets)

            if len(boxes) == 0:
                return 0.0

            max_conf = 0.0
            for box in boxes:
                if len(box) >= 5:
                    c = float(box[4])
                    if c > max_conf:
                        max_conf = c

            return float(len(boxes)) * 10.0 + max_conf

    def _reset_embedding_cache(self):
        self._cached_emb_orig = None

    def _embed_orig(self, img_orig_tensor):
        if self._cached_emb_orig is not None:
            return self._cached_emb_orig
        img_single = img_orig_tensor[0] if img_orig_tensor.ndim == 4 else img_orig_tensor
        if img_single.max() <= 1.0:
            img_single = img_single * 255.0
        emb = self.extract_emb_fn(img_single.detach().cpu())
        self._cached_emb_orig = emb
        return emb

    def _get_biometric_loss(self, img_tensor, emb_orig):
        """L = -cos(emb_orig, emb_anon). Minimizzare = massimizzare sim."""
        with torch.no_grad():
            img_single = img_tensor[0] if img_tensor.ndim == 4 else img_tensor
            if img_single.max() <= 1.0:
                img_single = img_single * 255.0
            img_uint8 = img_single.detach().byte()

            try:
                img_anon = self.anonymizer_fn(img_uint8)
            except Exception as e:
                print(f"[BaseBlackBox] anonymizer error: {e}")
                return 0.0

            img_anon = self._extract_image_tensor(img_anon)

            if isinstance(img_anon, torch.Tensor):
                img_anon_chw = img_anon.detach().cpu()
                if img_anon_chw.dim() == 4:
                    img_anon_chw = img_anon_chw.squeeze(0)
                if img_anon_chw.dtype == torch.uint8:
                    img_anon_chw = img_anon_chw.float()
            else:
                img_anon_chw = torch.from_numpy(np.array(img_anon)).permute(2, 0, 1).float()

            emb_anon = self.extract_emb_fn(img_anon_chw)
            if emb_orig is None or emb_anon is None:
                return 0.0

            sim = self.compute_sim_fn(emb_orig, emb_anon)
            return -float(sim)

    def _is_evaded(self, img_tensor):
        with torch.no_grad():
            eval_input = (img_tensor.squeeze(0).byte().float()
                          if img_tensor.ndim == 4 else img_tensor.byte().float())
            if eval_input.ndim == 4:
                eval_input = eval_input[0]
            dets = self.detector_wrapper(eval_input)
            boxes = self._parse_detections(dets)
            return len(boxes) == 0

    @abstractmethod
    def perturb(self, img_tensor, epsilon=16.0):
        pass
