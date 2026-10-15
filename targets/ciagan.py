import sys
from pathlib import Path

import cv2
import dlib
import numpy as np
import torch
from PIL import Image

KAGGLE_INPUT = Path("/kaggle/input")

if KAGGLE_INPUT.exists():
    repo_matches = list(KAGGLE_INPUT.glob("**/ciagan-repository"))
    models_matches = list(KAGGLE_INPUT.glob("**/ciagan-models"))

    if repo_matches:
        ciagan_repo_root = (repo_matches[0] / "ciagan").resolve()
    else:
        ciagan_repo_root = Path("/kaggle/input/ciagan-repository/ciagan").resolve()

    if models_matches:
        ciagan_models_dir = models_matches[0].resolve()
    else:
        ciagan_models_dir = Path("/kaggle/input/ciagan-models").resolve()
else:
    base_local = Path("./data")
    ciagan_repo_root = (base_local / "ciagan-repository/ciagan").resolve()
    ciagan_models_dir = (base_local / "ciagan-models").resolve()

ciagan_source_dir = ciagan_repo_root / "source"

for p in (ciagan_repo_root, ciagan_source_dir):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from targets.base_target import BaseDeidentificationTarget


# Costanti di preprocessing ufficiali CIAGAN
CIAGAN_W = 178
CIAGAN_H = 218
MODEL_SIZE = 128
PRE_RESIZE = 144  # 128 * 1.125
CENTER_OFFSET = (PRE_RESIZE - MODEL_SIZE) // 2
NUM_IDENTITIES = 1200


class DlibDetectorWrapper:
    """Wrapper compatibile con BenchmarkRunner che utilizza dlib per rilevare i volti."""
    def __init__(self, detector, device):
        self.detector = detector
        self.device = device

    def __call__(self, img_tensor):
        if img_tensor.ndim == 4:
            img = img_tensor[0]
        else:
            img = img_tensor
        
        if img.max() <= 1.0:
            img = img * 255.0
        img_np = img.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        rects = self.detector(gray, 1)
        boxes = []
        for r in rects:
            boxes.append([r.left(), r.top(), r.right(), r.bottom(), 1.0])
        
        if len(boxes) > 0:
            return [torch.tensor(boxes, device=self.device, dtype=torch.float32)]
        return [None]


class CIAGANTarget(BaseDeidentificationTarget):
    def __init__(self, models_dir=None, device="cuda"):
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        if models_dir is not None:
            global ciagan_models_dir
            ciagan_models_dir = Path(models_dir)

        self.model = self._load_model()

        self.dlib_detector = dlib.get_frontal_face_detector()
        predictor_path = ciagan_source_dir / "shape_predictor_68_face_landmarks.dat"
        if not predictor_path.exists():
            fallback = Path(
                "/kaggle/input/datasets/domenicovicenti/ciagan-repository/"
                "ciagan/source/shape_predictor_68_face_landmarks.dat"
            )
            if fallback.exists():
                predictor_path = fallback

        if not predictor_path.exists():
            raise FileNotFoundError(
                "Predictor dlib non trovato: "
                f"{predictor_path}. Verifica la presenza di shape_predictor_68_face_landmarks.dat."
            )
        self.predictor = dlib.shape_predictor(str(predictor_path))

        # Esposizione delle interfacce per BenchmarkRunner
        self.detector_wrapper = DlibDetectorWrapper(self.dlib_detector, self.device)
        self.pipeline = self
        self.anonymizer = self

    def _load_model(self):
        from arch.arch_unet_flex import Generator

        model = Generator()
        checkpoint_path = ciagan_models_dir / "modelG_ciagan.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint CIAGAN non trovato: {checkpoint_path}")

        print(f"[CIAGAN] Caricamento pesi: {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)

        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "generator", "G"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

        if not isinstance(checkpoint, dict):
            raise TypeError("Il checkpoint CIAGAN non contiene un state_dict valido.")

        state = {}
        for key, value in checkpoint.items():
            clean_key = key[7:] if key.startswith("module.") else key
            state[clean_key] = value

        expected = model.state_dict()
        converted = {}
        derived_v = []
        ignored_derived_weight = []

        for key, value in state.items():
            if key.endswith(".weight") and key[:-len(".weight")] + ".weight_orig" in state:
                ignored_derived_weight.append(key)
                continue
            if key in expected and hasattr(value, "shape") and expected[key].shape == value.shape:
                converted[key] = value

        for key, weight_orig in state.items():
            if not key.endswith(".weight_orig") or not isinstance(weight_orig, torch.Tensor):
                continue
            prefix = key[:-len("weight_orig")]
            u_key = prefix + "weight_u"
            v_key = prefix + "weight_v"
            if v_key not in expected or v_key in converted:
                continue
            if u_key not in state:
                continue

            u = state[u_key].detach().float().reshape(-1)
            w = weight_orig.detach().float().reshape(weight_orig.shape[0], -1)
            if u.numel() != w.shape[0] or expected[v_key].numel() != w.shape[1]:
                continue

            u = torch.nn.functional.normalize(u, dim=0, eps=1e-12)
            v = torch.nn.functional.normalize(torch.mv(w.t(), u), dim=0, eps=1e-12)
            for _ in range(5):
                u = torch.nn.functional.normalize(torch.mv(w, v), dim=0, eps=1e-12)
                v = torch.nn.functional.normalize(torch.mv(w.t(), u), dim=0, eps=1e-12)

            converted[v_key] = v.to(dtype=expected[v_key].dtype).reshape(expected[v_key].shape)
            derived_v.append(v_key)

        missing_expected = [k for k in expected if k not in converted]
        shape_mismatches = [
            (k, tuple(state[k].shape), tuple(expected[k].shape))
            for k in state
            if k in expected and hasattr(state[k], "shape") and state[k].shape != expected[k].shape
        ]
        if missing_expected or shape_mismatches:
            raise RuntimeError(
                "Checkpoint CIAGAN non convertibile in modo sicuro. "
                f"Pesi mancanti dopo conversione: {len(missing_expected)}; "
                f"dimensioni incompatibili: {len(shape_mismatches)}."
            )

        missing, unexpected = model.load_state_dict(converted, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Caricamento CIAGAN incompleto: missing={missing}, unexpected={unexpected}"
            )

        model.to(self.device)
        model.eval()
        return model

    def __call__(self, img_tensor, truncation_value=1.0, multi_modal_truncation=False, amp=False, **kwargs):
        """Interfaccia callable compatibile con il BenchmarkRunner."""
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor.float() / 255.0
        else:
            img_tensor = img_tensor.float()
        output_tensor = self.process_image(img_tensor)
        return {"im": (output_tensor * 255.0).byte()}

    @staticmethod
    def _to_rgb_float(image_tensor: torch.Tensor) -> torch.Tensor:
        t = image_tensor.detach().float()
        if t.ndim == 4:
            if t.shape[0] != 1:
                raise ValueError("CIAGANTarget.process_image accetta una singola immagine per chiamata.")
            t = t[0]
        if t.ndim != 3 or t.shape[0] < 3:
            raise ValueError(f"Formato immagine non supportato: shape={tuple(t.shape)}; atteso [3,H,W].")

        t = t[:3]
        t_min = float(t.min().item())
        t_max = float(t.max().item())
        if t_min < -0.01 and t_min >= -1.01 and t_max <= 1.01:
            t = (t + 1.0) / 2.0
        elif t_max > 1.5:
            t = t / 255.0
        return t.clamp(0.0, 1.0)

    @staticmethod
    def _official_preprocess(img_178x218: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(img_178x218.astype(np.uint8), mode="RGB")
        pil_img = pil_img.resize(
            (PRE_RESIZE, PRE_RESIZE),
            resample=Image.Resampling.LANCZOS,
        )
        pil_img = pil_img.crop(
            (CENTER_OFFSET, CENTER_OFFSET,
             CENTER_OFFSET + MODEL_SIZE, CENTER_OFFSET + MODEL_SIZE)
        )
        return np.asarray(pil_img, dtype=np.uint8)

    @staticmethod
    def _landmark_point(shape, index, crop_left, crop_top, crop_width, crop_height):
        x = (shape.part(index).x - crop_left) * (CIAGAN_W / float(crop_width))
        y = (shape.part(index).y - crop_top) * (CIAGAN_H / float(crop_height))
        return (
            int(np.clip(round(x), 0, CIAGAN_W - 1)),
            int(np.clip(round(y), 0, CIAGAN_H - 1)),
        )

    def _make_conditioning(self, shape, crop_left, crop_top, crop_width, crop_height):
        lndm_rgb = np.full((CIAGAN_H, CIAGAN_W, 3), 255, dtype=np.uint8)

        def pt(i):
            return self._landmark_point(
                shape, i, crop_left, crop_top, crop_width, crop_height
            )

        for i in range(16):
            cv2.line(lndm_rgb, pt(i), pt(i + 1), (0, 0, 255), thickness=1)
        for i in range(27, 30):
            cv2.line(lndm_rgb, pt(i), pt(i + 1), (0, 0, 255), thickness=1)
        for i in range(60, 63):
            cv2.line(lndm_rgb, pt(i), pt(i + 1), (0, 0, 255), thickness=1)
        for i in range(64, 67):
            cv2.line(lndm_rgb, pt(i), pt(i + 1), (0, 0, 255), thickness=1)
        cv2.line(lndm_rgb, pt(67), pt(60), (0, 0, 255), thickness=1)

        lndm_model = self._official_preprocess(lndm_rgb)

        mask_rgb = np.full((CIAGAN_H, CIAGAN_W, 3), 255, dtype=np.uint8)
        contour = [
            (pt(0)[0], pt(19)[1]),
            *[pt(i) for i in range(17)],
            (pt(16)[0], pt(24)[1]),
        ]
        cv2.fillPoly(mask_rgb, [np.asarray(contour, dtype=np.int32)], (0, 0, 0))
        mask_model_rgb = self._official_preprocess(mask_rgb)
        mask_gray = mask_model_rgb.astype(np.float32).mean(axis=2) / 255.0
        face_mask = (1.0 - mask_gray) > 0.2

        return lndm_model, face_mask

    def process_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img_float = self._to_rgb_float(image_tensor)
        img_rgb = (
            img_float.permute(1, 2, 0).cpu().numpy() * 255.0
        ).round().clip(0, 255).astype(np.uint8)
        orig_h, orig_w = img_rgb.shape[:2]

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        detections = self.dlib_detector(gray, 1)
        if not detections:
            print("[CIAGAN][WARN] Nessun volto rilevato: restituisco l'immagine originale.")
            return img_float

        det = max(detections, key=lambda r: max(0, r.width()) * max(0, r.height()))
        shape = self.predictor(gray, det)

        eye_dist = abs(shape.part(42).x - shape.part(39).x)
        if eye_dist < 2:
            print("[CIAGAN][WARN] Landmark oculari non validi: restituisco l'immagine originale.")
            return img_float

        center_x = int((shape.part(42).x + shape.part(39).x) / 2)
        center_y = int((shape.part(42).y + shape.part(39).y) / 2)
        half_h = max(2, int(eye_dist * 5))
        half_w = max(2, int(half_h / CIAGAN_H * CIAGAN_W))

        crop_left, crop_right = center_x - half_w, center_x + half_w
        crop_top, crop_bottom = center_y - half_h, center_y + half_h
        crop_width, crop_height = crop_right - crop_left, crop_bottom - crop_top

        pad = max(crop_width, crop_height) + 4
        img_pad = cv2.copyMakeBorder(
            img_rgb, pad, pad, pad, pad, borderType=cv2.BORDER_REPLICATE
        )
        px1, px2 = crop_left + pad, crop_right + pad
        py1, py2 = crop_top + pad, crop_bottom + pad
        raw_crop = img_pad[py1:py2, px1:px2]
        if raw_crop.size == 0:
            print("[CIAGAN][WARN] Crop del volto vuoto: restituisco l'immagine originale.")
            return img_float

        crop_178x218 = cv2.resize(
            raw_crop, (CIAGAN_W, CIAGAN_H), interpolation=cv2.INTER_CUBIC
        )
        
        lab_in = cv2.cvtColor(crop_178x218, cv2.COLOR_RGB2LAB)
        l_in, a_in, b_in = cv2.split(lab_in)
        clahe_in = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_in = clahe_in.apply(l_in)
        crop_178x218 = cv2.cvtColor(cv2.merge((l_in, a_in, b_in)), cv2.COLOR_LAB2RGB)

        face_model_rgb = self._official_preprocess(crop_178x218)
        lndm_model_rgb, mask_np = self._make_conditioning(
            shape, crop_left, crop_top, crop_width, crop_height
        )

        face_tensor = (
            torch.from_numpy(face_model_rgb.copy()).permute(2, 0, 1).float()
            .unsqueeze(0).to(self.device) / 255.0
        )
        lndm_tensor = (
            torch.from_numpy(lndm_model_rgb.copy()).permute(2, 0, 1).float()
            .unsqueeze(0).to(self.device) / 255.0
        )
        mask_tensor = (
            torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            .repeat(1, 3, 1, 1).to(self.device)
        )

        output_id = 750
        labels = torch.zeros((1, NUM_IDENTITIES), dtype=torch.float32, device=self.device)
        labels[0, output_id] = 1.0

        with torch.inference_mode():
            model_input = torch.cat(
                (lndm_tensor, face_tensor * (1.0 - mask_tensor)), dim=1
            )
            generated = self.model(model_input, onehot=labels)
            
            output = torch.clamp(generated, 0.0, 1.0)

        output_rgb = (
            output[0].permute(1, 2, 0).cpu().numpy() * 255.0
        ).round().clip(0, 255).astype(np.uint8)

        margin_x = int(round(crop_width * CENTER_OFFSET / PRE_RESIZE))
        margin_y = int(round(crop_height * CENTER_OFFSET / PRE_RESIZE))
        ix1, ix2 = px1 + margin_x, px2 - margin_x
        iy1, iy2 = py1 + margin_y, py2 - margin_y

        ix1 = max(0, min(ix1, img_pad.shape[1] - 1))
        ix2 = max(ix1 + 1, min(ix2, img_pad.shape[1]))
        iy1 = max(0, min(iy1, img_pad.shape[0] - 1))
        iy2 = max(iy2 + 1, min(iy2, img_pad.shape[0]))
        region_w, region_h = ix2 - ix1, iy2 - iy1

        generated_high = cv2.resize(
            output_rgb, (region_w, region_h), interpolation=cv2.INTER_CUBIC
        ).astype(np.float32)

        original_region = img_pad[iy1:iy2, ix1:ix2].astype(np.float32)

        hull_mask = np.zeros((MODEL_SIZE, MODEL_SIZE), dtype=np.float32)
        pts_hull = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float32)
        pts_hull[:, 0] = (pts_hull[:, 0] + pad - ix1) * (MODEL_SIZE / float(region_w))
        pts_hull[:, 1] = (pts_hull[:, 1] + pad - iy1) * (MODEL_SIZE / float(region_h))
        hull = cv2.convexHull(pts_hull.astype(np.int32))
        cv2.fillConvexPoly(hull_mask, hull, 1.0)
        
        hull_mask = cv2.GaussianBlur(hull_mask, (21, 21), 0)
        alpha_high = cv2.resize(hull_mask, (region_w, region_h), interpolation=cv2.INTER_LINEAR)
        
        mask_3ch = np.clip(alpha_high, 0.0, 1.0)[..., None]
        
        gen_mean = (generated_high * mask_3ch).sum(axis=(0, 1)) / (mask_3ch.sum() + 1e-5)
        orig_mean = (original_region * mask_3ch).sum(axis=(0, 1)) / (mask_3ch.sum() + 1e-5)
        
        generated_high = np.clip((generated_high - gen_mean) + orig_mean, 0, 255)

        blended = generated_high * mask_3ch + original_region * (1.0 - mask_3ch)
        img_pad[iy1:iy2, ix1:ix2] = np.clip(blended, 0, 255).round().astype(np.uint8)

        final_rgb = img_pad[pad:pad + orig_h, pad:pad + orig_w].copy()
        return torch.from_numpy(final_rgb).permute(2, 0, 1).float().to(self.device) / 255.0
