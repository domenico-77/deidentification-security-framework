import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as T


class LFWDataset(Dataset):
    """
    Dataset LFW con indicizzazione efficiente.

    Supporta:
      root/persona/immagine.jpg
      root/lfw-deepfunneled/persona/immagine.jpg
      root/lfw-deepfunneled/lfw-deepfunneled/persona/immagine.jpg

    Restituisce:
      tensor_img: RGB float32 in [0, 255]
      path: percorso dell'immagine
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir).expanduser()
        self.transform = transform or T.ToTensor()
        self.image_paths = []

        print(
            f"[INFO] Indicizzazione LFW da: {self.root_dir}",
            flush=True,
        )

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"La directory del dataset non esiste: {self.root_dir}"
            )

        if not self.root_dir.is_dir():
            raise NotADirectoryError(
                f"Il percorso non è una directory: {self.root_dir}"
            )

        # Individua la directory che contiene le cartelle delle persone.
        base_path = self.root_dir

        # Gestisce directory LFW eventualmente duplicate.
        while True:
            nested = base_path / "lfw-deepfunneled"
            if nested.is_dir():
                base_path = nested
            else:
                break

        print(f"[INFO] Directory immagini: {base_path}", flush=True)

        # scandir evita parte dell'overhead di Path.iterdir()
        # durante la lettura di directory molto popolate.
        try:
            with os.scandir(base_path) as entries:
                person_dirs = [
                    entry.path
                    for entry in entries
                    if entry.is_dir(follow_symlinks=False)
                ]
        except OSError as exc:
            raise OSError(
                f"Impossibile leggere la directory {base_path}: {exc}"
            ) from exc

        print(
            f"[INFO] Cartelle persona individuate: {len(person_dirs)}",
            flush=True,
        )

        total_dirs = len(person_dirs)

        for i, person_dir in enumerate(person_dirs, start=1):
            try:
                with os.scandir(person_dir) as entries:
                    for entry in entries:
                        if not entry.is_file(follow_symlinks=False):
                            continue

                        if Path(entry.name).suffix.lower() in self.EXTENSIONS:
                            self.image_paths.append(entry.path)

            except OSError as exc:
                print(
                    f"[WARN] Impossibile leggere {person_dir}: {exc}",
                    flush=True,
                )
                continue

            if i % 500 == 0 or i == total_dirs:
                print(
                    f"[INFO] Scansione cartelle: {i}/{total_dirs} | "
                    f"immagini trovate: {len(self.image_paths)}",
                    flush=True,
                )

        # Ordinamento deterministico per rendere riproducibili gli esperimenti.
        self.image_paths.sort()

        print(
            f"[INFO] Indicizzazione completata: "
            f"{len(self.image_paths)} immagini.",
            flush=True,
        )

        if not self.image_paths:
            raise RuntimeError(
                "Nessuna immagine trovata. Controlla root_dir e la struttura "
                "del dataset: sono supportati JPG, JPEG e PNG."
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                tensor_img = self.transform(img)

        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(
                f"Impossibile caricare l'immagine all'indice {idx}: {path}"
            ) from exc

        # ToTensor restituisce normalmente [0, 1].
        # Manteniamo il contratto originale del progetto: [0, 255].
        tensor_img = tensor_img.float() * 255.0

        return tensor_img, path
