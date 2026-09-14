# =====================================================================
# SCRIPTS: POST-BENCHMARK VISUALIZATION
# =====================================================================

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

def select_target_image(csv_path):
    """Seleziona un'immagine ideale in cui l'attacco ha avuto successo con un epsilon basso."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File CSV non trovato: {csv_path}")
        
    df = pd.read_csv(csv_path)
    successful = df[df["attack_success"] == True]
    
    if successful.empty:
        print("[WARNING] Nessun attacco riuscito trovato nel CSV. Scelgo la prima immagine disponibile.")
        return df.iloc[0]["image_path"], df.iloc[0]["epsilon"]
    
    # Seleziona la run di successo con l'epsilon minimo (perturbazione minima)
    best_run = successful.sort_values(by="epsilon").iloc[0]
    return best_run["image_path"], best_run["epsilon"]

def plot_defense_comparison(img_orig_path, pipeline_output_orig, pipeline_output_adv, save_path=None):
    """
    Mostra e opzionalmente salva il confronto visivo a 3 pannelli:
    1. Immagine Originale
    2. Output DeepPrivacy2 su Immagine Originale (Baseline protetta)
    3. Output DeepPrivacy2 su Immagine Adversarial (Evasione / Fallimento protezione)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Immagine Originale
    orig_img = cv2.imread(img_orig_path)
    if orig_img is not None:
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(orig_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # 2. Baseline DeepPrivacy2
    if pipeline_output_orig is not None:
        axes[1].imshow(cv2.cvtColor(pipeline_output_orig, cv2.COLOR_BGR2RGB))
    axes[1].set_title("DeepPrivacy2 (Baseline)")
    axes[1].axis("off")
    
    # 3. Adversarial DeepPrivacy2
    if pipeline_output_adv is not None:
        axes[2].imshow(cv2.cvtColor(pipeline_output_adv, cv2.COLOR_BGR2RGB))
    axes[2].set_title("DeepPrivacy2 (Adversarial PGD)")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Grafico di confronto salvato in: {save_path}")
        
    plt.show()
    plt.close()
