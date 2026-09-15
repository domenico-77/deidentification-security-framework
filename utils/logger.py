# =====================================================================
# UTILS: TERMINAL LOGGER
# =====================================================================

import time

def print_run_header(image_idx, total_images, image_name, epsilon):
    """Stampa l'intestazione per ogni iterazione di immagine/epsilon."""
    print("\n" + "#" * 70)
    print(f"IMAGE {image_idx}/{total_images} | {image_name} | epsilon={epsilon}")
    print("#" * 70)

def print_attack_results( 
    attack_success, 
    success_iteration, 
    metrics, 
    pipeline_mse, 
    elapsed_seconds, 
    current_run_idx, 
    total_expected
):
    """Stampa a terminale i risultati dettagliati del singolo attacco PGD e delle metriche."""
    print(f"ATTACK SUCCESS:    {attack_success}")
    print(f"Success iteration: {success_iteration}")
    print(f"L-inf:             {metrics['linf']:.2f}")
    print(f"L2:                {metrics['l2']:.2f}")
    print(f"MSE:               {metrics['mse']:.6f}")
    print(f"PSNR:              {metrics['psnr']:.2f} dB")
    print(f"Pipeline MSE:      {pipeline_mse:.8f}")
    print(f"Tempo:             {elapsed_seconds:.2f}s")
    print(f"CHECKPOINT:        {current_run_idx}/{total_expected}")
