# De-identification Security Evaluation Framework

Un framework modulare in PyTorch progettato per la valutazione della robustezza adversarial (PGD, FGSM, BIM, UAP) di pipelines di anonimizzazione e face de-identification.
(Per ora è implementato e testato PGD su DeepPrivacy2)

## Struttura del Progetto
- `attacks/`: Algoritmi di attacco adversarial basati sull'interfaccia comune astratta `BaseAttack` (include implementazioni come `PGDAttack`).
- `targets/`: Modelli target e pipeline di anonimizzazione (es. integrazione strutturata con `DeepPrivacy2`).
- `detectors/`: Rilevatori di volti interni (es. `DSFDDetector`) sfruttati come guida per la backpropagation e il calcolo dei gradienti.
- `benchmark/`: Runner centralizzato per l'orchestrazione degli epsilon-sweep, il calcolo delle metriche di pentesting e il logging in tempo reale (`BenchmarkRunner`).
- `data_loaders/`: Gestori e pipeline per i dataset di test (es. caricamento e gestione standardizzata del dataset LFW).
- `configs/`: File di configurazione in formato YAML per la gestione dei parametri di attacco e dei percorsi dei modelli.
- `experiments/`: Script di orchestrazione dedicati per l'avvio dei singoli attacchi (es. `run_pgd.py`) e script di post-analisi per la generazione automatica dei grafici (`plot_results.py`).
- `kaggle/`: Script di setup e configurazione dell'ambiente per l'esecuzione e il deployment su piattaforme cloud Kaggle.

---

## Guida all'Avvio su Kaggle

Per replicare l'esperimento, eseguire il benchmark PGD su scala, generare i grafici di analisi e archiviare la baseline, puoi copiare ed eseguire la seguente sequenza di comandi all'interno delle celle di un notebook Kaggle:

```bash
%cd /kaggle/working
!rm -rf deidentification-security-framework
!git clone [https://github.com/domenico-77/deidentification-security-framework.git](https://github.com/domenico-77/deidentification-security-framework.git)
%cd deidentification-security-framework
!python kaggle/setup.py
!PYTHONPATH=. python experiments/run_pgd.py --dataset_path /kaggle/input/datasets/jessicali9530/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled

!python experiments/plot_results.py
# Sposta la cartella attuale rinominandola come baseline PGD
!mv results results_baseline_pgd
# Crea una nuova cartella results vuota per i prossimi test
!mkdir results
