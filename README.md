# De-identification Security Evaluation Framework

Un framework modulare in PyTorch progettato per la valutazione della robustezza adversarial (PGD, FGSM, BIM, UAP) di pipelines di anonimizzazione e face de-identification (es. DeepPrivacy2).

## Struttura del Progetto
- `attacks/`: Implementazioni degli attacchi adversarial.
- `targets/`: Modelli e pipeline di anonimizzazione target.
- `detectors/`: Detector di volti interni usati come guida per la backpropagation.
- `benchmark/`: Runner per epsilon-sweep, benchmark e salvataggio metriche.

## Quick Start (Locale)
```bash
pip install -r requirements.txt
python experiments/run_pgd.py --dataset_path /path/to/lfw