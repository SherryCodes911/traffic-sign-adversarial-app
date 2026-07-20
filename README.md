# TrafficGuard : Traffic Sign Recognition

A CNN-based traffic sign classifier [GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign), 43 classes that demonstrates how
adversarial attacks can silently break a "clean" image classifier and how
adversarial training defends against it.

Two identical CNNs are trained on the same data: one with standard training,
one with adversarial training. Both are attacked with **FGSM** and **PGD**
at inference time so the difference in robustness is directly visible.

## Why this exists

Traffic sign recognition is a canonical example of a safety-critical
vision system. A model that scores 96%+ accuracy on clean images can still
be fooled by a human-imperceptible pixel perturbation, this project
measures exactly how much, and shows a concrete mitigation (adversarial
training) rather than just stating the vulnerability exists.

## Results

| Scenario                         | Accuracy |
|-----------------------------------|:--------:|
| Clean CNN → clean images          | 96.75%   |
| Clean CNN → FGSM (ε=0.05)         | 45.32%   |
| Clean CNN → PGD (ε=0.05)          | 28.68%   |
| Adversarial CNN → clean images    | 95.49%   |
| Adversarial CNN → FGSM (ε=0.05)   | 82.15%   |
| Adversarial CNN → PGD (ε=0.05)    | 76.87%   |

Adversarial training costs ~1.3 points of clean accuracy but recovers
**+37 points** of robustness under FGSM and **+48 points** under PGD.

## Project structure

```
├── app.py              # FastAPI backend: /predict, /stats, /health + web UI
├── predict.py           # Standalone CLI: classify one image, no server needed
├── requirements.txt
├── class_names.json     # 43 GTSRB class labels
├── model_config.json    # Architecture / preprocessing / attack hyperparameters
├── results.json         # Benchmark numbers used by /stats and the table above
├── models/
│   ├── clean_model.pth        # Standard training
│   └── adversarial_model.pth  # Adversarial (FGSM/PGD) training
├── examples/             # Sample GTSRB images for a quick test drive
└── static/
    └── index.html        # Zero-dependency frontend for app.py
```

## How it works

- **Architecture** : a 3-block CNN (Conv → BatchNorm → ReLU, ×2 per block,
  MaxPool + Dropout between blocks) followed by a fully connected classifier
  head. Trained at 32×32 resolution on GTSRB.
- **FGSM** (Goodfellow et al., 2015), a single-step attack that perturbs
  every pixel by `epsilon` in the direction of the loss gradient's sign.
- **PGD** (Madry et al., 2018), the iterative version of FGSM: multiple
  small steps with projection back into an `epsilon`-ball around the
  original image, a much stronger attack.
- **Adversarial training** : the "robust" model is trained on PGD-perturbed
  examples in addition to clean ones, so it learns a decision boundary that
  isn't as sensitive to small, worst-case perturbations.

## Quick start

```bash
git clone https://github.com/SherryCodes911/traffic-sign-adversarial-app

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Option A: CLI (fastest way to try it)

```bash
python predict.py examples/sample_1.png
python predict.py examples/sample_1.png --attack fgsm --epsilon 0.05
python predict.py examples/sample_1.png --attack pgd --epsilon 0.05 --pgd-steps 10
python predict.py examples/sample_1.png --model adversarial --attack fgsm
```

### Option B: Web UI

```bash
python app.py
# or: uvicorn app:app --reload --port 8000
```

Then open **http://localhost:8000** and upload an image to compare the
clean and adversarial models side by side, with an adjustable attack
strength.

## API reference (`app.py`)

| Endpoint   | Method | Description                                   |
|------------|--------|------------------------------------------------|
| `/health`  | GET    | Server status and which models are loaded.     |
| `/stats`   | GET    | Pre-computed benchmark accuracy (table above).  |
| `/predict` | POST   | Classify an image, optionally under attack.     |

**`POST /predict` form fields**

| Field        | Type   | Default   | Description                        |
|--------------|--------|-----------|-------------------------------------|
| `file`       | File   | required  | Image file (PNG/JPG/WEBP)          |
| `model_type` | string | `clean`   | `clean` or `adversarial`           |
| `attack`     | string | `none`    | `none`, `fgsm`, or `pgd`           |
| `epsilon`    | float  | `0.05`    | Perturbation magnitude (0.001–0.5) |
| `pgd_steps`  | int    | `10`      | PGD iteration count (1–100)        |

```json
{
  "model_type": "clean",
  "attack_applied": "fgsm",
  "epsilon": 0.05,
  "clean":       { "predicted_class": "Stop", "class_id": 14, "confidence": 97.34, "top5": [...] },
  "adversarial": { "predicted_class": "No entry", "class_id": 17, "confidence": 61.2, "top5": [...], "prediction_changed": true }
}
```

## References

- Goodfellow, I. et al. (2015). *Explaining and Harnessing Adversarial Examples.*
- Madry, A. et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks.*
- Stallkamp, J. et al. *The German Traffic Sign Recognition Benchmark (GTSRB).*
