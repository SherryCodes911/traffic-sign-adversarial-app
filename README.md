# Traffic Sign Classifier

Built on GTSRB (43 classes). Demonstrates adversarial attacks (FGSM, PGD) and defense via adversarial training.

---

## Project Structure

```
trafficsign-app/
├── app.py                  ← FastAPI backend (inference + attacks)
├── requirements.txt        ← Python dependencies
├── class_names.json        ← 43 GTSRB class names
├── model_config.json       ← Architecture & preprocessing params
├── results.json            ← Pre-computed benchmark results
├── models/
│   ├── clean_model.pth         ← Standard CNN (96.75% clean acc.)
│   └── adversarial_model.pth   ← Robust CNN (82.15% under FGSM)
└── static/
    └── index.html          ← Frontend UI (zero dependencies)
```

---

## Run Locally

```bash
# 1. Clone / copy this folder
cd trafficsign-app

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start server
python app.py
# OR
uvicorn app:app --reload --port 8000

# 5. Open browser
open http://localhost:8000
```

---

## Deploy — Option A: Render (Recommended, Full-Stack)

> Render runs Python natively. Free tier available.

1. Push this folder to GitHub
2. Go to **https://render.com** → New Web Service
3. Connect your repo
4. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Runtime**: Python 3.11
5. Deploy → copy your `https://xxxx.onrender.com` URL

---

## Deploy — Option B: Railway

```bash
# Install Railway CLI
npm i -g @railway/cli

railway login
railway init
railway up
```

---

## Deploy — Option C: Vercel (Frontend) + Render (Backend)

If you want the frontend on Vercel's CDN:

1. Deploy backend to Render (see Option A)
2. In `static/index.html`, update:
   ```js
   const API_URL = 'https://your-render-url.onrender.com';
   ```
3. Deploy static folder to Vercel:
   ```bash
   npx vercel --prod
   ```

---

## API Reference

### `GET /health`
Returns server status and loaded models.

### `GET /stats`
Returns pre-computed benchmark accuracy results.

### `POST /predict`
Classify an image with optional adversarial attack.

**Form fields:**

| Field       | Type   | Default       | Description                          |
|-------------|--------|---------------|--------------------------------------|
| `file`      | File   | required      | Image file (PNG/JPG/WEBP)            |
| `model_type`| string | `"clean"`     | `"clean"` or `"adversarial"`         |
| `attack`    | string | `"none"`      | `"none"`, `"fgsm"`, or `"pgd"`       |
| `epsilon`   | float  | `0.05`        | Perturbation magnitude (0.001–0.5)   |
| `pgd_steps` | int    | `10`          | PGD iteration count (1–100)          |

**Response:**
```json
{
  "model_type": "clean",
  "attack_applied": "fgsm",
  "epsilon": 0.05,
  "clean": {
    "predicted_class": "Stop",
    "class_id": 14,
    "confidence": 97.34,
    "top5": [...]
  },
  "adversarial": {
    "predicted_class": "No entry",
    "class_id": 17,
    "confidence": 61.2,
    "top5": [...],
    "prediction_changed": true
  }
}
```

---

## Benchmark Results

| Scenario                         | Accuracy |
|----------------------------------|----------|
| Clean CNN → Clean images         | 96.75%   |
| Clean CNN → FGSM (ε=0.05)        | 45.32%   |
| Clean CNN → PGD (ε=0.05)         | 28.68%   |
| Adversarial CNN → Clean images   | 95.49%   |
| Adversarial CNN → FGSM (ε=0.05)  | 82.15%   |
| Adversarial CNN → PGD (ε=0.05)   | 76.87%   |

---

## Why PyTorch ≠ Vercel Serverless

Vercel serverless functions have a 250 MB unzipped limit.
PyTorch CPU wheel alone is ~750 MB, which exceeds this.

**Solutions:**
- ✅ Use Render / Railway / Fly.io for the backend (no size limit)
- ✅ Use Vercel only for the static frontend
- 🔧 (Advanced) Convert models to ONNX + use `onnxruntime` (~12 MB)
