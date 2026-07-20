"""
Traffic Sign Classifier — FastAPI Backend
==========================================
Adversarial Robustness Demo: Clean CNN vs Adversarially Trained CNN
Supports FGSM and PGD attacks.

Run locally:
    uvicorn app:app --reload --port 8000
    (or: python app.py)

    Then open http://localhost:8000
"""

import io
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# ── Constants from model_config.json ──────────────────────────────────────────
with open(BASE_DIR / "model_config.json") as f:
    CFG = json.load(f)

IMG_SIZE  = CFG["img_resize"]          # 32
MEAN      = CFG["normalize_mean"]
STD       = CFG["normalize_std"]
PIXEL_MIN = CFG["pixel_min"]           # -2.5
PIXEL_MAX = CFG["pixel_max"]           #  2.5
NUM_CLS   = CFG["num_classes"]         # 43

with open(BASE_DIR / "class_names.json") as f:
    CLASS_NAMES: list[str] = json.load(f)

DEVICE = torch.device("cpu")           # CPU for serverless environments


# ── Model Architecture  ────────────────────────────────────────────────────────
class TrafficSignCNN(nn.Module):
    """
    Custom 3-block CNN for GTSRB traffic sign classification.
    Architecture mirrors exactly what was trained in ANN_Project.ipynb.
    """

    def __init__(self, num_classes: int = 43):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Model Cache (load-once, serve-many) ────────────────────────────────────────
_model_cache: dict[str, TrafficSignCNN] = {}


def get_model(model_type: str) -> TrafficSignCNN:
    """Load and cache model weights. Thread-safe for single-worker deployments."""
    if model_type not in _model_cache:
        path = MODELS_DIR / f"{model_type}_model.pth"
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}. "
                "Place clean_model.pth and adversarial_model.pth in /models/"
            )
        model = TrafficSignCNN(num_classes=NUM_CLS)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.eval()
        _model_cache[model_type] = model
    return _model_cache[model_type]


# ── Pre-warm models at startup ─────────────────────────────────────────────────
def _prewarm():
    for mt in ("clean", "adversarial"):
        try:
            get_model(mt)
            print(f"[OK] Loaded {mt} model")
        except FileNotFoundError as e:
            print(f"[WARN] {e}")

_prewarm()


# ── Image Preprocessing ────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def image_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    return _transform(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)


# ── Adversarial Attacks ────────────────────────────────────────────────────────
def fgsm_attack(
    model: TrafficSignCNN,
    x: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Fast Gradient Sign Method (Goodfellow et al., 2015)."""
    x_adv = x.clone().detach().requires_grad_(True)
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x_adv), model(x).argmax(1))
    loss.backward()
    return torch.clamp(x_adv.data + epsilon * x_adv.grad.data.sign(), PIXEL_MIN, PIXEL_MAX).detach()


def pgd_attack(
    model: TrafficSignCNN,
    x: torch.Tensor,
    epsilon: float,
    steps: int = 10,
) -> torch.Tensor:
    """Projected Gradient Descent (Madry et al., 2018)."""
    alpha = epsilon * 2.5 / steps
    delta = torch.zeros_like(x).uniform_(-epsilon, epsilon)
    delta = torch.clamp(x + delta, PIXEL_MIN, PIXEL_MAX) - x
    perturbed = (x + delta).detach()

    for _ in range(steps):
        perturbed = perturbed.clone().detach().requires_grad_(True)
        model.zero_grad()
        loss = nn.CrossEntropyLoss()(model(perturbed), model(x).argmax(1))
        loss.backward()
        perturbed = perturbed.data + alpha * perturbed.grad.data.sign()
        delta     = torch.clamp(perturbed - x, -epsilon, epsilon)
        perturbed = torch.clamp(x + delta, PIXEL_MIN, PIXEL_MAX).detach()

    return perturbed


def _infer(model: TrafficSignCNN, tensor: torch.Tensor) -> dict:
    """Run inference and return top-5 results."""
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top5_v, top5_i = probs.topk(5)
    return {
        "predicted_class": CLASS_NAMES[top5_i[0].item()],
        "class_id":        int(top5_i[0]),
        "confidence":      round(float(top5_v[0]) * 100, 2),
        "top5": [
            {"class": CLASS_NAMES[i.item()], "confidence": round(float(p) * 100, 2)}
            for i, p in zip(top5_i, top5_v)
        ],
    }


# ── FastAPI Application ────────────────────────────────────────────────────────
app = FastAPI(
    title="Traffic Sign Classifier API",
    description="Adversarial Robustness Demo — GTSRB 43-class CNN",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Uptime probe for deployment platforms."""
    loaded = list(_model_cache.keys())
    return {"status": "ok", "loaded_models": loaded, "device": str(DEVICE)}


@app.get("/stats")
def stats():
    """Pre-computed benchmark results from training."""
    with open(BASE_DIR / "results.json") as f:
        return json.load(f)


@app.post("/predict")
async def predict(
    file:       UploadFile = File(...),
    model_type: str        = Form("clean"),
    attack:     str        = Form("none"),
    epsilon:    float      = Form(0.05),
    pgd_steps:  int        = Form(10),
):
    """
    Classify a traffic sign image.

    Params
    ------
    file        : uploaded image (any PIL-readable format)
    model_type  : "clean" | "adversarial"
    attack      : "none" | "fgsm" | "pgd"
    epsilon     : perturbation magnitude (0.01 – 0.3)
    pgd_steps   : PGD iteration count (5 – 40)
    """

    # Validate inputs
    if model_type not in ("clean", "adversarial"):
        raise HTTPException(400, detail="model_type must be 'clean' or 'adversarial'")
    if attack not in ("none", "fgsm", "pgd"):
        raise HTTPException(400, detail="attack must be 'none', 'fgsm', or 'pgd'")
    if not (0.001 <= epsilon <= 0.5):
        raise HTTPException(400, detail="epsilon must be between 0.001 and 0.5")
    if not (1 <= pgd_steps <= 100):
        raise HTTPException(400, detail="pgd_steps must be between 1 and 100")

    # Read image
    raw_bytes = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, detail="Could not decode image. Upload a valid PNG/JPG.")

    # Preprocess
    tensor = image_to_tensor(pil_img)

    # Load model
    try:
        model = get_model(model_type)
    except FileNotFoundError as e:
        raise HTTPException(503, detail=str(e))

    # Clean inference
    clean_result = _infer(model, tensor)

    response = {
        "model_type":     model_type,
        "attack_applied": attack if attack != "none" else None,
        "epsilon":        epsilon if attack != "none" else None,
        "clean":          clean_result,
        "adversarial":    None,
    }

    # Adversarial inference
    if attack != "none":
        if attack == "fgsm":
            adv_tensor = fgsm_attack(model, tensor, epsilon)
        else:
            adv_tensor = pgd_attack(model, tensor, epsilon, pgd_steps)

        adv_result = _infer(model, adv_tensor)
        adv_result["prediction_changed"] = (
            adv_result["class_id"] != clean_result["class_id"]
        )
        response["adversarial"] = adv_result

    return JSONResponse(content=response)


# ── Serve frontend (last, so API routes take priority) ─────────────────────────
_static = BASE_DIR / "static"
if _static.exists():
    app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
