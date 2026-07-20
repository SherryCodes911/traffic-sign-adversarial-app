"""
predict.py — Command-line demo for the Traffic Sign Adversarial Robustness project.

Runs a single image through the clean CNN and the adversarially-trained CNN,
optionally attacking it with FGSM or PGD first, and prints a side-by-side
comparison. No server required.

Usage
-----
    python predict.py examples/sample_1.png
    python predict.py examples/sample_1.png --attack fgsm --epsilon 0.05
    python predict.py examples/sample_1.png --attack pgd --epsilon 0.05 --pgd-steps 10
    python predict.py examples/sample_1.png --model adversarial --attack fgsm
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DEVICE = torch.device("cpu")

with open(BASE_DIR / "model_config.json") as f:
    CFG = json.load(f)

IMG_SIZE = CFG["img_resize"]
MEAN = CFG["normalize_mean"]
STD = CFG["normalize_std"]
PIXEL_MIN = CFG["pixel_min"]
PIXEL_MAX = CFG["pixel_max"]
NUM_CLASSES = CFG["num_classes"]

with open(BASE_DIR / "class_names.json") as f:
    CLASS_NAMES = json.load(f)


class TrafficSignCNN(nn.Module):
    """3-block CNN for GTSRB traffic sign classification (43 classes)."""

    def __init__(self, num_classes: int = 43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
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


def load_model(model_type: str) -> TrafficSignCNN:
    path = MODELS_DIR / f"{model_type}_model.pth"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = TrafficSignCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def image_to_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return _transform(img).unsqueeze(0).to(DEVICE)


def fgsm_attack(model: TrafficSignCNN, x: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Fast Gradient Sign Method (Goodfellow et al., 2015)."""
    x_adv = x.clone().detach().requires_grad_(True)
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x_adv), model(x).argmax(1))
    loss.backward()
    return torch.clamp(x_adv.data + epsilon * x_adv.grad.data.sign(), PIXEL_MIN, PIXEL_MAX).detach()


def pgd_attack(model: TrafficSignCNN, x: torch.Tensor, epsilon: float, steps: int = 10) -> torch.Tensor:
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
        delta = torch.clamp(perturbed - x, -epsilon, epsilon)
        perturbed = torch.clamp(x + delta, PIXEL_MIN, PIXEL_MAX).detach()

    return perturbed


def predict(model: TrafficSignCNN, tensor: torch.Tensor, top_k: int = 3) -> dict:
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top_v, top_i = probs.topk(top_k)
    return {
        "predicted_class": CLASS_NAMES[top_i[0].item()],
        "class_id": int(top_i[0]),
        "confidence": round(float(top_v[0]) * 100, 2),
        "top_k": [
            {"class": CLASS_NAMES[i.item()], "confidence": round(float(p) * 100, 2)}
            for i, p in zip(top_i, top_v)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Traffic sign classification with adversarial robustness demo.")
    parser.add_argument("image", type=str, help="Path to a traffic sign image (PNG/JPG).")
    parser.add_argument("--model", choices=["clean", "adversarial"], default="clean",
                         help="Which model to run inference with (default: clean).")
    parser.add_argument("--attack", choices=["none", "fgsm", "pgd"], default="none",
                         help="Adversarial attack to apply before classifying (default: none).")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Perturbation magnitude (default: 0.05).")
    parser.add_argument("--pgd-steps", type=int, default=10, help="Number of PGD iterations (default: 10).")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    model = load_model(args.model)
    tensor = image_to_tensor(image_path)

    clean_result = predict(model, tensor)
    print(f"\nModel: {args.model}")
    print(f"Image: {image_path.name}")
    print("\n[Clean prediction]")
    print(f"  {clean_result['predicted_class']}  ({clean_result['confidence']}%)")

    if args.attack != "none":
        if args.attack == "fgsm":
            adv_tensor = fgsm_attack(model, tensor, args.epsilon)
        else:
            adv_tensor = pgd_attack(model, tensor, args.epsilon, args.pgd_steps)

        adv_result = predict(model, adv_tensor)
        flipped = adv_result["class_id"] != clean_result["class_id"]

        print(f"\n[{args.attack.upper()} attack, epsilon={args.epsilon}]")
        print(f"  {adv_result['predicted_class']}  ({adv_result['confidence']}%)")
        print(f"  Prediction flipped: {'YES' if flipped else 'no'}")
    print()


if __name__ == "__main__":
    main()
