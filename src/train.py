"""Calibration and training pipeline for the Dual-Stream Deepfake Detector."""

import argparse
import os
import sys
import time
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Ensure root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DetectorConfig
from src.models.dual_stream_detector import DualStreamDetector
from src.data.synthetic_generator import SyntheticForensicsDataset


def train_calibrated_model(
    num_samples: int = 200,
    epochs: int = 6,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    save_path: str = "checkpoints/calibrated_weights.pt",
    device: str = "auto",
) -> Tuple[DualStreamDetector, Dict[str, float]]:
    """Train/calibrate the dual-stream detector on synthetic forensic datasets."""
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    print(f"Preparing dataset ({num_samples} samples) on {dev}...")
    dataset = SyntheticForensicsDataset(num_samples=num_samples, size=128, seed=42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    cfg = DetectorConfig()
    cfg.spectral.image_size = 128
    model = DualStreamDetector(cfg).to(dev)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Starting calibration training for {epochs} epochs...")
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in train_loader:
            images = images.to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad()
            out = model(images)
            
            # Multi-task supervision
            loss_fused = criterion(out["logits"], labels)
            loss_sp = criterion(out["spatial_logits"], labels)
            loss_fr = criterion(out["freq_logits"], labels)
            loss = loss_fused + 0.3 * loss_sp + 0.3 * loss_fr

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(out["logits"], dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler.step()
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(dev)
                labels = labels.to(dev)
                out = model(images)
                preds = torch.argmax(out["logits"], dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / max(1, val_total)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss / total:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    print(f"Calibration complete! Best Val Acc: {best_val_acc*100:.1f}%. Saved to: {save_path}")
    return model, {"best_val_acc": best_val_acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    train_calibrated_model(epochs=args.epochs, num_samples=args.samples, device=args.device)
