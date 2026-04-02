import argparse
import glob
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import FocalLoss, PointNetSegmentation


class Config:
    data_dir = r""
    save_dir = r""

    num_points = 16384

    batch_size = 8
    epochs = 100
    lr = 0.001
    weight_decay = 1e-4

    class_weights = [1.0, 3.0]

    use_focal_loss = True

    seed = 42
    num_workers = 0
    early_stop = 15


def load_pointcloud(filepath: str) -> np.ndarray:
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(filepath)
        if "X" in df.columns:
            cols = ["X", "Y", "Z", "Intensity"]
        elif "x" in df.columns:
            cols = ["x", "y", "z", "intensity"]
        else:
            cols = df.columns[:4].tolist()
        return df[cols].values.astype(np.float32)
    data = np.loadtxt(filepath)
    if data.shape[1] >= 4:
        return data[:, :4].astype(np.float32)
    intensity = np.zeros((len(data), 1), dtype=np.float32)
    return np.hstack([data[:, :3], intensity]).astype(np.float32)


def find_defect_file(sample_dir: str) -> Optional[str]:
    for name in ["defect", "cavity", "crack"]:
        for ext in [".txt", ".csv"]:
            path = os.path.join(sample_dir, name + ext)
            if os.path.exists(path):
                return path
    return None


def find_file(directory: str, base_name: str) -> Optional[str]:
    for ext in [".txt", ".csv"]:
        path = os.path.join(directory, base_name + ext)
        if os.path.exists(path):
            return path
    return None


def load_samples(data_dir: str):
    samples = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue

        sample_dirs = sorted(glob.glob(os.path.join(split_dir, "sample_*")))
        for sample_dir in sample_dirs:
            full_file = find_file(sample_dir, "full")
            defect_file = find_defect_file(sample_dir)
            if full_file and defect_file:
                samples[split].append((full_file, defect_file))

    return samples["train"], samples["val"], samples["test"]


class PointCloudDataset(Dataset):
    def __init__(self, sample_list: list, num_points: int = 16384):
        self.samples = sample_list
        self.num_points = num_points

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        full_file, defect_file = self.samples[idx]

        full_points = load_pointcloud(full_file)
        defect_points = load_pointcloud(defect_file)

        labels = np.zeros(len(full_points), dtype=np.int64)
        if len(defect_points) > 0:
            tree = cKDTree(full_points[:, :3])
            distances, indices = tree.query(defect_points[:, :3], k=1)
            matched = indices[distances < 0.1]
            labels[matched] = 1

        n = len(full_points)
        if n >= self.num_points:
            idx = np.random.choice(n, self.num_points, replace=False)
        else:
            idx = np.random.choice(n, self.num_points, replace=True)

        points = full_points[idx]
        labels = labels[idx]

        points = self._normalize(points)

        return torch.FloatTensor(points), torch.LongTensor(labels)

    def _normalize(self, points):
        coords = points[:, :3].copy()
        intensity = points[:, 3:4].copy()

        centroid = coords.mean(axis=0)
        coords = coords - centroid
        max_dist = np.max(np.linalg.norm(coords, axis=1)) + 1e-6
        coords = coords / max_dist

        intensity_normalized = (intensity + 30) / 30
        intensity_normalized = np.clip(intensity_normalized, 0, 1)

        intensity_flag = (intensity > -4).astype(np.float32)

        return np.hstack([coords, intensity_normalized, intensity_flag])


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    os.makedirs(config.save_dir, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print("loading data...")
    train_samples, val_samples, test_samples = load_samples(config.data_dir)

    if not train_samples or not val_samples:
        print(
            f"insufficient data: train={len(train_samples)}, val={len(val_samples)}"
        )
        return

    print(
        f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
    )

    train_dataset = PointCloudDataset(train_samples, config.num_points)
    val_dataset = PointCloudDataset(val_samples, config.num_points)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = PointNetSegmentation(in_channels=5, num_classes=2).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    if config.use_focal_loss:
        criterion = FocalLoss(alpha=0.5, gamma=2.0)
        print("loss: FocalLoss (alpha=0.5)")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(config.class_weights).to(device)
        )
        print(f"loss: CrossEntropy (weights={config.class_weights})")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    print("training...")
    best_iou = 0
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_acc": []}

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for points, labels in pbar:
            points = points.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs.reshape(-1, 2), labels.reshape(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for points, labels in val_loader:
                points = points.to(device)
                labels = labels.to(device)

                outputs = model(points)
                loss = criterion(outputs.reshape(-1, 2), labels.reshape(-1))
                val_loss += loss.item()

                preds = outputs.argmax(dim=2)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss /= len(val_loader)

        all_preds = np.concatenate(all_preds).flatten()
        all_labels = np.concatenate(all_labels).flatten()

        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()
        tn = ((all_preds == 0) & (all_labels == 0)).sum()

        acc = (tp + tn) / (tp + fp + fn + tn + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(float(iou))
        history["val_acc"].append(float(acc))

        if iou > best_iou:
            best_iou = iou
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "iou": iou,
                    "config": {"num_points": config.num_points, "in_channels": 5},
                },
                os.path.join(config.save_dir, "best_model.pth"),
            )
        else:
            no_improve += 1

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:3d} | Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"IoU: {iou:.4f} | P/R: {precision:.3f}/{recall:.3f} | LR: {lr:.6f}"
        )

        if config.early_stop > 0 and no_improve >= config.early_stop:
            print(f"early stop: no improvement for {config.early_stop} epochs")
            break

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "iou": iou,
            "config": {"num_points": config.num_points, "in_channels": 5},
        },
        os.path.join(config.save_dir, "final_model.pth"),
    )

    with open(os.path.join(config.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"done. best IoU: {best_iou:.4f}")
    print(f"checkpoints: {config.save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-points", type=int, default=16384)
    parser.add_argument(
        "--focal-loss",
        action="store_true",
        default=True,
        help="use Focal Loss (default: on)",
    )
    parser.add_argument(
        "--no-focal-loss",
        dest="focal_loss",
        action="store_false",
        help="use CrossEntropyLoss instead",
    )
    parser.add_argument(
        "--defect-weight",
        type=float,
        default=3.0,
        help="defect class weight (CrossEntropy only)",
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=0,
        help="early stopping patience (0 disables)",
    )
    args = parser.parse_args()

    config = Config()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.num_points = args.num_points
    config.use_focal_loss = args.focal_loss
    config.class_weights = [1.0, args.defect_weight]
    config.early_stop = args.early_stop

    train(config)


if __name__ == "__main__":
    main()
