import argparse
import glob
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from tqdm import tqdm

from model import PointNetSegmentation


MODEL_PATH = r""
TEST_DIR = r""
OUTPUT_DIR = r""


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


def predict_pointcloud(
    model,
    pointcloud,
    device,
    num_points=16384,
    threshold=0.6,
    in_channels=5,
):
    n = len(pointcloud)
    scores = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)

    stride = num_points // 4

    for start in range(0, n, stride):
        end = min(start + num_points, n)

        if end - start >= num_points:
            idx = np.arange(start, end)
        elif end - start < 512:
            continue
        else:
            idx = np.random.choice(range(start, end), num_points, replace=True)

        window = pointcloud[idx].copy()

        window_coords = window[:, :3].copy()
        centroid = window_coords.mean(axis=0)
        window_coords = window_coords - centroid
        max_dist = np.max(np.linalg.norm(window_coords, axis=1)) + 1e-6
        window_coords = window_coords / max_dist

        window_intensity = window[:, 3:4]

        if in_channels == 5:
            intensity_norm = (window_intensity + 30) / 30
            intensity_norm = np.clip(intensity_norm, 0, 1)
            intensity_flag = (window_intensity > -4).astype(np.float32)
            features = np.hstack([window_coords, intensity_norm, intensity_flag])
        else:
            window_intensity = (window_intensity - window_intensity.mean()) / (
                window_intensity.std() + 1e-6
            )
            window_intensity = np.clip(window_intensity, -3, 3)
            features = np.hstack([window_coords, window_intensity])

        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(device)
            output = model(x)
            probs = F.softmax(output, dim=2)[0, :, 1].cpu().numpy()

        valid_len = min(len(idx), end - start)
        scores[idx[:valid_len]] += probs[:valid_len]
        counts[idx[:valid_len]] += 1

    mask = counts > 0
    scores[mask] /= counts[mask]
    labels = (scores > threshold).astype(np.int32)

    return labels, scores


def save_prediction(output_dir, sample_name, points, pred, labels):
    sample_dir = os.path.join(output_dir, sample_name)
    os.makedirs(sample_dir, exist_ok=True)

    result = np.column_stack([points[:, :4], pred, labels])
    np.savetxt(
        os.path.join(sample_dir, "segmented.txt"),
        result,
        fmt="%.6f %.6f %.6f %.6f %d %d",
        header="X Y Z Intensity Pred GT",
    )

    defect_mask = pred == 1
    if defect_mask.sum() > 0:
        defect_points = points[defect_mask]
        np.savetxt(
            os.path.join(sample_dir, "pred_defect.txt"),
            defect_points[:, :4],
            fmt="%.6f %.6f %.6f %.6f",
        )

    with open(os.path.join(sample_dir, "visualization.txt"), "w") as f:
        for j in range(len(points)):
            x, y, z = points[j, :3]
            intensity = points[j, 3] if points.shape[1] > 3 else 0
            p, l = pred[j], labels[j]

            if p == 1 and l == 1:
                r, g, b = 0, 255, 0
            elif p == 1 and l == 0:
                r, g, b = 255, 0, 0
            elif p == 0 and l == 1:
                r, g, b = 255, 255, 0
            else:
                r, g, b = 100, 100, 255

            f.write(f"{x:.6f} {y:.6f} {z:.6f} {intensity:.6f} {r} {g} {b}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=MODEL_PATH)
    parser.add_argument("--test-dir", "-t", type=str, default=TEST_DIR)
    parser.add_argument("--output", "-o", type=str, default=OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    print("IAPCNet evaluation")

    if not os.path.exists(args.test_dir):
        print(f"test directory not found: {args.test_dir}")
        return

    test_samples = sorted(glob.glob(os.path.join(args.test_dir, "sample_*")))
    test_pairs = []
    for sample_dir in test_samples:
        full_file = find_file(sample_dir, "full")
        defect_file = find_defect_file(sample_dir)
        if full_file and defect_file:
            test_pairs.append((full_file, defect_file))

    if not test_pairs:
        print("no test samples with full + defect")
        return

    print(f"samples: {len(test_pairs)}")
    print(f"threshold: {args.threshold}")

    if not os.path.exists(args.model):
        print(f"checkpoint not found: {args.model}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    num_points = config.get("num_points", 16384)
    in_channels = config.get("in_channels", 4)

    model = PointNetSegmentation(in_channels=in_channels, num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  in_channels: {in_channels}")
    print(f"  num_points: {num_points}")
    print(f"  checkpoint IoU: {checkpoint.get('iou', 'n/a')}")

    os.makedirs(args.output, exist_ok=True)

    all_metrics = []
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    print("running...")
    for full_file, defect_file in tqdm(test_pairs, desc="eval"):
        sample_name = os.path.basename(os.path.dirname(full_file))

        full_points = load_pointcloud(full_file)
        defect_points = load_pointcloud(defect_file)

        gt_labels = np.zeros(len(full_points), dtype=np.int32)
        if len(defect_points) > 0:
            tree = cKDTree(full_points[:, :3])
            distances, indices = tree.query(defect_points[:, :3], k=1)
            matched = indices[distances < 0.1]
            gt_labels[matched] = 1

        pred_labels, _ = predict_pointcloud(
            model,
            full_points,
            device,
            num_points,
            args.threshold,
            in_channels,
        )

        tp = ((pred_labels == 1) & (gt_labels == 1)).sum()
        fp = ((pred_labels == 1) & (gt_labels == 0)).sum()
        fn = ((pred_labels == 0) & (gt_labels == 1)).sum()
        tn = ((pred_labels == 0) & (gt_labels == 0)).sum()

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

        metrics = {
            "sample": sample_name,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
        all_metrics.append(metrics)

        save_prediction(args.output, sample_name, full_points, pred_labels, gt_labels)

        print(
            f"  {sample_name}: IoU={iou:.4f}, F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}"
        )

    total_accuracy = (total_tp + total_tn) / (
        total_tp + total_fp + total_fn + total_tn
    )
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = (
        2 * total_precision * total_recall / (total_precision + total_recall)
        if (total_precision + total_recall) > 0
        else 0
    )
    total_iou = (
        total_tp / (total_tp + total_fp + total_fn)
        if (total_tp + total_fp + total_fn) > 0
        else 0
    )

    overall = {
        "accuracy": float(total_accuracy),
        "precision": float(total_precision),
        "recall": float(total_recall),
        "f1": float(total_f1),
        "iou": float(total_iou),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "tn": int(total_tn),
    }

    print("overall")
    print(f"  accuracy:  {overall['accuracy']:.4f} ({overall['accuracy'] * 100:.2f}%)")
    print(
        f"  precision: {overall['precision']:.4f} ({overall['precision'] * 100:.2f}%)"
    )
    print(f"  recall:    {overall['recall']:.4f} ({overall['recall'] * 100:.2f}%)")
    print(f"  f1:        {overall['f1']:.4f} ({overall['f1'] * 100:.2f}%)")
    print(f"  iou:       {overall['iou']:.4f} ({overall['iou'] * 100:.2f}%)")

    print("confusion (totals)")
    print(f"  tp: {overall['tp']:,}")
    print(f"  fp: {overall['fp']:,}")
    print(f"  fn: {overall['fn']:,}")
    print(f"  tn: {overall['tn']:,}")

    ious = [m["iou"] for m in all_metrics]
    f1s = [m["f1"] for m in all_metrics]
    precisions = [m["precision"] for m in all_metrics]
    recalls = [m["recall"] for m in all_metrics]
    accs = [m["accuracy"] for m in all_metrics]

    print("per-sample mean ± std")
    print(f"  accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"  recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"  f1:        {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  iou:       {np.mean(ious):.4f} ± {np.std(ious):.4f}")

    results = {"model": "IAPCNet", "overall": overall, "per_sample": all_metrics}

    results_path = os.path.join(args.output, "iapcnet_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"saved: {results_path}")


if __name__ == "__main__":
    main()
