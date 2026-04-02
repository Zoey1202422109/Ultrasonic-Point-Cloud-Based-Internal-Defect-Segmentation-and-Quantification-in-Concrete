
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import find_objects, label
from tqdm import tqdm

from model import PointNetSegmentation


INPUT_FILE = r""
MODEL_PATH = r""
OUTPUT_DIR = r""
THRESHOLD = 0.6
MIN_DEFECT_POINTS = 10


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


def predict_pointcloud(
    model, pointcloud, device, num_points=16384, threshold=0.6, in_channels=5
):
    n = len(pointcloud)
    scores = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)

    stride = num_points // 4

    for start in tqdm(range(0, n, stride), desc="infer"):
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


def analyze_defects(defect_points, min_points=MIN_DEFECT_POINTS):
    """Estimate defect size; coordinates assumed in cm."""
    if len(defect_points) < min_points:
        return {"num_defects": 0, "defects": [], "total_volume": 0}

    coords = defect_points[:, :3]

    unique_x = np.sort(np.unique(np.round(coords[:, 0], 4)))
    unique_y = np.sort(np.unique(np.round(coords[:, 1], 4)))
    unique_z = np.sort(np.unique(np.round(coords[:, 2], 4)))

    spacing_x = np.diff(unique_x).mean() if len(unique_x) > 1 else 0.31
    spacing_y = np.diff(unique_y).mean() if len(unique_y) > 1 else 0.31
    spacing_z = np.diff(unique_z).mean() if len(unique_z) > 1 else 0.31
    voxel_vol = spacing_x * spacing_y * spacing_z

    nx, ny, nz = len(unique_x), len(unique_y), len(unique_z)

    if nx * ny * nz == 0 or nx * ny * nz > 5000000:
        min_c = coords.min(axis=0)
        max_c = coords.max(axis=0)
        size = max_c - min_c
        return {
            "num_defects": 1,
            "defects": [
                {
                    "id": 1,
                    "points": len(defect_points),
                    "size_cm": {
                        "x": round(size[0], 2),
                        "y": round(size[1], 2),
                        "z": round(size[2], 2),
                    },
                    "volume_cm3": round(len(defect_points) * voxel_vol, 2),
                }
            ],
            "total_volume": round(len(defect_points) * voxel_vol, 2),
        }

    grid = np.zeros((nx, ny, nz), dtype=np.int32)
    for x, y, z in coords:
        ix = np.argmin(np.abs(unique_x - x))
        iy = np.argmin(np.abs(unique_y - y))
        iz = np.argmin(np.abs(unique_z - z))
        grid[ix, iy, iz] = 1

    labeled, num = label(grid)

    defects = []
    for i in range(1, num + 1):
        region = labeled == i
        pts = region.sum()
        if pts < min_points:
            continue

        slices = find_objects(labeled)[i - 1]

        x_start, x_end = unique_x[slices[0].start], unique_x[min(slices[0].stop - 1, nx - 1)]
        y_start, y_end = unique_y[slices[1].start], unique_y[min(slices[1].stop - 1, ny - 1)]
        z_start, z_end = unique_z[slices[2].start], unique_z[min(slices[2].stop - 1, nz - 1)]

        size_x = x_end - x_start
        size_y = y_end - y_start
        size_z = z_end - z_start

        volume = pts * voxel_vol

        cx = (x_start + x_end) / 2
        cy = (y_start + y_end) / 2
        cz = (z_start + z_end) / 2

        defects.append(
            {
                "id": len(defects) + 1,
                "points": int(pts),
                "size_cm": {
                    "x": round(size_x, 2),
                    "y": round(size_y, 2),
                    "z": round(size_z, 2),
                },
                "volume_cm3": round(volume, 2),
                "center_cm": {"x": round(cx, 2), "y": round(cy, 2), "z": round(cz, 2)},
            }
        )

    defects.sort(key=lambda x: x["volume_cm3"], reverse=True)
    for i, d in enumerate(defects):
        d["id"] = i + 1

    return {
        "num_defects": len(defects),
        "defects": defects,
        "total_volume": round(sum(d["volume_cm3"] for d in defects), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default=None)
    parser.add_argument("--model", "-m", type=str, default=MODEL_PATH)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--threshold", "-t", type=float, default=THRESHOLD)
    parser.add_argument(
        "--min-points",
        type=int,
        default=100,
        help="minimum points per connected defect region",
    )
    args = parser.parse_args()

    input_file = args.input or INPUT_FILE
    output_dir = args.output or OUTPUT_DIR

    print("point cloud defect detection")

    if not os.path.exists(input_file):
        print(f"input not found: {input_file}")
        return

    if not os.path.exists(args.model):
        print(f"checkpoint not found: {args.model}")
        return

    print(f"load: {input_file}")
    pointcloud = load_pointcloud(input_file)
    print(f"points: {len(pointcloud):,}")

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

    print(f"infer (threshold={args.threshold})")
    labels, scores = predict_pointcloud(
        model, pointcloud, device, num_points, args.threshold, in_channels
    )

    defect_mask = labels == 1
    defect_count = defect_mask.sum()

    print("summary")
    print(f"  defect: {defect_count:,} ({100 * defect_count / len(pointcloud):.2f}%)")
    print(f"  normal: {len(pointcloud) - defect_count:,}")

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_file))[0]

    pd.DataFrame(
        {
            "X": pointcloud[:, 0],
            "Y": pointcloud[:, 1],
            "Z": pointcloud[:, 2],
            "Intensity": pointcloud[:, 3],
            "Score": scores,
            "Label": labels,
        }
    ).to_csv(os.path.join(output_dir, f"{base}_segmented.csv"), index=False)

    defect_points = pointcloud[defect_mask]
    pd.DataFrame(defect_points, columns=["X", "Y", "Z", "Intensity"]).to_csv(
        os.path.join(output_dir, f"{base}_defect.csv"), index=False
    )

    with open(os.path.join(output_dir, f"{base}_defect_cc.txt"), "w") as f:
        for p in defect_points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {p[3]:.6f} 255 50 50\n")

    with open(os.path.join(output_dir, f"{base}_colored_cc.txt"), "w") as f:
        for i, p in enumerate(pointcloud):
            if labels[i] == 1:
                r, g, b = 255, 50, 50
            else:
                s = scores[i]
                r, g, b = int(s * 200), int((1 - s) * 200), int((1 - s) * 255)
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {p[3]:.6f} {r} {g} {b}\n")

    print(f"saved under: {output_dir}")

    if defect_count > 0:
        result = analyze_defects(defect_points, min_points=args.min_points)

        print("defect analysis (coordinates in cm)")
        print(f"  regions: {result['num_defects']}")
        print(f"  total volume (cm^3): {result['total_volume']:.2f}")

        for d in result["defects"][:5]:
            size = d["size_cm"]
            print(
                f"  #{d['id']}: {size['x']:.1f} x {size['y']:.1f} x {size['z']:.1f} cm, "
                f"vol={d['volume_cm3']:.1f} cm^3, pts={d['points']}"
            )

        if result["num_defects"] > 5:
            print(f"  ... and {result['num_defects'] - 5} more")

        with open(os.path.join(output_dir, f"{base}_report.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print("done")


if __name__ == "__main__":
    main()
