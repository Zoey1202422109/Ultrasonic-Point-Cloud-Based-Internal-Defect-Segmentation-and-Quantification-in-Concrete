import numpy as np
import pandas as pd
from scipy.ndimage import zoom, gaussian_filter
from typing import Dict
import os
import time


def detect_grid_info(coords: np.ndarray) -> Dict:
    """Infer regular grid axes and spacing from point coordinates."""
    unique_x = np.sort(np.unique(np.round(coords[:, 0], decimals=4)))
    unique_y = np.sort(np.unique(np.round(coords[:, 1], decimals=4)))
    unique_z = np.sort(np.unique(np.round(coords[:, 2], decimals=4)))

    spacing_x = np.diff(unique_x).mean() if len(unique_x) > 1 else 1.0
    spacing_y = np.diff(unique_y).mean() if len(unique_y) > 1 else 1.0
    spacing_z = np.diff(unique_z).mean() if len(unique_z) > 1 else 1.0

    return {
        'x': unique_x, 'y': unique_y, 'z': unique_z,
        'nx': len(unique_x), 'ny': len(unique_y), 'nz': len(unique_z),
        'spacing': (spacing_x, spacing_y, spacing_z),
        'origin': (unique_x.min(), unique_y.min(), unique_z.min()),
        'extent': (unique_x.max(), unique_y.max(), unique_z.max())
    }


def pointcloud_to_grid(coords: np.ndarray, intensities: np.ndarray,
                       grid_info: Dict) -> np.ndarray:
    """Map irregular points onto a 3D grid by nearest axis index."""
    nx, ny, nz = grid_info['nx'], grid_info['ny'], grid_info['nz']
    unique_x, unique_y, unique_z = grid_info['x'], grid_info['y'], grid_info['z']

    grid = np.zeros((nx, ny, nz), dtype=np.float64)

    for i in range(len(coords)):
        x, y, z = coords[i]
        ix = np.argmin(np.abs(unique_x - round(x, 4)))
        iy = np.argmin(np.abs(unique_y - round(y, 4)))
        iz = np.argmin(np.abs(unique_z - round(z, 4)))
        grid[ix, iy, iz] = intensities[i]

    return grid


def grid_to_pointcloud(grid: np.ndarray, x_coords: np.ndarray,
                       y_coords: np.ndarray, z_coords: np.ndarray) -> np.ndarray:
    """Flatten grid to Nx4 array (x, y, z, intensity)."""
    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
    points = np.zeros((nx * ny * nz, 4), dtype=np.float32)

    idx = 0
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            for k, z in enumerate(z_coords):
                points[idx] = [x, y, z, grid[i, j, k]]
                idx += 1

    return points


def db_to_amplitude(db_values: np.ndarray) -> np.ndarray:
    """dB to linear amplitude."""
    db_values = np.clip(db_values, -120, 120)
    return np.power(10.0, db_values / 20.0)


def amplitude_to_db(amplitude_values: np.ndarray) -> np.ndarray:
    """Linear amplitude to dB."""
    amplitude_values = np.maximum(amplitude_values, 1e-10)
    return 20.0 * np.log10(amplitude_values)


def acoustic_trilinear_upsample(grid_db: np.ndarray, factor: int) -> np.ndarray:
    """
    Upsample in linear amplitude with trilinear interpolation (order=1), then map back to dB.
    """
    amp = db_to_amplitude(grid_db)
    amp_up = zoom(amp, factor, order=1, mode='nearest')
    return amplitude_to_db(amp_up)


def intensity_aware_enhancement(input_file: str,
                                output_file: str = None,
                                subdivide_factor: int = 3,
                                smooth: bool = False,
                                smooth_sigma: float = 0.3,
                                verbose: bool = True):
    """
    Load CSV (X,Y,Z,Intensity in dB), acoustic trilinear upsample, optional Gaussian smooth, save CSV.
    """
    def log(msg):
        if verbose:
            print(msg)

    log("=" * 70)
    log("Acoustic trilinear grid enhancement")
    log("=" * 70)

    start_time = time.time()

    if not os.path.exists(input_file):
        print(f"[ERROR] File not found: {input_file}")
        return None, None

    log(f"\n[1/4] Loading: {input_file}")
    df = pd.read_csv(input_file)
    pointcloud = df[['X', 'Y', 'Z', 'Intensity']].values.astype(np.float32)
    coords = pointcloud[:, :3]
    intensities = pointcloud[:, 3]

    log(f"      Points: {len(pointcloud):,}")
    log(f"      Intensity range: [{intensities.min():.2f}, {intensities.max():.2f}] dB")

    log(f"\n[2/4] Grid conversion...")
    grid_info = detect_grid_info(coords)
    grid = pointcloud_to_grid(coords, intensities, grid_info)
    log(f"      Grid size: {grid_info['nx']} x {grid_info['ny']} x {grid_info['nz']}")

    log(f"\n[3/4] Acoustic trilinear upsample (factor={subdivide_factor})...")
    subdivided = acoustic_trilinear_upsample(grid, subdivide_factor)
    new_shape = subdivided.shape
    log(f"      New grid: {new_shape[0]} x {new_shape[1]} x {new_shape[2]}")

    if smooth:
        log(f"      Gaussian smoothing (sigma={smooth_sigma})...")
        subdivided = gaussian_filter(subdivided, sigma=smooth_sigma)

    log(f"\n[4/4] Converting to point cloud...")
    new_x = np.linspace(grid_info['x'].min(), grid_info['x'].max(), new_shape[0])
    new_y = np.linspace(grid_info['y'].min(), grid_info['y'].max(), new_shape[1])
    new_z = np.linspace(grid_info['z'].min(), grid_info['z'].max(), new_shape[2])
    enhanced = grid_to_pointcloud(subdivided, new_x, new_y, new_z)

    elapsed = time.time() - start_time

    log(f"\n" + "=" * 70)
    log(f"Results:")
    log(f"  Final points: {len(enhanced):,}")
    log(f"  Expansion ratio: {len(enhanced) / len(pointcloud):.1f}x")
    log(f"  Original intensity: [{intensities.min():.2f}, {intensities.max():.2f}] dB")
    log(f"  Enhanced intensity: [{enhanced[:, 3].min():.2f}, {enhanced[:, 3].max():.2f}] dB")
    log(f"  Processing time: {elapsed:.2f}s")

    if output_file is None:
        base_dir = os.path.dirname(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(base_dir, f"{base_name}_enhanced_x{subdivide_factor}.csv")

    df_output = pd.DataFrame(enhanced, columns=['X', 'Y', 'Z', 'Intensity'])
    df_output.to_csv(output_file, index=False)

    log(f"\n  Saved: {output_file}")
    log(f"  Size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    log("=" * 70)

    return enhanced, output_file


if __name__ == '__main__':
    import sys
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    input_file = r""

    enhanced, output_file = intensity_aware_enhancement(
        input_file=input_file,
        subdivide_factor=3,
        smooth=True,
        smooth_sigma=0.3,
        verbose=True
    )
