import numpy as np
from pathlib import Path
import re

def _cache_key(npy_files, shift_files, phase_idx):
    """Identifies the inputs a cached spectrum was built from.

    Changes if files are added/removed or any file is rewritten, so a stale
    cache is detected rather than silently reused.
    """
    mtimes = [f.stat().st_mtime for f in npy_files] + [f.stat().st_mtime for f in shift_files]
    return np.array([len(npy_files), len(shift_files), phase_idx, max(mtimes)], dtype=float)


def process_disk_directory(directory, phase_idx=0, use_cache=True):
    """
    Reads disk arrays and wavelength shifts from a directory,
    applies Doppler shifts, filters edges, and returns the binned spectrum.

    If use_cache is True, the binned result is cached in a 'binned_cache'
    directory alongside `directory` and reused when the inputs are unchanged.
    Pass use_cache=False to always recompute from the raw files.
    """
    directory = Path(directory)

    # 1. Load intensity maps
    npy_files = sorted(directory.glob('disk_inp*.npy'))
    if not npy_files:
        raise FileNotFoundError(f"No disk_inp*.npy files found in {directory}")

    shift_files_for_key = sorted(directory.glob('wavelength_shifts_inp*.csv'))
    cache_path = directory.parent / 'binned_cache' / f'{directory.name}_phase{phase_idx}.npz'
    if use_cache:
        key = _cache_key(npy_files, shift_files_for_key, phase_idx)
        if cache_path.is_file():
            cached = np.load(cache_path)
            if cached['key'].shape == key.shape and np.array_equal(cached['key'], key):
                return cached['bin_centers'], cached['binned_intensity']

    def parse_wl(name):
        m = re.search(r'inp([0-9.]+)\.npy$', name)
        return float(m.group(1)) if m else np.nan
        
    intensity_maps = {parse_wl(f.name): np.load(f) for f in npy_files}
    wavelengths = np.array(sorted(intensity_maps.keys()))
    
    # 2. Load shift maps
    shift_files = shift_files_for_key
    def parse_shift_wl(name):
        m = re.search(r'inp([0-9.]+)\.csv$', name)
        return float(m.group(1)) if m else np.nan
    
    wavelength_shifts_by_wl = {parse_shift_wl(f.name): np.loadtxt(f) for f in shift_files}
    wavelength_shifts_all = np.stack([wavelength_shifts_by_wl[wl] for wl in wavelengths], axis=0)
    
    # 3. Apply Shifts & Map to 1D
    shifted_lambda_all = []
    intensity_all = []
    shift_scale = 1
    
    for inp_wl in wavelengths:
        # Get specified phase
        intensity_2d = intensity_maps[inp_wl][phase_idx, :, :]
        dlam_2d = wavelength_shifts_by_wl[inp_wl] * shift_scale
        lam_obs = inp_wl + dlam_2d
        
        m = intensity_2d > 0
        shifted_lambda_all.append(lam_obs[m].ravel())
        intensity_all.append(intensity_2d[m].ravel())
        
    shifted_lambda_all = np.concatenate(shifted_lambda_all)
    intensity_all = np.concatenate(intensity_all)
    
    # 4. Remove Edge Points
    max_shift = wavelength_shifts_all.max()
    wl_min_edge = wavelengths.min() + max_shift
    wl_max_edge = wavelengths.max() - max_shift
    
    valid_mask = (shifted_lambda_all >= wl_min_edge) & (shifted_lambda_all <= wl_max_edge)
    shifted_lambda_filtered = shifted_lambda_all[valid_mask]
    intensity_filtered = intensity_all[valid_mask]
    
    # 5. Binning
    bin_centers_full = np.linspace(wavelengths.min(), wavelengths.max(), len(wavelengths))
    dw = bin_centers_full[1] - bin_centers_full[0] if len(bin_centers_full) > 1 else 0
    bin_edges_full = np.linspace(bin_centers_full[0] - dw/2, bin_centers_full[-1] + dw/2, len(bin_centers_full) + 1)
    
    valid_bins_mask = (bin_edges_full[:-1] >= wl_min_edge) & (bin_edges_full[1:] <= wl_max_edge)
    bin_centers = bin_centers_full[valid_bins_mask]
    valid_indices = np.where(valid_bins_mask)[0]
    
    lam_rounded = np.round(shifted_lambda_filtered, 10)
    edges_rounded = np.round(bin_edges_full, 10)

    # Assign each point to its bin in one pass: side='right' minus 1 gives the
    # half-open convention [edge_i, edge_i+1) used by the previous loop.
    n_bins_full = len(edges_rounded) - 1
    point_bins = np.searchsorted(edges_rounded, lam_rounded, side='right') - 1
    in_range = (point_bins >= 0) & (point_bins < n_bins_full)
    point_bins = point_bins[in_range]

    counts_full = np.bincount(point_bins, minlength=n_bins_full)
    sums_full = np.bincount(point_bins, weights=intensity_filtered[in_range],
                            minlength=n_bins_full)
    # Empty bins stay 0.0, matching the previous behaviour.
    # An empty bin has no intensity to average. Returning 0.0 would be
    # indistinguishable from a genuine zero and would silently become inf/nan
    # when used as a contrast denominator, so fail loudly instead.
    empty = counts_full[valid_indices] == 0
    if empty.any():
        raise ValueError(
            f"{empty.sum()} of {len(valid_indices)} wavelength bins are empty in {directory} "
            f"(phase_idx={phase_idx}); no disc pixel falls in them after Doppler shifting. "
            f"First empty bin centre: {bin_centers[empty][0]:.4f} A. "
            "The bin grid is finer than the shifted sampling supports.")

    means_full = np.divide(sums_full, counts_full,
                           out=np.zeros(n_bins_full), where=counts_full > 0)
    binned_intensity = means_full[valid_indices]

    if use_cache:
        cache_path.parent.mkdir(exist_ok=True)
        np.savez(cache_path, key=key, bin_centers=bin_centers,
                 binned_intensity=binned_intensity)

    return bin_centers, binned_intensity

def get_simulation_params(directory):
    """Reads the header from the first available csv parameter file."""
    directory = Path(directory)
    first_csv = next(directory.glob('wavelength_shifts_inp*.csv'), None)
    if first_csv:
        with open(first_csv, 'r') as f:
            return f.readline().strip().lstrip('# ')
    return "Unknown parameters"