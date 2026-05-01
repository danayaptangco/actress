import numpy as np
from pathlib import Path
import re

def process_disk_directory(directory, phase_idx=0):
    """
    Reads disk arrays and wavelength shifts from a directory, 
    applies Doppler shifts, filters edges, and returns the binned spectrum.
    """
    directory = Path(directory)
    
    # 1. Load intensity maps
    npy_files = sorted(directory.glob('disk_inp*.npy'))
    if not npy_files:
        raise FileNotFoundError(f"No disk_inp*.npy files found in {directory}")
        
    def parse_wl(name):
        m = re.search(r'inp([0-9.]+)\.npy$', name)
        return float(m.group(1)) if m else np.nan
        
    intensity_maps = {parse_wl(f.name): np.load(f) for f in npy_files}
    wavelengths = np.array(sorted(intensity_maps.keys()))
    
    # 2. Load shift maps
    shift_files = sorted(directory.glob('wavelength_shifts_inp*.csv'))
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
    
    binned_intensity = np.zeros(len(bin_centers))
    for i, idx in enumerate(valid_indices):
        mask = (np.round(shifted_lambda_filtered, 10) >= np.round(bin_edges_full[idx], 10)) & \
               (np.round(shifted_lambda_filtered, 10) < np.round(bin_edges_full[idx + 1], 10))
        if mask.any():
            binned_intensity[i] = intensity_filtered[mask].sum() / len(intensity_filtered[mask])
            
    return bin_centers, binned_intensity

def get_simulation_params(directory):
    """Reads the header from the first available csv parameter file."""
    directory = Path(directory)
    first_csv = next(directory.glob('wavelength_shifts_inp*.csv'), None)
    if first_csv:
        with open(first_csv, 'r') as f:
            return f.readline().strip().lstrip('# ')
    return "Unknown parameters"