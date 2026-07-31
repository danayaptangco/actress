"""Transit analogue of ../full_disks_and_wavelength_shifts/disk_utils.py.

Where disk_utils reads a `disk_inp*.npy` stack indexed by rotation phase, the transit runs
(written by `transit_disk_save`, see `simulator.py`'s `transit_lc(..., returndisc=True, v_eq=...)`)
store `transit_disk_inp*.npy` indexed by transit position, with the planet's shadow already
masked out of the disc. The binning here follows 1_analyze_transits.ipynb: bin edges are the
midpoints of the input wavelength grid, and bins within max|shift| of either edge are dropped
because the Doppler correction is incomplete there.
"""

import re
from pathlib import Path

import numpy as np

# Loading a transit directory means reading ~84 MB of .npy plus binning 49 positions, so cache
# both the raw stacks and the finished spectrum-vs-time array per directory.
_disc_cache = {}
_spectrum_cache = {}

shift_scale = 1  # shift maps are stored in the same units as the wavelength filenames (Angstrom)


def _parse_wl(name, ext):
    m = re.search(rf'inp([0-9.]+)\.{ext}$', name)
    return float(m.group(1)) if m else np.nan


def load_transit_directory(directory):
    """Load one transit run.

    Returns a dict with:
      wavelengths      (n_wl,)              input wavelength grid, sorted
      discs            {wl: (n_pos, H, W)}  intensity, planet shadow already masked out
      shifts           {wl: (H, W)}         static Doppler shift map (Angstrom)
      shifts_all       (n_wl, H, W)
      positions        (n_pos, 2)           planet (x, y) per transit time-step
      n_positions      int
    """
    key = str(Path(directory).resolve())
    if key in _disc_cache:
        return _disc_cache[key]

    directory = Path(directory)

    npy_files = sorted(directory.glob('transit_disk_inp*.npy'))
    if not npy_files:
        raise FileNotFoundError(f"No transit_disk_inp*.npy files found in {directory}")
    discs = {_parse_wl(f.name, 'npy'): np.load(f) for f in npy_files}
    wavelengths = np.array(sorted(discs))

    shift_files = sorted(directory.glob('wavelength_shifts_inp*.csv'))
    if not shift_files:
        raise FileNotFoundError(
            f"No wavelength_shifts_inp*.csv files found in {directory}. "
            "(These only get written when transit_lc was run with v_eq!=0.)")
    shifts = {_parse_wl(f.name, 'csv'): np.loadtxt(f) for f in shift_files}
    missing = [wl for wl in wavelengths if wl not in shifts]
    if missing:
        raise ValueError(f"Missing shift maps in {directory} for wavelengths: {missing}")
    shifts_all = np.stack([shifts[wl] for wl in wavelengths], axis=0)

    pos_file = next(directory.glob('transit_positions_inp*.csv'), None)
    if pos_file is None:
        raise FileNotFoundError(f"No transit_positions_inp*.csv file found in {directory}")
    positions = np.loadtxt(pos_file)

    data = {
        'wavelengths': wavelengths,
        'discs': discs,
        'shifts': shifts,
        'shifts_all': shifts_all,
        'positions': positions,
        'n_positions': positions.shape[0],
    }
    _disc_cache[key] = data
    return data


def wavelength_bins(data):
    """Bin centres / edges for a loaded run, edge-trimmed by the maximum Doppler shift."""
    wavelengths = data['wavelengths']
    max_shift = np.abs(data['shifts_all']).max()
    wl_min_edge = wavelengths.min() + max_shift
    wl_max_edge = wavelengths.max() - max_shift

    midpoints = [(wavelengths[i] + wavelengths[i + 1]) / 2 for i in range(len(wavelengths) - 1)]
    bin_edges = np.array([wavelengths[0] - (midpoints[0] - wavelengths[0]),
                          *midpoints,
                          wavelengths[-1] + (wavelengths[-1] - midpoints[-1])], dtype=float)

    valid_bins_mask = (bin_edges[:-1] >= wl_min_edge) & (bin_edges[1:] <= wl_max_edge)
    bin_centers = wavelengths[valid_bins_mask]
    valid_indices = np.where(valid_bins_mask)[0]

    if len(bin_centers) == 0:
        raise ValueError(
            "No valid wavelength bins survived edge-trimming: the wavelength window "
            f"({wavelengths.min():.4f}-{wavelengths.max():.4f} A) is too narrow relative to "
            f"the Doppler shift range (+/-{max_shift:.4f} A).")

    return bin_centers, bin_edges, valid_indices


def doppler_shifted_spectrum(data, position_idx):
    """Per-pixel (observed wavelength, intensity) scatter at one transit position."""
    lam, inten = [], []
    for wl in data['wavelengths']:
        intensity_2d = data['discs'][wl][position_idx]
        lam_obs = wl + data['shifts'][wl] * shift_scale
        m = intensity_2d > 0  # keep only on-disc, unblocked pixels
        lam.append(lam_obs[m].ravel())
        inten.append(intensity_2d[m].ravel())
    return np.concatenate(lam), np.concatenate(inten)


def process_transit_directory(directory, position_idx=0):
    """Binned Doppler-corrected spectrum at one transit position.

    Transit counterpart of disk_utils.process_disk_directory(directory, phase_idx).
    Returns (bin_centers, binned_intensity).
    """
    bin_centers, spectrum_vs_time = transit_spectrum_vs_time(directory)
    return bin_centers, spectrum_vs_time[position_idx]


def transit_spectrum_vs_time(directory):
    """Binned spectrum at every transit position.

    Returns (bin_centers (n_bins,), spectrum_vs_time (n_positions, n_bins)). Cached per directory,
    since this is the expensive step and every plot downstream wants the whole stack.
    """
    key = str(Path(directory).resolve())
    if key in _spectrum_cache:
        return _spectrum_cache[key]

    data = load_transit_directory(directory)
    bin_centers, bin_edges, valid_indices = wavelength_bins(data)

    spectrum_vs_time = np.full((data['n_positions'], len(bin_centers)), np.nan)
    edges_rounded = np.round(bin_edges, 10)
    for p in range(data['n_positions']):
        shifted_lambda, intensity = doppler_shifted_spectrum(data, p)
        lam_rounded = np.round(shifted_lambda, 10)
        for i, idx in enumerate(valid_indices):
            mask = (lam_rounded >= edges_rounded[idx]) & \
                   (lam_rounded < edges_rounded[idx + 1])
            # An empty bin leaves nan here, which would propagate silently through
            # transit_contrast_vs_time. Fail loudly instead.
            if not mask.any():
                raise ValueError(
                    f"Wavelength bin {i} (centre {bin_centers[i]:.4f} A) is empty at transit "
                    f"position {p} of {data['n_positions']} in {directory}; no unblocked disc "
                    "pixel falls in it after Doppler shifting. The bin grid is finer than the "
                    "shifted sampling supports, or the planet has masked out the contributing "
                    "pixels.")
            spectrum_vs_time[p, i] = intensity[mask].sum() / mask.sum()

    _spectrum_cache[key] = (bin_centers, spectrum_vs_time)
    return bin_centers, spectrum_vs_time


def transit_contrast_vs_time(active, inactive):
    """Contrast (active - inactive) / inactive at every transit position.

    Returns (bin_centers, contrast (n_positions, n_bins)). Raises if the two runs disagree on
    the wavelength grid or the number of transit positions.
    """
    bc_a, spec_a = transit_spectrum_vs_time(active)
    bc_i, spec_i = transit_spectrum_vs_time(inactive)

    if not np.allclose(bc_a, bc_i):
        raise ValueError(f"Wavelength grids differ between {active} and {inactive}")
    if spec_a.shape != spec_i.shape:
        raise ValueError(
            f"Transit position counts differ: {spec_a.shape[0]} ({active}) vs "
            f"{spec_i.shape[0]} ({inactive})")

    return bc_a, (spec_a - spec_i) / spec_i


def disc_images(directory, wavelength=None):
    """Disc intensity image per transit position, for the animation's side panel.

    Averaged over all wavelengths (broadband-ish) unless `wavelength` picks one input wavelength.
    Returns (n_positions, H, W).
    """
    data = load_transit_directory(directory)
    if wavelength is None:
        return np.stack([data['discs'][wl] for wl in data['wavelengths']], axis=0).mean(axis=0)
    wl = min(data['wavelengths'], key=lambda w: abs(w - wavelength))
    return data['discs'][wl]


def inactive_counterpart(active):
    """Inactive (quiet-star) directory matching an active one.

    Same convention as 2_compare_disks.ipynb: zero out the band / lat / long / spot-radius
    specifiers in the directory name.
    """
    inactive = re.sub(r'band_lat-?[\d\.]+_-?[\d\.]+', 'fac_lat0_long0_r0', active)
    inactive = re.sub(r'lat-?\d+', 'lat0', inactive)
    inactive = re.sub(r'long-?\d+', 'long0', inactive)
    inactive = re.sub(r'r(\d+)_', r'r0_', inactive)
    return inactive


def get_simulation_params(directory):
    """Reads the header from the first available csv parameter file."""
    first_csv = next(Path(directory).glob('wavelength_shifts_inp*.csv'), None)
    if first_csv:
        with open(first_csv, 'r') as f:
            return f.readline().strip().lstrip('# ')
    return "Unknown parameters"
