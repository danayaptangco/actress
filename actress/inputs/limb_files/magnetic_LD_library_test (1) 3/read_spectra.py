
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

Array1D = npt.NDArray[np.float64]
Array2D = npt.NDArray[np.float64]

DATA_FOLDER = Path(__file__).parent


def get_magnetizations(
    star: str
) -> list[str]:
    """
    Read magnetizations from h5 file for given star.

    Parameters
    ----------
    star : str
        Star identifier, e.g. "F3_MH_00".

    Returns
    -------
    magnetizations : list[str]
        List of magnetization identifiers, e.g. ["hydro", "B100G", ...].
    """
    with h5py.File(_get_star_h5_file(star), 'r') as h5_file:
        return [m.decode() for m in h5_file[star]["magnetizations"][:]]


def get_disk_integrated_flux(
    star: str,
    magnetization: str
) -> Array1D:
    """
    Read disk integrated flux from h5 file for given star and magnetization.

    Parameters
    ----------
    star : str
        Star identifier, e.g. "F3_MH_00".
    magnerization : str
        Magnetization identifier, e.g. "B100G". Read available identifiers with `get_magnetizations`.

    Returns
    -------
    disk_integrated_flux : Array1D
        Has same length as the returned wavelengths array.
    """
    with h5py.File(_get_star_h5_file(star), 'r') as h5_file:
        m = get_magnetizations(star).index(magnetization)
        disk_integrated_flux_m: Array1D = h5_file[star]["integrated_flux"][m, :]
        return disk_integrated_flux_m


def get_spectra(
    star: str,
    magnetization: str,
) -> Array2D:
    """
    Read spectra from h5 file for given star and magnetization.

    Parameters
    ----------
    star : str
        Star identifier, e.g. "F3_MH_00".
    magnerization : str
        Magnetization identifier, e.g. "B100G". Read available identifiers with `get_magnetizations`.

    Returns
    -------
    spectra : Array2D
        Has shape (L, M), where L is the lenght of the wavelengths array, 
        and M is the length of the mu array.
    """    
    with h5py.File(_get_star_h5_file(star), 'r') as h5_file:
        m = get_magnetizations(star).index(magnetization)
        spectra: Array2D = h5_file[star]["spectra"][m]
        return spectra


def get_limb_darkening(
    star: str,
    magnetization: str,
) -> Array2D:
    """
    Read limb darkening from h5 file for given star and magnetization.

    Parameters
    ----------
    star : str
        Star identifier, e.g. "F3_MH_00".
    magnerization : str
        Magnetization identifier, e.g. "B100G". Read available identifiers with `get_magnetizations`.

    Returns
    -------
    limb_darkening : Array2D
        Has shape (L, M), where L is the lenght of the wavelengths array, 
        and M is the length of the mu array.
    """
    with h5py.File(_get_star_h5_file(star), 'r') as h5_file:
        m = get_magnetizations(star).index(magnetization)
        limb_darkening: Array2D = h5_file[star]["limb_darkening"][m]
        return limb_darkening


def get_fundamental_parameters(
    star: str,
    magnetization: str
) -> tuple[float, float, float]:
    """
    returns effective temperature teff, metallicity MH, surface gravity logg,
    """
    with h5py.File(_get_star_h5_file(star), 'r') as h5_file:
        m = get_magnetizations(star).index(magnetization)
        return h5_file[star]['teff'][m], h5_file[star].attrs["MH"], h5_file[star].attrs["logg"]


def get_units_for_disk_integrated_flux(
    star: str
) -> str:
    """
    Read units of disk integrated flux from h5 file for given star.

    Parameters
    ----------
    star : str
        Star identifier, e.g. "F3_MH_00".    

    Returns
    -------
    units : str
    """    
    with h5py.File(_get_star_h5_file(star), 'r') as h5_file:
        return h5_file[star]["integrated_flux"].attrs['units']


def get_units_for_spectra(
    star: str
) -> str:
    """
    Read units of spectra from h5 file for given star.

    Parameters
    ----------
    star : str
        Star identifier, e.g. "F3_MH_00".    

    Returns
    -------
    units : str
    """
    with h5py.File(_get_star_h5_file(star), 'r') as h5_file:
        return h5_file[star]["spectra"].attrs['units']


def get_wavelengths(star: str) -> Array1D:
    """
    Read wavelengths from h5 file for given star.

    Parameters
    ----------
    star : str
        Star identifier, e.g. "F3_MH_00".    

    Returns
    -------
    wavelengths : Array1D    
    """
    with h5py.File(_get_star_h5_file(star), 'r') as h5_file:
        return h5_file[star]["wavelengths"][:]
    
def get_units_for_wavelengths(
    star: str
) -> str:
    """
    Read units of wavelengths from h5 file for given star.

    Parameters
    ----------
    star : str
        Star identifier, e.g. "F3_MH_00".    

    Returns
    -------
    units : str
    """
    with h5py.File(_get_star_h5_file(star), 'r') as h5_file:
        return h5_file[star]["wavelengths"].attrs['units']


def get_mu(star: str) -> Array1D:
    """
    Read mu values from h5 file for given star.

    Parameters
    ----------
    star : str
        Star identifier, e.g. "F3_MH_00".    

    Returns
    -------
    mu : Array1D    
    """
    with h5py.File(_get_star_h5_file(star), 'r') as h5_file:
        return h5_file[star]["mu"][:]


def _get_star_h5_file(star: str) -> Path:
    return DATA_FOLDER / f"{star}.h5"
