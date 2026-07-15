
from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.typing as npt
import h5py
import matplotlib.pyplot as plt


STAR = "F3_MH_00"
file = Path(__file__).parent / f"{STAR}.h5"


def main() -> None:

    with h5py.File(file, "r") as h5_file:

        print("-" * 8, "data sets in file", "-"*8)
        for key in h5_file[STAR]:
            print(f"{STAR}/{key}")

        magnetizations: Iterable[bytes] = h5_file[STAR]["magnetizations"][:]
        wavelengths: npt.NDArray[np.float64] = h5_file[STAR]["wavelengths"][:]
        integrated_flux: npt.NDArray[np.float64] = h5_file[STAR]["integrated_flux"][:]

        fig, ax = plt.subplots()

        print("-" * 8, "plotting integrated flux for different magnetizations", "-"*8)

        for i, magnetization in enumerate(magnetizations):
            m = magnetization.decode()
            print("magnetization", m)
            ax.plot(
                wavelengths, integrated_flux[i], label=magnetization.decode()
            )
        ax.legend()
        ax.set_xlabel(f"wavelength, [ {h5_file[STAR]["wavelengths"].attrs["units"]}]")
        ax.set_ylabel(f"flux, [{h5_file[STAR]["integrated_flux"].attrs["units"]}]")
    fig.tight_layout()
    print("-"*16)

    plt.show()


if __name__ == "__main__":
    main()
