# Magnetic LD Library

This repository contains a library of synthetic stellar spectra, wavelength-dependent limb-darkening profiles, and disc-integrated fluxes.

The data are based on 3D radiative magnetohydrodynamic simulations performed with the MURaM code and spectral synthesis performed with MPS-ATLAS. For each stellar model, spectra are provided for different magnetization cases and viewing angles, allowing the analysis of wavelength-dependent limb darkening.

# General description of the data

Each HDF5 file contains the data for one stellar model. A stellar model is characterized by its spectral type, surface gravity, and metallicity. For each stellar model, several magnetic configurations are included.

The files contain:

* Wavelength grid in the range 200 -- 10,000 nm
* μ grid, where μ = cos(θ), in the range [0.1 -- 1.0]
* Magnetization cases
* Effective temperature values, in K
* Synthetic spectra / specific intensities 
* limb darkening profiles
* disk-integrated fluxes

The magnetization cases are:

* hydro: hydrodynamical simulation without magnetic field
* ssd: small-scale dynamo simulation, representing the quiet-star magnetic condition
* B100G: simulation with imposed magnetic field of 100 G
* B200G: simulation with imposed magnetic field of 200 G
* B300G: simulation with imposed magnetic field of 300 G

The B100G, B200G, and B300G simulations represent facular regions with different magnetic field strengths.

# Description of files

## Name of the files
Each file name corresponds to the spectral type of the star and its metallicity.

* Example file names:
    G2_MH_m10.h5
    G2_MH_p05.h5
    M0_MH_00.h5

where:

G2, M0, etc. indicate the stellar spectral type
MH_00, MH_m10, MH_p05, etc. indicate the metallicity [M/H] = 0.0, -1.0, 0.5

## Content of the files

Each HDF5 file contains one main group corresponding to the stellar model.

The general structure is:
/<star_group>/
    mu
    wavelengths
    magnetizations
    teff
    mh
    logg
    spectra
    limb_darkening
    integrated_flux

* Datasets
Dataset	        Shape	                                Description
mu	            (n_mu,)	                                μ grid, where μ = cos(θ)
wavelengths	    (n_wavelength,)	                        Wavelength grid
magnetizations	(n_magnetization,)	                    Names of the magnetic simulation runs
teff	        (n_magnetization,)	                    Effective temperature for each magnetization case
spectra	        (n_magnetization, n_wavelength, n_mu)	Synthetic spectra / specific intensities
limb_darkening	(n_magnetization, n_wavelength, n_mu)	Limb-darkening profiles normalized to disc center
integrated_flux	(n_magnetization, n_wavelength)	        Disk-integrated flux

* Attributes

The stellar parameters are stored as attributes of the stellar group.

Attribute	    Description
star_name	    Name of the stellar model
logg	        Surface gravity
logg_units	    log10(cm s^-2)
MH	            Metallicity [M/H]
MH_units	    dex

* Units

The default units are:

Quantity	            Units
Wavelength	            nm, in vacuum
μ	                    cos(theta)
Spectra	                erg s^-1 cm^-2 sr^-1 Å^-1
Limb darkening	        normalized to disc center
Integrated flux	        erg s^-1 cm^-2 Å^-1
Effective temperature	K

## How to read the files
In the file
`read_spectra.py`

there are many functions which allow you to get different data from the file:
`get_disk_integrated_flux`, `get_limb_darkening`, `get_spectra`, `get_wavelengths`, `get_mu`

and additional information:
`get_fundamental_parameters`, `get_magnetizations`, `get_units_for_disk_integrated_flux`, `get_units_for_spectra`


Alternatively, the data can be explored interactively with the provided Jupyter notebook:
`read_spectral_library.ipynb`

The notebook allows the user to specify the stellar model and magnetization case, and to select the quantity of interest: synthetic spectra, limb-darkening profiles, or disc-integrated fluxes.


# Citation:
If you use this library, please cite the corresponding Magnetic LD library paper:

N. Kostogryz, A. I. Shapiro, V. Witzke, T. Bhatia, S. K. Solanki, I. Kuhlemann, V. Vasilyev, Y. C. Unruh, Effect of surface magnetic fields on limb darkening in main-sequence stars accepted for A&A 2026


Please also cite the codes used to produce the simulations and spectra:

* MURAM code:
@ARTICLE{2005A&A...429..335V,
       author = {{V{\"o}gler}, A. and {Shelyag}, S. and {Sch{\"u}ssler}, M. and {Cattaneo}, F. and {Emonet}, T. and {Linde}, T.},
        title = "{Simulations of magneto-convection in the solar photosphere.  Equations, methods, and results of the MURaM code}",
      journal = {\aap},
     keywords = {magnetohydrodynamics (MHD), Sun: magnetic fields, Sun: photosphere, Sun: granulation, Sun: faculae, plages},
         year = 2005,
        month = jan,
       volume = {429},
        pages = {335-351},
          doi = {10.1051/0004-6361:20041507},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2005A&A...429..335V},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

* MPS-ATLAS code:
@ARTICLE{2021A&A...653A..65W,
       author = {{Witzke}, V. and {Shapiro}, A.~I. and {Cernetic}, M. and {Tagirov}, R.~V. and {Kostogryz}, N.~M. and {Anusha}, L.~S. and {Unruh}, Y.~C. and {Solanki}, S.~K. and {Kurucz}, R.~L.},
        title = "{MPS-ATLAS: A fast all-in-one code for synthesising stellar spectra}",
      journal = {\aap},
     keywords = {stars: atmospheres, stars: late-type, radiative transfer, opacity, convection, Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2021,
        month = sep,
       volume = {653},
          eid = {A65},
        pages = {A65},
          doi = {10.1051/0004-6361/202140275},
archivePrefix = {arXiv},
       eprint = {2105.13611},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021A&A...653A..65W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

