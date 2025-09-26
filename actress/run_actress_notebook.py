import numpy as np
import matplotlib.pyplot as plt
import os
import actress as ac
import multiprocessing as mp
mp.set_start_method('fork',force=True)
from multiprocessing import Pool

class Transitparams(object):
    
    def __init__(self):
        self.res = None
        self.rp = None
        self.b = None
        self.N = None
        self.mode = None
        self.fac_r = None
        self.fac_long = None
        self.fac_lat = None
        self.fac_band = False
        self.fac_band_low = None
        self.fac_band_high = None
        self.a = None
        self.T = None
        self.phi = None
        self.ld = None

class Transitsim(object):
    
    def __init__(self,params):
        self.res = params.res
        self.rp = params.rp
        self.b = params.b
        self.N = params.N
        self.mode = params.mode
        self.fac_r = params.fac_r
        self.fac_long = params.fac_long
        self.fac_lat = params.fac_lat
        self.fac_band = params.fac_band
        self.fac_band_low = params.fac_band_low
        self.fac_band_high = params.fac_band_high
        self.a = params.a
        self.T = params.T
        self.phi = params.phi
        self.ld = params.ld
        
    def ld_law(self, mu, I0, a, b, c=None, d=None):
        if self.ld == 'quadratic':
            y = I0 * (1 - a*(1-mu) - b*((1-mu)**2))
            
        if self.ld == 'power2':
            y = I0 * (1-(a*(1-(mu**b))))
            
        if self.ld == 'claret':
            y = I0 * (1 - a*(1-(mu**0.5)) - b*(1-mu) - c*(1-(mu**1.5)) - d*(1-(mu**2)))
            
        return y
        
    def actress_run(self,wavelength, wavelength_fac, I0,c1,c2, I0_fac,c1_fac,c2_fac, c3=None,c4=None,c3_fac=None,c4_fac=None, gif_save=None, lightcurve_save=None):
        sim = ac.Simulator() #create simulation instance
        sim.setxsize(self.res)

        # Plot HealPix map
        import healpy as hp
        m = sim.makemap(mode=self.mode)
        hp.mollview(m, title=f'HealPix Map ({self.mode})', cmap='viridis')
       # plt.show()
        
        # Calculate surface normal projection (mu) and line-of-sight velocity
        nside = hp.npix2nside(len(m))
        theta, phi = hp.pix2ang(nside, np.arange(len(m)))
        
        # Surface normal projection (mu)
        mu = np.sin(theta) * np.cos(phi)
        mu[mu <= 0] = np.nan
        hp.mollview(mu, title='Surface Normal Projection (μ) - Front Side Only', cmap='plasma')
        plt.show()
        
        # Line-of-sight velocity for vertical axis rotation
        v_eq = 10  # Equatorial velocity in m/s
        v_los = v_eq * np.sin(theta) * np.sin(phi)
        front_mask = np.sin(theta) * np.cos(phi) > 0
        v_los[~front_mask] = np.nan
        hp.mollview(v_los, title='Line-of-Sight Velocity (m/s) - Vertical Axis Rotation', cmap='RdBu')
        plt.show()



        #sim.setresolution(15)

            #define limb-darkening parameters (lists in a dictionary):
        ld_dict = {'phot':[I0,c1,c2,c3,c4], #photospheric coeffs
                   'spot':[12955331.717385203,0.3,0.2,0.1,0.1], #spot coeffs
                   'fac':[I0_fac,c1_fac,c2_fac,c3_fac,c4_fac],  #facular coeffs
                   'func':self.ld_law}       #limb-darkening fn (that takes N arguments)

        sim.setld(ld_dict) #update simulation ld coeffs
        
        if self.mode == 'faconly':
            #sim.addfeature(r = self.fac_r, lon= self.fac_long, lat=self.fac_lat, feature = 'fac') #add a circular facular region with radius r [deg], longitude lon [deg] and latitude lat [deg]
            #sim.addfeature(r=5, lon=80, lat=0, feature = 'spot') #add a circular spot with radius r [deg], longitude lon [deg] and latitude lat [deg]

            if self.fac_band is True and self.fac_r is not None: 
                for i in range(0,len(self.fac_r)): 
                    sim.addfeature(r = self.fac_r[i], lon= self.fac_long[i], lat=self.fac_lat[i], feature = 'fac') #add a circular facular region with radius r [deg], longitude lon [deg] and latitude lat [deg]
                for i in range(0,len(self.fac_band_low)):
                    sim.addstrip(lower=self.fac_band_low[i], upper=self.fac_band_high[i], feature='fac') #TO GET STRIP OF SPOTS/FACULAE

            elif self.fac_band is True and self.fac_r is None: 
                for i in range(0,len(self.fac_band_low)):
                    sim.addstrip(lower=self.fac_band_low[i], upper=self.fac_band_high[i], feature='fac') #TO GET STRIP OF SPOTS/FACULAE

            elif self.fac_band is False and self.fac_r is not None:
                for i in range(0,len(self.fac_r)): 
                    sim.addfeature(r = self.fac_r[i], lon= self.fac_long[i], lat=self.fac_lat[i], feature = 'fac') #add a circular facular region with radius r [deg], longitude lon [deg] and latitude lat [deg]



        """
        for all following, 
        i: stellar inclination [deg] (i=90 deg = equator-on)
        N: number of datapoints
        mode: available modes - 'both' (spot+fac), 'faconly' (faculae only), 'spotonly' (spots only), 'quiet' (no features)
        """
        with Pool() as pool:

            N = 50

            print(f'{wavelength * 1e10:.3f} Angstroms')
            wavelength_text = f"{wavelength * 1e10:.3f}"  # meters → Ångstroms
            wavelength_fac_text = f"{wavelength_fac * 1e10:.3f}"

            if gif_save:
                gif_save_directory = f'./outputs/gifs/{gif_save}/'
                os.makedirs(gif_save_directory, exist_ok=True)
                gif_save = f'{gif_save_directory}anim_{(wavelength_text)}.gif'
                rotate_anim = sim.rotate_anim(inc=90, N=N, fluxunits='erg', save=gif_save, norm=False, wavelength=wavelength, outputLC=False) #create animation of rotating star and resulting lightcurve (same as above) #Dana edit making N=different from 10

            if lightcurve_save:
                lightcurve_save_directory = f'./outputs/lightcurves/{lightcurve_save}/'
                os.makedirs(lightcurve_save_directory, exist_ok=True)
                lightcurve_save = f'{lightcurve_save_directory}lc_{(wavelength_text)}.txt'
                lcr = sim.rotate_lc(inc=90,N=N, mode='faconly', wavelength=wavelength_text) #calculate single-period rotational lightcurve
                np.savetxt(lightcurve_save, lcr)

            # lct = sim.transit_lc(radratio=self.rp, inc=90, b=self.b, N=self.N, mode=self.mode, a=self.a, T=self.T, phi = self.phi, save_transit=None) #calculate transit lightcurve, with planet/star radius ratio rr
            
            # tmin = 0.5*self.T*(0.5 - self.phi)
            # tmax = 0.5*self.T*(0.5 + self.phi)
            # t = np.linspace(0, tmax - tmin, self.N)
            
            
            
            #return t, lct

    def sim_spectrum(self, hd_ld_file, fac_ld_file, gif_save=True, lightcurve_save=True):
        hd_ld = np.loadtxt(hd_ld_file)

        if self.mode == 'faconly':
            fac_ld = np.loadtxt(fac_ld_file)
        val = []
        time = []



        for i in range(0,len(hd_ld)):
            wavelength = hd_ld[i][0] #Dana edit
            I0 = hd_ld[i][1]
            c1 = hd_ld[i][2]
            c2 = hd_ld[i][3]
            if self.ld == 'claret':
                c3 = hd_ld[i][4]
                c4 = hd_ld[i][5]
            if self.mode == 'faconly':
                wavelength_fac = fac_ld[i][0] #Dana edit
                I0_fac = fac_ld[i][1]
                c1_fac = fac_ld[i][2]
                c2_fac = fac_ld[i][3]
                if self.ld == 'claret':
                    c3_fac = fac_ld[i][4]
                    c4_fac = fac_ld[i][5]
            elif self.mode == 'quiet':
                I0_fac = 100
                c1_fac = 0.1
                c2_fac = 0.1
                if self.ld == 'claret':
                    c3_fac = 0.1
                    c4_fac = 0.1
            if self.ld == 'claret':
                self.actress_run(wavelength, wavelength_fac, I0,c1,c2,I0_fac,c1_fac,c2_fac,c3=c3,c4=c4,c3_fac=c3_fac,c4_fac=c4_fac, gif_save=gif_save, lightcurve_save=lightcurve_save) #Dana edit
            else:
                self.actress_run(wavelength, wavelength_fac,I0,c1,c2,I0_fac,c1_fac,c2_fac, gif_save=gif_save, lightcurve_save=lightcurve_save) #Dana edit 
        #     val.append(lct)
        #     time.append(t)
        # val = np.asarray(val)
        # time = np.asarray(time)


    def sim_phot(self, hd_ld_file, fac_ld_file, save=None):
        hd_ld = np.loadtxt(hd_ld_file)

        if self.mode == 'faconly':
            fac_ld = np.loadtxt(fac_ld_file)
            
        I0 = hd_ld[1]
        c1 = hd_ld[2]
        c2 = hd_ld[3]
        if self.ld == 'claret':
            c3 = hd_ld[4]
            c4 = hd_ld[5]
        if self.mode == 'faconly':
            I0_fac = fac_ld[1]
            c1_fac = fac_ld[2]
            c2_fac = fac_ld[3]
            if self.ld == 'claret':
                c3_fac = fac_ld[4]
                c4_fac = fac_ld[5]
        elif self.mode == 'quiet':
            I0_fac = 100
            c1_fac = 0.1
            c2_fac = 0.1
            if self.ld == 'claret':
                c3_fac = 0.1
                c4_fac = 0.1
        if self.ld == 'claret':
            t, lct = self.actress_run(I0,c1,c2,I0_fac,c1_fac,c2_fac,c3=c3,c4=c4,c3_fac=c3_fac,c4_fac=c4_fac)
        else:
            t, lct = self.actress_run(I0,c1,c2,I0_fac,c1_fac,c2_fac)
        lct = np.asarray(lct)
        t = np.asarray(t)
        if save is not None:
            v = np.column_stack((t,lct))
            np.savetxt(save,v)
        
        
        
