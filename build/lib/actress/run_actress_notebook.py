import numpy as np
import matplotlib.pyplot as plt
import os
import actress as ac

class Transitparams(object):
    
    def __init__(self):
        self.res2d = None
        self.res3d = None
        self.N = None
        self.mode = None
        self.fac_r = None
        self.fac_long = None
        self.fac_lat = None
        self.fac_band = False
        self.fac_band_low = None
        self.fac_band_high = None
        self.ld = None
        self.v_eq = None
        self.inc = None

        self.rp = None
        self.b = None
        self.a = None
        self.T = None
        self.phi = None


class Transitsim(object):
    
    def __init__(self,params):
        self.res2d = params.res2d
        self.res3d = params.res3d
        self.mode = params.mode
        self.fac_r = params.fac_r
        self.fac_long = params.fac_long
        self.fac_lat = params.fac_lat
        self.fac_band = params.fac_band
        self.fac_band_low = params.fac_band_low
        self.fac_band_high = params.fac_band_high
        self.ld = params.ld
        self.v_eq = params.v_eq
        self.inc = params.inc

        self.rp = params.rp
        self.b = params.b
        self.N = params.N
        self.a = params.a
        self.T = params.T
        self.phi = params.phi
        self.hd_ld_file = None
        self.fac_ld_file = None



        
    def ld_law(self, mu, I0, a, b, c=None, d=None):
        if self.ld == 'quadratic':
            y = I0 * (1 - a*(1-mu) - b*((1-mu)**2))
            
        if self.ld == 'power2':
            y = I0 * (1-(a*(1-(mu**b))))
            
        if self.ld == 'claret':
            y = I0 * (1 - a*(1-(mu**0.5)) - b*(1-mu) - c*(1-(mu**1.5)) - d*(1-(mu**2)))
            
        return y
        
    def actress_run(self,wavelength, wavelength_fac, I0,c1,c2, I0_fac,c1_fac,c2_fac, c3=None,c4=None,c3_fac=None,c4_fac=None, gif_save=None, lightcurve_save=None, disk_save=None, transit_save=None, transit_disk_save=None):
        sim = ac.Simulator(faculae=[], spots=[], fac_strips=[], spot_strips=[]) #create simulation instance (explicit empty lists: Simulator's [] defaults are shared between instances)
        sim.setxsize(self.res2d)
        sim.setresolution(self.res3d) #set resolution of 3d star (number of points across diameter)
        

        # Plot HealPix map
        #import healpy as hp
    

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

        disk_fill = sim.getdiscfill(feature='fac', inc = self.inc) 

        print(f'Facular filling factor {disk_fill}')

        """
        for all following, 
        i: stellar inclination [deg] (i=90 deg = equator-on)
        N: number of datapoints
        mode: available modes - 'both' (spot+fac), 'faconly' (faculae only), 'spotonly' (spots only), 'quiet' (no features)
        """
        print(f'{wavelength * 1e10:.4f} Angstroms')
        wavelength_text = f"{wavelength * 1e10:.4f}"  # meters → Ångstroms
        wavelength_fac_text = f"{wavelength_fac * 1e10:.4f}"

        # Construct a header with all self parameters
        header_str = f"disk_fill_phase_0={disk_fill}, " + ', '.join([f"{k}={v}" for k, v in self.__dict__.items()])

        if gif_save:
            gif_save_directory = f'./outputs/gifs/{gif_save}/'
            os.makedirs(gif_save_directory, exist_ok=True)
            gif_save = f'{gif_save_directory}anim_{(wavelength_text)}.gif'
            rotate_anim = sim.rotate_anim(inc=self.inc, N=self.N, fluxunits='erg', save=gif_save, norm=False, wavelength=wavelength, outputLC=False) #create animation of rotating star and resulting lightcurve (same as above) #Dana edit making N=different from 10

        if lightcurve_save:
            lightcurve_save_directory = f'./outputs/lightcurves/{lightcurve_save}/'
            os.makedirs(lightcurve_save_directory, exist_ok=True)
            lightcurve_save = f'{lightcurve_save_directory}lc_{(wavelength_text)}.csv'
            lcr, _ = sim.rotate_lc(v_eq=self.v_eq, inc=self.inc, N=self.N, mode='faconly', wavelength=wavelength_text, returndisc=False) #calculate single-period rotational lightcurve
            np.savetxt(lightcurve_save, lcr, header=header_str)

        if disk_save:
            disk_save_directory = f'./outputs/full_disks_and_wavelength_shifts/{disk_save}/'
            os.makedirs(disk_save_directory, exist_ok=True)
            disk_save = f'{disk_save_directory}disk_inp{(wavelength_text)}.npy'
            intensity_map, wavelength_shifts = sim.rotate_lc(v_eq=self.v_eq, inc=self.inc, N=self.N, mode='faconly', wavelength=wavelength_text, returndisc=True) #calculate single-period rotational lightcurve
            np.save(disk_save, intensity_map)
            np.savetxt(f'{disk_save_directory}/wavelength_shifts_inp{(wavelength_text)}.csv', wavelength_shifts, header=header_str)

        if transit_save:
            lct = sim.transit_lc(radratio=self.rp, inc=90, b=self.b, N=self.N, mode=self.mode, a=self.a, T=self.T, phi = self.phi) #calculate transit lightcurve, with planet/star radius ratio rr
            transit_save_directory = f'./outputs/transit_lightcurves/{transit_save}/'
            os.makedirs(transit_save_directory, exist_ok=True)
            transit_save = f'{transit_save_directory}transit_lc_{(wavelength_text)}.csv'
            np.savetxt(transit_save, lct, header=header_str)

        if transit_disk_save:
            transit_disk_save_directory = f'./outputs/transit_disks/{transit_disk_save}/'
            os.makedirs(transit_disk_save_directory, exist_ok=True)
            discs, wavelength_shifts, P = sim.transit_lc(radratio=self.rp, inc=90, b=self.b, N=self.N, mode=self.mode, a=self.a, T=self.T, phi=self.phi,
                                                          returndisc=True, retP=True, v_eq=self.v_eq, wavelength=wavelength_text,
                                                          rotate_during_transit=False) #per-position disc with the planet-covered fraction blocked out; disc Doppler-shifted at v_eq but held fixed (no stellar rotation between positions)
            np.save(f'{transit_disk_save_directory}transit_disk_inp{(wavelength_text)}.npy', discs)
            np.savetxt(f'{transit_disk_save_directory}transit_positions_inp{(wavelength_text)}.csv', np.asarray(P), header=header_str)
            if wavelength_shifts is not None:
                np.savetxt(f'{transit_disk_save_directory}wavelength_shifts_inp{(wavelength_text)}.csv', wavelength_shifts, header=header_str)

    def sim_spectrum(self, hd_ld_file, fac_ld_file, gif_save=None, lightcurve_save=None, disk_save=None, transit_save=None, transit_disk_save=None):
        self.hd_ld_file = hd_ld_file
        self.fac_ld_file = fac_ld_file
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
                self.actress_run(wavelength, wavelength_fac, I0,c1,c2,I0_fac,c1_fac,c2_fac,c3=c3,c4=c4,c3_fac=c3_fac,c4_fac=c4_fac, gif_save=gif_save, lightcurve_save=lightcurve_save, disk_save=disk_save, transit_save=transit_save, transit_disk_save=transit_disk_save) #Dana edit
            else:
                self.actress_run(wavelength, wavelength_fac,I0,c1,c2,I0_fac,c1_fac,c2_fac, gif_save=gif_save, lightcurve_save=lightcurve_save, disk_save=disk_save, transit_save=transit_save, transit_disk_save=transit_disk_save) #Dana edit
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
        
        
        
