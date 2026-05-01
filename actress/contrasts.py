import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simps
from scipy.optimize import curve_fit
from scipy.optimize import minimize

def resort(t,f):
    t = np.array(t)
    f = np.array(f)
    array = np.column_stack([t,f])
    new_array= array[np.argsort(array[:,0])]
    new_t = new_array[:,0]
    new_f = new_array[:,1]
    return new_t, new_f

def wav_nm_to_m(wav):
    new = wav*1e-9
    return new

def cgs_to_si(data):
    new = (data*1e-7)*(100**2)
    return new

def intensity_si(data):
    intensity_si = []
    for i in range(1,10):
        intensity1_cgs = data[:,i]
        intensity1_si = cgs_to_si(intensity1_cgs)
        intensity_si.append(intensity1_si)
    return intensity_si

def wav_limits(wav, wav_lower, wav_higher):
    val_low = []
    val_high = []
    for i in range(0,len(wav)):
        new_low = abs(wav_lower - wav[i])
        new_high = abs(wav_higher - wav[i])
        val_low.append(new_low)
        val_high.append(new_high)
    indx_low = val_low.index(min(val_low))
    indx_high = val_high.index(min(val_high))
    return indx_low, indx_high

def wav_to_freq(wav):
    freq = 3e8/wav
    return freq

def bolo_intensity(freq,I):
    area_total = []
    for i in range(0,len(I[0:])):
        freq_new, intensity_si_new = resort(freq,I[i])
        area = simps(intensity_si_new,freq_new,)
        area_total.append(area)
    return area_total

def costheta_to_ster(mu):
    ster=(np.pi)*(1-(mu**2))
    return ster

def grad_intrcpt_inverse(x,y):
    m = (y[len(y)-1]-y[len(y)-2])/(x[len(x)-1]-x[len(x)-2])
    c = y[len(y)-1] - (m*x[len(x)-1])
    return m,c

def ld(x,a,b,I0,ld_law,c=None,d=None):
    mu = np.sqrt(1-(x/np.pi))
    if ld_law == 'quadratic':
        y = I0 * (1 - a*(1-mu) - b*((1-mu)**2))
    elif ld_law == 'power2':
        y = I0 * (1-(a*(1-(mu**b))))
    elif ld_law == 'custom':
        h1 = (1-a)/(1-(b/(1-a)))
        h2 = np.log10((1-a)/b)/np.log10(1.5)
        y = I0 * (1-(h1*(1-(mu**h2)))) #to make the function work I've relabelled a<->h1 and b<->h2
    elif ld_law == 'claret':
        y = I0 * (1 - a*(1-(mu**0.5)) - b*(1-mu) - c*(1-(mu**1.5)) - d*(1-(mu**2)))
    elif ld_law == 'empty':
        y = I0 * (1-(a*(1-(mu**b)))) #a is 0
    return y

def ld_fit(X,Y,I0,ld_law):
    xData = np.asarray(X)
    yData = np.asarray(Y)

    sigma = np.ones(len(xData))
    
    def model_func(x,a,b,I0=I0,ld_law=ld_law):
        return ld(x,a,b,I0=I0,ld_law=ld_law)
    
    def model_func4(x,a,b,c,d,I0=I0,ld_law=ld_law):
        return ld(x,a,b,I0=I0,ld_law=ld_law,c=c,d=d)

    if ld_law == 'claret':
        popt, pcov = curve_fit(model_func4, xData, yData, p0=(0.67595786, 0.7330585, 0.67595786, 0.7330585), sigma=sigma, maxfev=10000)

        a_best = popt[0]
        b_best = popt[1]
        c_best = popt[2]
        d_best = popt[3]

        return a_best, b_best, c_best, d_best
    
    else:
        popt, pcov = curve_fit(model_func, xData, yData, p0=(0.67595786, 0.7330585), sigma=sigma, maxfev=10000)

        a_best = popt[0]
        b_best = popt[1]

        return a_best, b_best


def validate_inputs(fac_file, phot_file, wav_lower, wav_higher, graphs=False):
    # Using pandas to seamlessly handle identical data arrays regardless of header comments
    fac_data = pd.read_csv(fac_file, sep=r'\s+', comment='#').values
    phot_data = pd.read_csv(phot_file, sep=r'\s+', comment='#').values
    if graphs == True:
        
        plt.figure(figsize=(16, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(phot_data[:, 0], fac_data[:, 0],  label='Wavelengths')
        plt.xlabel('Photosphere Wavelengths')
        plt.ylabel('Faculae Wavelengths')
        plt.title('Faculae vs Photosphere Wavelengths')
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.plot(phot_data[:, 0], phot_data[:, 1], alpha=.5, label='i_1.0 photosphere')
        plt.plot(fac_data[:, 0], fac_data[:, 1], alpha=.5, label='i_1.0 facular')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensities')
        plt.title('Faculae vs Photosphere Intensities')
        plt.legend()
        plt.xlim(wav_lower * 1e9, wav_higher * 1e9)
        plt.grid(True)

        plt.subplot(2, 3, 3)
        plt.plot(phot_data[:, 0], phot_data[:, 6], alpha=.5, label='i_0.6 photosphere')
        plt.plot(fac_data[:, 0], fac_data[:, 6], alpha=.5, label='i_0.6 facular')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensities')
        plt.title('Faculae vs Photosphere Intensities')
        plt.legend()
        plt.xlim(wav_lower * 1e9, wav_higher * 1e9)
        plt.grid(True)

        plt.subplot(2, 3, 4)
        plt.plot(phot_data[:, 0], phot_data[:, 9], alpha=.5, label='i_0.2 photosphere')
        plt.plot(fac_data[:, 0], fac_data[:, 9], alpha=.5, label='i_0.2 facular')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensities')
        plt.title('Faculae vs Photosphere Intensities')
        plt.legend()
        plt.xlim(wav_lower * 1e9, wav_higher * 1e9)
        plt.grid(True)

        plt.subplot(2, 3, 5)
        plt.plot(phot_data[:,0], (fac_data[:,1] - phot_data[:,1]) / phot_data[:,1], label='Contrast Disk center')
        plt.xlim(wav_lower * 1e9, wav_higher * 1e9)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Contrast')
        plt.title('Contrast vs Wavelength')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    if fac_data.shape[0] != phot_data.shape[0]:
        raise ValueError(f"File lengths do not match! {fac_file} has {fac_data.shape[0]} rows and {phot_file} has {phot_data.shape[0]} rows.")

    if not np.allclose(fac_data[:, 0], phot_data[:, 0], atol=1e-10):
        mismatch_idx = np.where(~np.isclose(fac_data[:, 0], phot_data[:, 0], atol=1e-10))[0][0]
        raise ValueError(f"Wavelengths in column 1 do not exactly match! First mismatch found at index {mismatch_idx}: {fac_data[mismatch_idx, 0]} vs {phot_data[mismatch_idx, 0]}")
    

def validate_outputs(fac_out_file, phot_out_file, graphs = False):
    # Using pandas to seamlessly handle identical data arrays regardless of header comments
    fac_out = pd.read_csv(fac_out_file, sep=r'\s+', comment='#').values
    phot_out = pd.read_csv(phot_out_file, sep=r'\s+', comment='#').values

    if fac_out.shape[0] != phot_out.shape[0]:
        raise ValueError(f"Output lengths do not match: {fac_out_file} has {fac_out.shape[0]} rows, {phot_out_file} has {phot_out.shape[0]} rows.")
        
    if not np.array_equal(fac_out[:, 0], phot_out[:, 0]):
        mismatch_idx = np.where(fac_out[:, 0] != phot_out[:, 0])[0][0]
        raise ValueError(
            f"Output wavelengths (column 1) do not match! "
            f"First mismatch at index {mismatch_idx}: "
            f"{fac_out[mismatch_idx, 0]} vs {phot_out[mismatch_idx, 0]}"
        )
    if graphs == True:
        fac_intensity = fac_out[:, 1]
        phot_intensity = phot_out[:, 1]
        wavelength = fac_out[:, 0]

        contrast = ( fac_intensity - phot_intensity ) / phot_intensity

        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(wavelength, fac_intensity, label='Faculae Disk Center', alpha=0.7)
        plt.plot(wavelength, phot_intensity, label='Photosphere Disk Center', alpha=0.7)
        plt.xlabel('Wavelength')
        plt.ylabel('Intensity')
        plt.title('Disk Center Intensities')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(wavelength, contrast)
        plt.xlabel('Wavelength')
        plt.ylabel('Contrast')
        plt.title('Contrast')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


def contrast(data, wav_lower, wav_higher, N, ld_law, save=None, graphs=False):
    # Using pandas to gracefully skip headers and comment lines, regardless of file format
    file = pd.read_csv(data, sep=r'\s+', comment='#').values
    wavelength = file[:,0]
    wavelength_m = wav_nm_to_m(wavelength)
    indx_low, indx_high = wav_limits(wavelength_m,wav_lower,wav_higher)
    intensity = intensity_si(file)
    n = (indx_high - indx_low)/N
    #print(n, indx_low, indx_high)
    
    wavs = []
    I0s = []
    a_bests = []
    b_bests = []
    c_bests = []
    d_bests = []
    wavelength_m=list(wavelength_m)
    for i in range(indx_low,indx_high+1):
        if i%round(n) == 0:
            delta_freq = ((-(3e8/(wavelength_m[i+1]))+(3e8/(wavelength_m[i])))/2) + ((-(3e8/(wavelength_m[i]))+(3e8/(wavelength_m[i-1])))/2)
    
            I = []
            for j in range(0,9):
                new = intensity[j][i] * delta_freq
                I.append(new)
            I = np.asarray(I)
    
            area_total = I #dana question, what does area_total hold? how is it used in ld_fit. Where does it account for the intensity across this entire wavelength range?
            costheta = np.array([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2])
            ster = costheta_to_ster(costheta)
    
            if ld_law == 'claret':
                a_best, b_best, c_best, d_best = ld_fit(ster,area_total,I0=area_total[0],ld_law=ld_law)

            elif ld_law == 'empty':
                a_best, b_best = ld_fit(ster,area_total,I0=area_total[0],ld_law=ld_law)
                a_best = 0
                b_best = b_best
                
            else:
                a_best, b_best = ld_fit(ster,area_total,I0=area_total[0],ld_law=ld_law)
    
            wavs.append(wavelength_m[i])
            I0s.append(area_total[0])
            a_bests.append(a_best)
            b_bests.append(b_best)
            
            if ld_law == 'claret':
                c_bests.append(c_best)
                d_bests.append(d_best)

            if graphs is True:
                fig, ax = plt.subplots(2, figsize=(15, 15), sharex=True)
                x = np.linspace(0,np.pi,100000)
                mu = np.sqrt(1-(x/np.pi))
                if ld_law == 'claret':
                    y_best = ld(x,a_best,b_best,I0=area_total[0],ld_law=ld_law,c=c_best,d=d_best)
                    
                else:
                    y_best = ld(x,a_best,b_best,I0=area_total[0],ld_law=ld_law)
                    
                ax[0].plot(costheta,area_total,'x',label='Input data')
                ax[0].plot(mu,y_best,'-',label='LD fitted $I_{\\lambda}(\\mu)$')
                ax[0].set_ylabel('$I_{\\mu}$ (W/$m^{2}$/ster)')
                ax[0].invert_xaxis()
                ax[0].legend(loc='upper right')

                mu_val = []
                y_best_val = []
                for k in range(0,len(costheta)):
                    val = []
                    for j in range(0,len(mu)):
                        new = abs(mu[j]-costheta[k])
                        val.append(new)
                    mu_val.append(mu[val.index(min(val))])
                    y_best_val.append(y_best[val.index(min(val))])
                res = np.asarray(area_total) - np.asarray(y_best_val)
                ax[1].plot(costheta,res,'x')
                ax[1].set_ylabel('residual (Photons/s/$m^{2}$/ster)')
                ax[1].set_xlabel('$\\mu$')

                                
                csv_path = './limb_darkening_laws/'+str(ld_law)+'_profile'+str(save[:4])+str(wavelength[i])+'nm'+'.csv' ##DANA EDIT
                #os.makedirs('./limb_darkening_laws', exist_ok=True)
                data_to_save = np.column_stack((costheta, area_total))
                header = 'costheta,area_total'
                np.savetxt(csv_path, data_to_save, delimiter=',', header=header, comments='') ##DANA EDIT^^

                
                plt.savefig('./limb_darkening_laws/'+str(ld_law)+'_profile'+str(save[:4])+str(wavelength[i])+'nm'+'.png')
                plt.close()

    if len(wavs) == 0:
        raise RuntimeError(f"Error: No data points were generated for {save}. Check if wav_lower and wav_higher are within the data range, or if N is causing the spacing condition to fail.")

    if ld_law == 'claret':
        v = np.column_stack((wavs,I0s,a_bests,b_bests,c_bests,d_bests))

    # if ld_law = None:
    
        
    else:
        v = np.column_stack((wavs,I0s,a_bests,b_bests))
    
    if save is not None:            
        np.savetxt(save, v) 
        print("Saved to",save)


def teff(data,ld_law=None):
    file = pd.read_csv(data, sep=r'\s+', comment='#').values
    wavelength = file[:,0]
    wavelength_m = wav_nm_to_m(wavelength)
    freq = wav_to_freq(wavelength_m)
    intensity = intensity_si(file)
    area_total = bolo_intensity(freq,intensity)

    costheta = np.array([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2])
    ster = costheta_to_ster(costheta)

    if ld_law is None: #Linearly interpolate to 90degrees
        m,c = grad_intrcpt_inverse(ster,area_total)
        area10 = (m*np.pi)+c
        area_total.append(area10)
        ster = list(ster)
        ster.append(np.pi)
        area_final = simps(area_total,ster,)
        sigma = 5.67e-8
        temp = ((area_final)/sigma)**0.25
        print("The effective temperature is",temp,"K")

    elif ld_law is not None:
        x = np.linspace(0,np.pi,100000)
        mu = np.sqrt(1-(x/np.pi))
        if ld_law == 'claret':
            a_best, b_best, c_best, d_best = ld_fit(ster,area_total,I0=area_total[0],ld_law=ld_law)
            y_best = ld(x,a_best,b_best,I0=area_total[0],ld_law=ld_law,c=c_best,d=d_best)           
        else:
            a_best, b_best = ld_fit(ster,area_total,I0=area_total[0],ld_law=ld_law)
            y_best = ld(x,a_best,b_best,I0=area_total[0],ld_law=ld_law)
        area_final = simps(y_best,x,) #area_final is area under y_best vs solid angle(ster) as x is solid angle. I plotted above in terms of mu just so it's easier to see.
        sigma = 5.67e-8
        temp = ((abs(area_final)/sigma))**0.25
        print("The effective temperature is",temp,"K")



        
