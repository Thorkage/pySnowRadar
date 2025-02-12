import numpy as np
import pywt
from scipy.signal import find_peaks

def Wavelet_TN(data, null_2_space, delta_fast_time_range, n_snow, ref_snow_layer, cwt_precision, **kwargs):
    '''
    Function to detect 2 interface layers from a given SnowRadar signal:
        Air-Snow interface
        Snow-Ice Interface

    Uses the Continuous Wavelet Transform (cwt) method originally developed
    by Thomas Newman

    Arguments:
        data: 1D radar data array
        null_2_space: trough to trough distance
        delta_fast_time_range: radar bin range in m
        n_snow: the refractive index of snow
        ref_snow_layer: reference snow depth in m (default 1)
        cwt_precision: precision arg for cwt (default 10)

    Outputs:
        locs_as: Air-snow interface bin index (integer)
        locs_si: Snow-ice interface bin index (integer)

    '''
    ref_scale_lin_m = 2 * null_2_space
    max_scale_lin = np.ceil(ref_scale_lin_m / delta_fast_time_range)
    lin_scale_vect = np.arange(2, max_scale_lin, 1)[1::2]
    
    snow_layer_opl = ref_snow_layer * n_snow * 2
    ref_scale_log_m = 2 * snow_layer_opl
    max_scale_log = np.ceil(ref_scale_log_m / delta_fast_time_range)
    
    # !!! CHANGED TO LIN SCALE VECT INSTEAD OF LOG SCALE VECT !!!
    # log_scale_vect = np.arange(2, max_scale_lin, 1)[1::2] # !!! CHANGED TO LIN SCALE VECT INSTEAD OF LOG SCALE VECT !!!
    log_scale_vect = np.arange(2, max_scale_log, 1)[1::2] # !!! CHANGED TO LIN SCALE VECT INSTEAD OF LOG SCALE VECT !!!
    
    # !!! CHANGED TO LIN SCALE VECT INSTEAD OF LOG SCALE VECT !!!
    
    
    lin_coefs = cwt(data, pywt.Wavelet('haar'), lin_scale_vect, cwt_precision)
    log_coefs = cwt(10 * np.log10(data), pywt.Wavelet('haar'), log_scale_vect, cwt_precision)
    # log_coefs = cwt(np.log(data), pywt.Wavelet('haar'), log_scale_vect, cwt_precision) #!!! CHANGED TO LN INSTEAD OF 10log10 !!!!
    # 
    # Negating edge effects here, we use half the max scale on either end
    # Some discussion is needed on this approach because it can sometimes lead to weird picks
    lin_coefs[:, 0:np.ceil(max_scale_lin/2).astype(int)] = 0
    lin_coefs[:, -np.ceil(max_scale_lin/2).astype(int):] = 0

    log_coefs[:, 0:np.ceil(max_scale_log/2).astype(int)] = 0
    log_coefs[:, -np.ceil(max_scale_log/2).astype(int):] = 0
    
    sum_log_coefs = np.sum(log_coefs,axis=0) / log_coefs.shape[0]
    sum_lin_coefs = np.sum(lin_coefs,axis=0) / lin_coefs.shape[0]
    
    locs_si = np.argmax(-sum_lin_coefs) 
    locs_as = np.argmax(-sum_log_coefs)
    
    # locs_as_1per = (sum_log_coefs > sum_log_coefs[locs_as] * 0.99) #& (sum_log_coefs < sum_log_coefs[locs_as] * 1.01)
    
    
    #5% of the max value (left and right) for air-snow interface
    locs_as_left = np.argmax(-sum_log_coefs > np.max(-sum_log_coefs) * 0.99)
    locs_as_right = len(sum_log_coefs) - 1 - np.argmax((-sum_log_coefs > np.max(-sum_log_coefs) * 0.99)[::-1])
    
    #5% of the max value (left and right) for snow-ice interface
    locs_si_left = np.argmax(-sum_lin_coefs > np.max(-sum_lin_coefs) * 0.99)
    locs_si_right = len(sum_lin_coefs) - 1 - np.argmax((-sum_lin_coefs > np.max(-sum_lin_coefs) * 0.99)[::-1])  
    
    #how large is the found peak in snow-ice compared to next highest peak in the signal
    # peaks, _ = find_peaks(abs(ds['wavelet_coefs_lin'][:,ind]), height=0.5*np.max(abs(ds['wavelet_coefs_lin'][:,ind].values)))
    peaks,_ = find_peaks(-sum_lin_coefs, height=0.25*np.max(-sum_lin_coefs))
    if len(peaks) > 1:
        si_ratio = ((-sum_lin_coefs[peaks]) / np.max(-sum_lin_coefs))[1]
    else:
        si_ratio = 1
        
    return locs_as, locs_as_left, locs_as_right, locs_si, locs_si_left, locs_si_right, sum_log_coefs, sum_lin_coefs, si_ratio #locs_as_1per



def Wavelet_JK(data, scale_vect, **kwargs):
    log_gaus1_coefs, _ =  pywt.cwt(10 * np.log10(data),scale_vect,'gaus1')
    log_gaus1_coefs[:, 0:np.ceil(scale_vect[-1]*2).astype(int)] = np.nan
    log_gaus1_coefs[:, -np.ceil(scale_vect[-1]*2).astype(int):] = np.nan
    sum_log_gaus1_coefs = np.sum(log_gaus1_coefs,axis=0) / log_gaus1_coefs.shape[0]
    locs_as = np.nanargmin(sum_log_gaus1_coefs)

    lin_gaus2_coefs, _ = pywt.cwt(data,scale_vect,'gaus2')
    lin_gaus2_coefs[:, 0:np.ceil(scale_vect[-1]*2).astype(int)] = np.nan
    lin_gaus2_coefs[:, -np.ceil(scale_vect[-1]*2).astype(int):] = np.nan
    sum_lin_gaus2_coefs = np.sum(lin_gaus2_coefs,axis=0) / lin_gaus2_coefs.shape[0]
    locs_si = np.nanargmax(sum_lin_gaus2_coefs)

    return locs_as, locs_si

def cwt(data, wavelet, scales, precision):
    '''
    Implementation of the Continuous Wavelet Transform

    Arguments:
        data: preprocessed snowradar signal data
        wavelet: the specific Wavelet to use (currently the Haar wavelet)
        scales: 
        precision: precision to apply to wavelet operations (default 10)

    Outputs:
        out_coefs:(?)
    '''
    out_coefs = np.zeros((np.size(scales), data.size))
    int_psi, x = pywt.integrate_wavelet(wavelet, precision=precision)
    step = x[1] - x[0]
    x_step = (x[-1] - x[0]) + 1
    
    j_a = [np.arange(scale * x_step) / (scale * step) for scale in scales]
    j_m = [np.delete(j, np.where((j >= np.size(int_psi)))[0]) 
           for j in j_a if np.max(j) >= np.size(int_psi)]
    coef_a = [-np.sqrt(scales[i]) 
               * np.diff(np.convolve(data, int_psi[x.astype(int)][::-1]))
              for (i, x) in enumerate(j_m)]
    out_coefs = np.asarray([coef[int(np.floor((coef.size - data.size) / 2))
                            :int(-np.ceil((coef.size - data.size) /2))] for coef in coef_a])
    return out_coefs
