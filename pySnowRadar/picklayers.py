import numpy as np
import pywt

def picklayers(data, null_2_space, delta_fast_time_range, n_snow, ref_snow_layer=1):
    ref_scale_lin_m = 2 * null_2_space
    max_scale_lin = np.ceil(ref_scale_lin_m / delta_fast_time_range)
    lin_scale_vect_pre = np.arange(2, max_scale_lin, 1)
    lin_scale_vect = lin_scale_vect_pre[np.mod(lin_scale_vect_pre, 2) == 1]
    
    snow_layer_opl = ref_snow_layer * n_snow * 2
    ref_scale_log_m = 2 * snow_layer_opl
    max_scale_log = np.ceil(ref_scale_log_m / delta_fast_time_range)
    log_scale_vect_pre = np.arange(2, max_scale_log, 1)
    log_scale_vect = log_scale_vect_pre[np.mod(log_scale_vect_pre, 2) == 1]
    
    lin_coefs = cwt(data, pywt.Wavelet('haar'), lin_scale_vect)
    log_coefs = cwt(10 * np.log10(data), pywt.Wavelet('haar'), log_scale_vect)
    
    lin_coefs[:, 0:np.ceil(max_scale_lin / 2).astype(int)] = 0
    lin_coefs[:, -np.ceil(max_scale_lin / 2).astype(int):-1] = 0
    
    log_coefs[:, 0:np.ceil(max_scale_log / 2).astype(int)] = 0
    log_coefs[:, -np.ceil(max_scale_log / 2).astype(int):-1] = 0
    
    sum_log_coefs = np.sum(log_coefs,axis=0) / log_coefs.shape[0]
    sum_lin_coefs = np.sum(lin_coefs,axis=0) / lin_coefs.shape[0]
    
    locs_si = np.argmax(-sum_lin_coefs)
    locs_as = np.argmax(-sum_log_coefs)

    return locs_as, locs_si

def cwt(data, wavelet, scales, precision=10):
    out_coefs = np.zeros((np.size(scales), data.size))
    int_psi, x = pywt.integrate_wavelet(wavelet, precision=precision)
    step = x[1] - x[0]
    x_step = (x[-1] - x[0]) + 1
    
    j_a = [np.arange(scale * x_step) / (scale * step) for scale in scales]
    j_m = [np.delete(j, np.where((j >= np.size(int_psi)))[0]) 
           for j in j_a if np.max(j) >= np.size(int_psi)]
    coef_a = [-np.sqrt(scales[i]) 
               * np.diff(np.convolve(data, int_psi[x.astype(np.int)][::-1]))
              for (i, x) in enumerate(j_m)]
    d_a = [(coef.size - data.size) / 2 for coef in coef_a]
    out_coefs = np.asarray([coef[int(np.floor((coef.size - data.size) / 2))
                            :int(-np.ceil((coef.size - data.size) /2))] for coef in coef_a])
    return out_coefs