import os
import matplotlib.pyplot as plt
import warnings
import logging
import numpy as np
from datetime import datetime, timedelta
from scipy import signal
from shapely.geometry import box, Point, LineString

from . import matfunc, timefunc
import pandas as pd
from pyproj import Transformer
transformer = Transformer.from_crs(4326, 3413, always_xy=True)
 
C = 299792458 # Vacuum speed of light

# https://ops.cresis.ku.edu/wiki/index.php/Raw_File_Guide
CRESIS_RAW_FILE_LUT = {
    'snow': 'OIB_MAT', 'kuband': 'OIB_MAT',
    'snow2': 'OIB_MAT', 'kuband2': 'OIB_MAT',
    'snow3': 'OIB_MAT', 'kuband3': 'OIB_MAT',
    'snow4': 'OIB_MAT', 'kuband4': 'OIB_MAT',
    'snow5': 'AWI_MAT',
    'snow8': 'OIB_MAT',
    'snow9': 'OIB_MAT',
    'snow10': 'OIB_MAT'
}

LOGGER = logging.getLogger(__name__)

class SnowRadar:
    '''
    Python representation of a <proper-name-of-snowradar-instrument> dataset
    Supported formats: 
        .mat v5     (OIB)
        .mat v7     (OIB, AWI)
        .nc         (NSIDC L1b)

    Arguments:
        file_path: absolute or relative path to input SnowRadar dataset
        l_case: 'meta' or 'full' to either load just metadata, or the entire dataset

    '''
    # The OIB snow radar (2-8 GHz) data comes as matlab v5 or v7
    # The AWI snow radar data comes as matlab v7 so its closer to a HDF file
    def __init__(self, file_path, l_case):
        self.file_path = os.path.abspath(file_path)
        self.file_name = os.path.basename(self.file_path)
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(self.file_path)
        if not l_case.lower() in ['meta', 'full']:
            raise ValueError(
                "Load case: %s not understood. " % l_case +\
                "Must be one of ['meta', 'full']" 
            )
        LOGGER.debug('Loading: %s (%s)', self.file_name, l_case)
        self.load_type = l_case
        self.air_snow = None
        self.snow_ice = None
        self.epw = None # equiv_pulse_width 
        self.n2n = None # Null to Null space
        # we try to populate these in _populate_instanceattr()
        self.surface = None
        self.elv_corr = None
        # becomes True if FMCW echograms are compressed from source
        self.compressed = False
        radar_dat = matfunc.unified_loader(self)
        self._populate_instanceattr(radar_dat)

    def _populate_instanceattr(self, radar_dat):
        '''
        This is part of the unified loader mechanism that automatically accounts
        for the differences between SnowRadar source datasets
        
        Currently-supported source datasets:
        Operation IceBridge             (2016): Matlab v5 HDF
        Operation IceBridge             (2017): Matlab v7 HDF
        Alfred Wegener Institute        (2017): Matlab v7 HDF
        National Snow & Ice Data Center L1b   : NetCDF-4 NC

        Arguments:
            radar_dat: a dictionary containing data and attributes from 
                       the current SnowRadar instance's source dataset
        '''
        # scrape some metadata in order to decide how to treat the sourcefile
        self.radar_name_code = radar_dat['param_records']['radar_name']
        if not self.file_name.endswith('.nc'):
            self.data_type = CRESIS_RAW_FILE_LUT.get(self.radar_name_code, 'Unknown')
        else:
            self.data_type = 'NSIDC_NC'
        self.season = radar_dat['param_records']['season_name']
        self.mission = radar_dat['param_records']['cmd']['mission_names']
        self.day, self.segment = radar_dat['param_records']['day_seg'].split('_')
        self.day = datetime.strptime(self.day, '%Y%m%d')
        self.gps_source = radar_dat['param_records']['gps_source']

        f0 = radar_dat['param_records']['radar']['wfs']['f0']
        f1 = radar_dat['param_records']['radar']['wfs']['f1']
        try:
            fmult = radar_dat['param_records']['radar']['wfs']['fmult']
        except KeyError:
            # OIB 2019 matfiles don't contain fmult 
            warnings.warn('No value for fmult found, using fmult = 1', UserWarning)
            fmult = 1.0
        if self.data_type == 'AWI_MAT':
            # AWI mat files store 2 values for f0, f1 and fmult
            # Here we force the script to use the 2nd value
            f0 = float(f0[1])
            f1 = float(f1[1])
            fmult = float(fmult[1])
        lats = radar_dat['Latitude']
        lons = radar_dat['Longitude']
        x, y = transformer.transform(lons, lats)
        gps_times = radar_dat['GPS_time']
        # NSIDC netCDF files store UTC-time already
        # but apparently only in seconds-since-beginning-of-day! Argh!
        try:
            utc_times = np.asarray([
                np.floor((self.day + timedelta(seconds=t)).timestamp()) for t in radar_dat['UTC_time']
            ])
            utc_times = np.asarray(pd.Series(utc_times).apply(lambda x: datetime(1970,1,1) + timedelta(hours=2, seconds=x)).to_list())
        # OIB/AWI matfiles do not!
        except KeyError:
            utc_times = np.asarray([timefunc.utcleap(gps) for gps in gps_times])
            
        self.file_epoch = utc_times
        fast_times = radar_dat['Time']
        self.bandwidth = np.abs((f1 - f0) * fmult)
        self.centerfrequency = np.abs(f1 - f0) / 2 
        self.dft = fast_times[1] - fast_times[0] # delta fast time
        self.dfr = self.dft * 0.5 * C # delta fast time range
        
        
        # FOR SNOW RADAR DATA FROM LATER THAN 2016, THIS DOES NOT WORK, PARAMS ARE STORED DIFFERENTLY
        self.decimate_factor = radar_dat['param_get_heights']['get_heights']['decimate_factor'] # param_get_heights(1).get_heights(1).decimate_factor
        self.presums = radar_dat['param_get_heights']['get_heights']['presums'] #param_get_heights(1).get_heights(1).presums
        self.number_averages = self.decimate_factor * self.presums
        
        self.prf = radar_dat['param_get_heights']['radar']['prf']
        
        # load just the metadata concerning timing
        if self.load_type == 'meta':
            time_start = gps_times.min()
            time_end = gps_times.max()
            self.time_gps = np.asarray((time_start, time_end))
            self.time_utc = np.asarray((
                timefunc.utcleap(time_start),
                timefunc.utcleap(time_end)
            ))
        # load the full dataset including as much as possible
        elif self.load_type == 'full':
            data_radar = radar_dat['Data']
            elevation = radar_dat['Elevation']
            if data_radar.shape[0] == elevation.shape[0]:
                self.data_radar = np.transpose(data_radar)
            else:
                self.data_radar = data_radar
            self.elevation = elevation
            self.time_gps = gps_times
            self.time_utc = utc_times
            self.time_fast = fast_times
            self.lat = lats
            self.lon = lons
            self.x, self.y =  x, y
            
            # stored as radians, so we convert to degrees
            self.roll = np.degrees(radar_dat['Roll'])
            self.pitch = np.degrees(radar_dat['Pitch'])
            # Sometimes the surface is recorded in the matfile
            try:
                self.surface = radar_dat['Surface']
            except KeyError:
                self.surface = None
            # Sometimes there are elev corrections available
            try:
                self.elv_corr = radar_dat['Elevation_Correction'].astype(int)
            except KeyError:
                self.elv_corr = None
            # Check if the FMCW echograms are compressed in the matfile
            try:
                self.trunc_bins = radar_dat['Truncate_Bins'].astype(int)
            except KeyError:
                self.trunc_bins = None
            # Check if the file has previously been elevation corrected
            # TODO: check type of output
            try:
                self.elev_corrected = np.any(radar_dat['param_records']['get_heights']['elev_correction'])
            except KeyError:
                self.elev_corrected = False

        # Geospatial boundary box
        self.extent = np.hstack((
            lons.min(), lats.min(),
            lons.max(), lats.max()
        )).ravel()
        self.poly = box(*self.extent)
        # Simplified polyline of track
        points_xy = [Point(xy) for xy in zip(x, y)]
        line_xy = LineString(points_xy)
        
        points_lonlat = [Point(xy) for xy in zip(lons, lats)]
        line_lonlat = LineString(points_lonlat)
        # try to simplify the line with a tight tolerance (degrees)
        self.line_xy = line_xy.simplify(tolerance=1e-6)
        self.line_lonlat = line_lonlat.simplify(tolerance=1e-6)
        

    def get_surface(self, smooth=True, window = 5):
        '''
        Simple surface tracker based on maximum
        This should be refined and is largely a place holder 
        TODO: surf_time is broken unless the time axis is interpolated
        
        '''
        # identify if any all-nan traces exist
        all_nans =  np.where(np.apply_along_axis(np.all, axis=0, arr=np.isnan(self.data_radar)))[0]
        if len(all_nans) == 0:
            surf_bin = np.nanargmax(self.data_radar, axis=0)
        else:
            # make a copy of the data array so we don't modify it 
            arr = self.data_radar.copy()
            # replace nan-traces with 0
            arr[:, all_nans] = 0
            surf_bin = np.nanargmax(arr, axis=0)

        if smooth:
            shape = surf_bin.shape[:-1] + (surf_bin.shape[-1] - window + 1, window)
            strides = surf_bin.strides + (surf_bin.strides[-1],)
            surf_bin = np.median(np.lib.stride_tricks.as_strided(surf_bin, shape=shape, strides=strides),1).astype(int)

        surf_time = np.interp(surf_bin,np.arange(0,len(self.time_fast)),self.time_fast)
        return surf_bin , surf_time
    
    def get_bounds(self, m_above=None, m_below=5):
        '''
        Get bin numbers where there is valid data (non-nan)
        A threshold can be supplied

        Arguments:
            m_above: bin padding above the signal (?)
            m_below: bin padding below the signal (?)

        Outputs:
            null_lower: the lower bin number bound for use in data-subsetting
            null_upper: the upper bin number bound for use in data-subsetting
        
        '''
        if m_above:
            null_lower = self.surf_bin.max() + (m_below / self.dfr).astype(int)
            null_upper = self.surf_bin.min() - (m_above / self.dfr).astype(int)
        else:
            null_space = np.argwhere(np.isnan(self.data_radar))[:,0]
            null_upper = null_space[null_space < self.surf_bin.min()].min()
            null_lower = null_space[null_space > self.surf_bin.max()].max()
        return null_lower, null_upper
    
    def calcpulsewidth(self, oversample_num=1000, num_nyquist_ts=100):
        '''
        Using the current SnowRadar instance's radar bandwidth (self.bandwidth),
        calculate and set the values for the null-to-null pulse width (self.n2n)
        and the equivalent pulse width (self.epw) 

        The windowing process used is `signal.hann`

        Arguments:
            oversample_num: the bin-amount to oversample the nyquist by
            num_nyquist_ts: the number of nyquist timestamps to use when windowing(?)
        
        '''
        # Time Vector
        nyquist_sf = 2 * self.bandwidth
        fs = nyquist_sf * oversample_num 
        time_step = 1 / fs 
        max_time = num_nyquist_ts * oversample_num * time_step
        time_vect = np.linspace(-max_time, max_time, ((max_time*2)/time_step).astype(int))  
    
        # Frequency domain object
        half_bandwidth = self.bandwidth / 2
        n_FFT = len(time_vect)
        f = fs * np.linspace(-0.5, 0.5, n_FFT)
        n_band_points = np.sum(np.abs(f) <= half_bandwidth)
    
        # Create spectral window 
        spectral_win = signal.hann(n_band_points, sym = False)
    
        # Frequency domain processing
        # JK: Need to be careful here, f becomes an array if bandwidth is as well.
        # Change it to use f.shape?
        freq_domain_signal = np.zeros(len(f)) 
        freq_domain_signal[np.abs(f) < half_bandwidth] = spectral_win
        shift_freq_domain_signal = np.fft.ifftshift(freq_domain_signal)
        time_domain_signal = np.fft.ifft(shift_freq_domain_signal) * n_FFT
        time_sig = np.fft.fftshift(time_domain_signal)
        power_signal = np.abs(time_sig ** 2)
        power_signal_norm = power_signal / np.max(power_signal)
        max_idx = np.argmax(power_signal_norm)
    
        # Calc the equivalent pulse width
        equiv_pulse_width_val = np.sum(power_signal_norm)
        equiv_pulse_width_time = equiv_pulse_width_val * time_step
        self.epw = equiv_pulse_width_time * C
    
        # Calc null-to-null pulse width
        with np.errstate(divide = 'ignore'):
            invert_l10_power = -10 * np.log10(power_signal_norm)
        peak_idx, _ = signal.find_peaks(invert_l10_power)
        closest_peaks = np.sort(np.abs(peak_idx - max_idx))
    
        null_2_width = 2 * np.mean(closest_peaks[0:1])
        null_2_time = null_2_width * time_step
        self.n2n = null_2_time * C
    
    def decompress_data(self):
        '''
        Ported to Python by Josh King from CRESIS uncompress_echogram MatLab code by John Paden:
        https://data.cresis.ku.edu/data/loader/uncompress_echogram.m

        For data that arrives already-compressed and with the self.trunc_bins data attribute,
        decompress the radar data as well as the time array.

        Outputs:
            data_radar_decomp: 1D Numpy array of decompressed radar data
            time_decomp: 1D Numpy array of decompressed time data
        '''
        Nz = self.elv_corr.max()
        Nt = Nz + len(self.trunc_bins)
        data_radar_decomp = np.pad(self.data_radar, [(Nz,0 ), (0, 0)], 'constant', constant_values=(np.nan))
         
        for rline in np.arange(0,data_radar_decomp.shape[1]):
            data_radar_decomp[:, rline] = np.roll(data_radar_decomp[:, rline], -self.elv_corr[rline])
        if len(self.time_fast) == len(self.trunc_bins):
            t0 = self.time_fast[0] - Nz*self.dft
        else:
            t0 =  self.time_fast[self.trunc_bins[0]] - Nz * self.dft
        
        time_decomp = t0 + self.dft*np.arange(0, Nt-1)
        return data_radar_decomp, time_decomp
    
    def elevation_compensation(self):
        '''
        Adapted by Josh King from CRESIS elevation_compensation.m by John Paden
        https://github.com/kingjml/pyWavelet/blob/master/pyWavelet/legacy/elevation_compensation.m

        Outputs:
            radar_comp: 2D numpy array of elevation-compensated radar data 
            elev_axis: elevation axis based on bin timing and an assumption of permittivity
        '''        
        max_elev = np.max(self.elevation)
        delta_range = max_elev - self.elevation
        delta_time = self.time_fast[1] - self.time_fast[0]
        delta_bins = np.round(
            delta_range / (C / 2) / delta_time
        ).astype(int)
        zero_pad_len = np.max(np.abs(delta_bins)).astype(int)
       
        radar_comp = np.concatenate((
                self.data_radar, 
                np.zeros((zero_pad_len, self.data_radar.shape[1]))),
                axis=0
            )

        time_comp = self.time_fast[0] + delta_time * np.arange(0,self.data_radar.shape[0])

        for idx, dbin in enumerate(delta_bins):
            radar_comp[:, idx] = np.roll(radar_comp[:, idx], dbin)
            #self.elevation[idx] = self.elevation[idx] + dbin*delta_time*C/2
            #self.surface[idx] = self.surface[idx] + dbin * delta_time
    
        self.elev_corrected = True
        return radar_comp, time_comp

    def plot_quicklook(self, ylim=None):
        '''
        Generic plotting function to visualize the radar data for the 
        current SnowRadar object instance

        Arguments:
            ylim: customize the upper bound of the plot
        '''
        with np.errstate(divide='ignore', invalid='ignore'):
            radar_sub = 10 * np.log10(self.data_radar)
        fig, ax = plt.subplots(figsize=(9,7))
        im = ax.imshow(radar_sub, cmap='gist_gray')
        ax.set_title(
            f'{self.file_name} ({self.data_type})',
            fontdict={'size':'x-large'}
        )
        if ylim:
            ax.set_ylim(ylim)
        ax.set_aspect('auto')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        plt.show()

    def as_dict(self):
        '''Generic metadata for current SnowRadar instance'''
        return {
            'fname': self.file_name,
            'fpath': self.file_path,
            'l_case': self.load_type,
            'tstart': self.time_utc.min(),
            'tend': self.time_utc.max(),
            'poly': self.poly.wkt,
            'line': self.line.wkt
        }

    def __str__(self):
        '''Fancy string override'''
        return f'{self.data_type} Datafile: {self.file_name}'
