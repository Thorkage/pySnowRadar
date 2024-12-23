import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
from pySnowRadar import ATM, SnowRadar, algorithms
from shapely import Polygon
from scipy.signal import find_peaks
import haversine
from pyproj import Transformer, transform

C = 299792458 # Vacuum speed of light

LOGGER = logging.getLogger(__name__)

def geo_filter_insitu_sites(path, site, input_sr_data):
    # '''
    # Given a list of SnowRadar datafiles (.mat, .h5, .nc), filter out
    # any files whose bounding geometry intersect with land

    # Landmask is based on NaturalEarth 1:10m Cultural v4.1.0 (Canada, Greenland, and USA)
    # http://www.naturalearthdata.com/

    # Arguments:
    #     input_sr_data: list of supported SnowRadar data files

    # Output:
    #     subset of input_sr_data where no land intersections occur
    # '''
    # Drop all data that intersects with land features
    # for site in sites:
    land = gpd.read_file(os.path.join(path, f'extent_{site}.shp'))
    x = np.array(list(land.loc[0, 'geometry'].exterior.xy[0])) - 360
    y = np.array(list(land.loc[0, 'geometry'].exterior.xy[1]))
    land.loc[0,'geometry'] = Polygon(zip(x,y))
    land = land.set_crs('EPSG:4326')
    
    # Load the datafiles in 'meta' mode to just scrape the simplified track line
    sr_meta = [SnowRadar(sr, 'meta') for sr in input_sr_data]
    sr_gdf = gpd.GeoDataFrame(
        data={'file': [sr.file_path for sr in sr_meta]}, 
        geometry=[sr.line for sr in sr_meta], 
        crs='epsg:4326'
    )
    sr_gdf = sr_gdf.loc[gpd.sjoin(sr_gdf, land, how='inner').index]
    
    if len(sr_gdf) < 1:
        LOGGER.warning('No suitable datafiles left after geospatial filtering')
        return []
        
    
    return sr_gdf.file.tolist()



def geo_filter(input_sr_data):
    '''
    Given a list of SnowRadar datafiles (.mat, .h5, .nc), filter out
    any files whose bounding geometry intersect with land

    Landmask is based on NaturalEarth 1:10m Cultural v4.1.0 (Canada, Greenland, and USA)
    http://www.naturalearthdata.com/

    Arguments:
        input_sr_data: list of supported SnowRadar data files

    Output:
        subset of input_sr_data where no land intersections occur
    '''
    # Drop all data that intersects with land features
    land = gpd.read_file('/vsizip/' + str(Path(__file__).parent / 
                         'data' / 'natearth' / 
                         'ne_10m_admin_0_countries_northamerica.zip'))
    # Load the datafiles in 'meta' mode to just scrape the simplified track line
    sr_meta = [SnowRadar(sr, 'meta') for sr in input_sr_data]
    sr_gdf = gpd.GeoDataFrame(
        data={'file': [sr.file_path for sr in sr_meta]}, 
        geometry=[sr.line for sr in sr_meta], 
        crs='epsg:4326'
    )
    sr_gdf = sr_gdf.drop(
        gpd.sjoin(sr_gdf, land, how='inner').index
    )
    if len(sr_gdf) < 1:
        LOGGER.warning('No suitable datafiles left after geospatial filtering')
        return []
    return sr_gdf.file.tolist()

def extract_layers(data_path, picker=algorithms.Wavelet_TN, params=None, dump_results=False, overwrite=True, path='./dump'):
    '''
    For a given SnowRadar datafile, estimate the air-snow and snow-ice interfaces
    using the supplied picker and snow density

    Arguments:
        data_path: file path to input SnowRadar data file
        picker: Which picker algorithm to apply (default is algorithms.Wavelet_TN)
        params: A dictorary of expected parameters to pass to the picker
        dump_results: whether or not to save the dataframe to a local csv file under ./dump/

    Output:
        A pandas dataframe with the following columns:
            'src': the name of the source SnowRadar data file
            'picker': the name of the picker algo
            'lat': latitude of trace
            'lon': longitude of trace
            'n_snow': the refractive index used during layer picking
            'b_ref': the reference bin considered as 0 in the origional file
            'b_as': the picked air-snow interface layer
            'b_si': the picked snow-ice interface layer
            'snow_depth': estimated snow depth based on picked layers
            'params': a dict of all input and generated params
    '''
    
    
    
    # Check that the picker passed exists
    if not(picker in algorithms.available_pickers()):
        raise ValueError(
            'Invalid picker name:' % picker.__name__
        )

    # TODO: Modify to allow refractive index (n_snow) as an alternative
    if 'snow_density' not in params:
        raise ValueError(
            'Snow density or refractive index input required for all pickers'
        )
    elif (not(0.1 <= params['snow_density'] <= 0.4)):
        raise ValueError(
            'Invalid snow density passed: %.3f (Must be between 0.1 and 0.4)' % params['snow_density']
        )
    
    # Load radar data 
    radar_dat = SnowRadar(data_path, 'full')
    radar_dat.surf_bin, radar_dat.surface = radar_dat.get_surface()
    radar_dat.calcpulsewidth()
    
    if dump_results: 
        outpath = Path(path)
        # outname = Path(data_path).stem + '_' + str(radar_dat.time_utc[0]).split('.')[0] + '_'  + str(radar_dat.time_utc[-1]).split('.')[0] + '_'+ '.csv'
        outname = Path(data_path).stem  + '.nc' #+ '_' + str(radar_dat.time_utc[0]).split('.')[0] + '_'  + str(radar_dat.time_utc[-1]).split('.')[0] + '_'+
        
        outfile = outpath / outname
        if outfile.exists() and overwrite == False:
            LOGGER.warning('File exists for %s. Skipping processing....', Path(data_path).name)
            result = pd.read_csv(str(outfile), index_col=0)
            return result
        
    # Subset radar traces to reduce computational load
    # TODO: Should we allow the subset bounds to be user defined?
    lower, upper = radar_dat.get_bounds(m_above=5)
    radar_sub = radar_dat.data_radar[upper:lower, :]
   
    # Calc or init other necessary params
    params['n_snow'] = np.sqrt((1 + 0.51 * params['snow_density']) ** 3)
    params['null_2_space']  = radar_dat.n2n
    params['delta_fast_time_range'] = radar_dat.dfr
    
    # Apply the picker to the file, trace by trace
    try:
        airsnow, snowice, log_coefs, lin_coefs = [], [], [], []
        for trace in radar_dat.data_radar.T:
            asnow, sice, log_coef, lin_coef = picker(trace, **params)
            airsnow.append(asnow)
            snowice.append(sice)
            log_coefs.append(log_coef)
            lin_coefs.append(lin_coef)
        
        airsnow = np.array(airsnow)
        snowice = np.array(snowice)
        log_coefs = np.array(log_coefs).T
        lin_coefs = np.array(lin_coefs).T
        
    
    except:
        # We catch and log the exception
        errtype, errval, _ = sys.exc_info()
        LOGGER.error('%s with picklayers on file %s: %s' % (
             errtype, radar_dat.file_name, errval
        ))
        # Set interfaces to NaN if anything goes wrong
        airsnow = np.array([np.nan] * radar_dat.lat.shape[0])
        snowice = np.array([np.nan] * radar_dat.lat.shape[0])
        log_coefs = np.array([np.nan] * radar_dat.lat.shape[0])
        lin_coefs = np.array([np.nan] * radar_dat.lat.shape[0])
        
    # Calc snow depth and remove back picks (ie negative snow depth)
    snow_depth = (snowice - airsnow) * radar_dat.dfr / params['n_snow']
    # trick to get around the invalid-value runtime warnings
    # props to Jaime: https://stackoverflow.com/a/25346972
    
    
    mask = ~np.isnan(snow_depth)
    mask[mask] &= snow_depth[mask] < 0
    snow_depth[mask] = np.nan
    # snowice[mask] = np.nan
    # airsnow[mask] = np.nan
    
    # Add SnowRadar source file
    data_src = np.array([radar_dat.file_name] * radar_dat.lat.shape[0])
    
    
    # if airsnow != np.nan and snowice != np.nan:
    
    _, _, elev_corr = lever_arm_compensation(get_phase_center(radar_dat.season), radar_dat)
    # radar_dat.lon = lon_corr
    # radar_dat.lat = lat_corr
    radar_dat.elevation = elev_corr
    
    #THIS IS VERY CLUMSY
    try:
        air_snow_distance = (radar_dat.time_fast[upper + airsnow] ) * C / 2
    except:
        air_snow_distance = []
        for i in airsnow:
            if np.isnan(i) == False:
                air_snow_distance.append((radar_dat.time_fast[upper + int(i)] ) * C / 2)
            else:
                air_snow_distance.append(np.nan)
                
    snow_ice_distance = air_snow_distance + snow_depth
    # radar_sub = 10 * np.log10(radar_dat.data_radar)
    pulse_peakiness = np.max(radar_dat.data_radar, axis=0) / [np.sum(radar_dat.data_radar.T[:,x]) for x in [find_peaks(x)[0] for x in radar_dat.data_radar.T]]
    
    dists = []
    for lon, lat in zip(radar_dat.lon, radar_dat.lat):
        dists.append(haversine.haversine((radar_dat.lat[0],radar_dat.lon[0]), (lat,lon),  unit='m'))
        
    elevation_axis = np.mean(radar_dat.elevation) - radar_dat.time_fast * C / 2 #is only an approximate, assuming that altitude of the aircraft does not change significantly within the file

    # Write to .nc
    ds = xr.Dataset(
        data_vars=dict(
            radar_data = (['range_bin','time'], radar_dat.data_radar),
            pulse_peakiness = (['time'], pulse_peakiness),
            
            # line =  (['time'], radar_dat.line),
            altitude=(["time"], radar_dat.elevation),
            roll=(["time"], radar_dat.roll),
            pitch=(["time"], radar_dat.pitch),
            
            wavelet_coefs_log=(['range_bin',"time"], log_coefs),
            wavelet_coefs_lin=(['range_bin',"time"], lin_coefs),
            
            air_snow_index=(["time"], upper + airsnow),
            snow_ice_index=(["time"], upper + snowice),
            
            air_snow_distance=(["time"], air_snow_distance),
            snow_ice_distance=(["time"], snow_ice_distance),
            
            air_snow_elevation=(["time"], radar_dat.elevation - air_snow_distance),
            snow_ice_elevation=(["time"], radar_dat.elevation - snow_ice_distance),
            snow_depth=(["time"], snow_depth),

        ),
        coords=dict(
            time=("time",radar_dat.time_utc),
            range_bin=("range_bin",range(np.shape(radar_dat.data_radar)[0])),
            elevation_axis=("range_bin",elevation_axis),
            along_track_distance=("time", dists),
            lon=("time", radar_dat.lon),
            lat=("time", radar_dat.lat),
        ),
        attrs=dict(
            radar_name=radar_dat.radar_name_code,
            src=radar_dat.file_name,
            picker=picker.__name__,
            n_snow=params['n_snow'],
            null_2_space=params['null_2_space'],
            delta_fast_time_range=params['delta_fast_time_range'],
            
        )
    )

    # result = pd.DataFrame({
    #     'src': data_src,
    #     'picker': picker.__name__,
    #     'time': radar_dat.time_utc,
    #     'lat': radar_dat.lat,
    #     'lon': radar_dat.lon,
    #     'altitude':radar_dat.elevation,
    #     ''
    #     'n_snow': params['n_snow'],
    #     'b_as': upper + airsnow,
    #     'b_si': upper + snowice,
    #     'snow_depth': snow_depth,
    # })
    
    if dump_results:
        outpath.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(str(outfile))
        
        # result.to_csv(str(outfile), na_rep='nan')
        
        # result.to_csv(str(outfile), na_rep='nan')
    
    return ds

def batch_process(input_sr_data, picker, params, workers=4, dump_results=False, overwrite=True, path='./'):
    '''
    For a given list of SnowRadar data file paths:
        1) Pick air-snow and snow-ice interface layers for the 
           data files using the provided picker and snow density

        2) Produce a dataframe with the all the picked interfaces for each 
           of the data files

    Arguments:
        picker: which picker algorithm to apply (Wavelet-TN, NSIDC, or GSFC-NK)
        params: dictonary of picker paramters or list of dicts
        workers: number of worker processes to use
        dump_results: dumps each dataframe to a local csv

    Output:
        A concatenated pandas dataframe with the following columns:
            'src': the name of the source SnowRadar data file
            'lat': latitude of trace(?)
            'lon': longitude of trace(?)
            'n_snow': the refractive index used during layer picking
            'b_as': the picked air-snow interface layer
            'b_si': the picked snow-ice interface layer
            'snow_depth': estimated snow depth based on picked layers

    '''   
    current_cores = cpu_count()
    if workers > current_cores:
        raise SystemError('workers argument (passed: %d) cannot ' % workers + 
                          'exceed current CPU count (%d)' % current_cores)

    # Generate the picker input and output paramters
    length = len(input_sr_data)
    dump_triggers = [dump_results] * length     
    overwrite_triggers = [overwrite] * length  
    path_triggers = [path] * length  
    picker_args = [picker] * length                     
    
    if isinstance(params, dict):
    # If the input parameters the same for every file
        process_args = zip(
            input_sr_data, 
            picker_args,
            [params] * length,
            dump_triggers,
            overwrite_triggers,
            path_triggers

        )
    # If the input parameters vary
    elif isinstance(params, list):
        process_args = zip(
            input_sr_data, 
            picker_args,
            params,
            dump_triggers,
            overwrite_triggers,
            path_triggers
        )
    
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(extract_layers, *foo) for foo in process_args]
        results = [f.result() for f in futures]
    
    # return a concatenated dataframe containing results for all input datasets
    return results

def fetch_atm_data(sr, atm_folder):
    '''
    Attempt to find and load any locally-available NASA ATM data granules 
    that share the same day as the passed SnowRadar object

    Inputs:
        atm_folder: the local directory where ATM granules should be found
    
    Outputs:
        A dataframe containing concatenated ATM data (if multiple local ATM files exist)
    '''
    if not os.path.isdir(atm_folder):
        raise FileNotFoundError('Cannot locate ATM folder: %s' % os.path.abspath(atm_folder))

    # check for temporal match (same day as current SnowRadar data)
    d = sr.day.strftime('%Y%m%d')
    relevant_atm_data = [
        ATM(os.path.join(r, f)) 
        for r, ds, fs in os.walk(atm_folder) 
        for f in fs if 
        'ATM' in f and 
        f.endswith('.h5') and 
        f.split('_')[1] == d
    ]
    if len(relevant_atm_data) == 0:
        LOGGER.warning('No ATM data found for %s' % str(sr))
        return
    
    # check for spatial match (very rough due to simplicity of atm.bbox)
    relevant_atm_data = [
        atm for atm in relevant_atm_data
        if atm.bbox.intersects(sr.line)
    ]
    if len(relevant_atm_data) == 0:
        LOGGER.warning('No ATM data found for %s' % str(sr))
        return

    # assuming we still have some ATM data after spatiotemporal filtering, 
    # we sort by filename and concatenate into one big dataframe
    relevant_atm_data.sort(key=lambda x: x.file_name)
    df = pd.concat([
        pd.DataFrame({
            'atm_src': [atm.file_name]*len(atm.pitch),
            'atm_lat': atm.latitude,
            'atm_lon': atm.longitude,
            'atm_elev': atm.elevation,
            'atm_pitch': atm.pitch,
            'atm_roll': atm.roll,
            'atm_time_gps': atm.time_gps
        })
        for atm in relevant_atm_data
    ]).reset_index(drop=True)
    return df


def lever_arm_compensation(phase_center, radar_dat):
    # !!!!!! At the moment this ignores offset in y (wing) direction, since this offset is super small (4cm) in the 2016 Greenland P3 campaign !!!!!
    # since lever arms are given in meters, it is probably easiest to convert lon, lat to x,y [m] first, then do the compensation and then convert back
    # elevation is simply an offset 
    
    transformer1 = Transformer.from_crs('EPSG:4326', "EPSG:3413", always_xy=True,)
    transformer2 = Transformer.from_crs('EPSG:3413', "EPSG:4326", always_xy=True,)

    x,y = transformer1.transform(radar_dat.lon, radar_dat.lat)
    headings = np.arctan2((x[1:]-x[:-1]), (y[1:]-y[:-1]))
    headings= np.insert(headings,0,headings[0])
    x_corr = x + np.sin(headings) * phase_center[0]
    y_corr = y + np.cos(headings) * phase_center[0]
    lon_corr, lat_corr = transformer2.transform(x_corr,y_corr)
    
    elev_corr = radar_dat.elevation + phase_center[2]
    
    return lon_corr, lat_corr, elev_corr


def get_phase_center(season):
    if season == '2016_Greenland_P3':
    #Lever arms for 2016 Greenland P3, adapted from: https://gitlab.com/openpolarradar/opr/-/blob/main/matlab/processing/lever_arm.m#L2914:
    # Snow Tx: X = 297.75",Y = 0;Z = -26.81"; RX: X = 168.5",Y = 0;Z = -38.75" 
    # X,Y,Z are in aircraft coordinates relatively to GPS antenna

        LArx = np.zeros(3)
        LAtx = np.zeros(3)

        LArx[0] = -168.5*2.54/100  #- 1.355 what are the additional numbers 
        LArx[1] =  0.045
        LArx[2] = -38.75*2.54/100 #+ 3.425

        LAtx[0] = -297.75*2.54/100  #- 1.355
        LAtx[1] = -0.045
        LAtx[2] = -26.81*2.54/100 #+ 3.425

        phase_center = (LArx + LAtx) / 2
    return phase_center

    