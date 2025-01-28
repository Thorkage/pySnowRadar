import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from shapely.geometry import box

import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
from pySnowRadar import ATM, SnowRadar, algorithms
from shapely.geometry import Polygon, Point, MultiPoint, LineString
from scipy.signal import find_peaks
import haversine

from pyproj import Transformer
transformer = Transformer.from_crs(4326, 3413, always_xy=True)
import matplotlib.pyplot as plt
from tqdm import tqdm

C = 299792458 # Vacuum speed of light
from scipy.constants import speed_of_light

LOGGER = logging.getLogger(__name__)

def geo_filter_insitu_sites(path, year, site, input_sr_data):
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
    land = gpd.read_file(os.path.join(path, f'EUREKA{year}_{site}_measurement_bounds.shp'))
    # print(land)
    # x = np.array(list(land.loc[0, 'geometry'].exterior.coords.xy[0])) - 360
    # y = np.array(list(land.loc[0, 'geometry'].exterior.coords.xy[1]))
    # land.loc[0,'geometry'] = Polygon(zip(x,y))
    land = land.set_crs('EPSG:3413')
    
    # Load the datafiles in 'meta' mode to just scrape the simplified track line
    sr_meta = [SnowRadar(sr, 'meta') for sr in input_sr_data]
    sr_gdf = gpd.GeoDataFrame(
        data={'file': [sr.file_path for sr in sr_meta]}, 
        geometry=[sr.line_xy for sr in sr_meta], 
        crs='epsg:3413'
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
        geometry=[sr.line_lonlat for sr in sr_meta], 
        crs='epsg:4326'
    )
    sr_gdf = sr_gdf.drop(
        gpd.sjoin(sr_gdf, land, how='inner').index
    )
    if len(sr_gdf) < 1:
        LOGGER.warning('No suitable datafiles left after geospatial filtering')
        return []
    return sr_gdf.file.tolist()


def lever_arm_compensation(phase_center, radar_dat):
    # !!!!!! At the moment this ignores offset in y (wing) direction, since this offset is super small (4cm) in the 2016 Greenland P3 campaign !!!!!
    # since lever arms are given in meters, it is probably easiest to convert lon, lat to x,y [m] first, then do the compensation and then convert back
    # elevation is simply an offset 
    
    # transformer1 = Transformer.from_crs('EPSG:4326', "EPSG:3413", always_xy=True,)
    # transformer2 = Transformer.from_crs('EPSG:3413', "EPSG:4326", always_xy=True,)

    # x,y = transformer1.transform(radar_dat.lon, radar_dat.lat)
    # headings = np.arctan2((x[1:]-x[:-1]), (y[1:]-y[:-1]))
    # headings= np.insert(headings,0,headings[0])
    # x_corr = x + np.sin(headings) * phase_center[0]
    # y_corr = y + np.cos(headings) * phase_center[0]
    # lon_corr, lat_corr = transformer2.transform(x_corr,y_corr)
    
    elev_corr = radar_dat.elevation + phase_center[2]
    
    return elev_corr


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
        
        
    if season == '2014_Greenland_P3':
        LArx = np.zeros(3)
        LAtx = np.zeros(3)
        
        LArx[0]   = -384.4*0.0254
        LArx[1]   = 10*0.0254
        LArx[2]   = -80.6*0.0254
        
        LAtx[0]   = -348.4*0.0254
        LAtx[1]   = 10*0.0254
        LAtx[2]   = -80.6*0.0254

    phase_center = (LArx + LAtx) / 2
        
    return phase_center



def construct_elevation_axis(params, elevation_axis, airsnow):
    elev_snow_compensation = params['delta_fast_time_range'] / params['n_snow'] #- params['delta_fast_time_range'] #
    eaxis_arr = []
    
    for asnow in airsnow:
        eaxis = elevation_axis.copy()
        
        if not np.isnan(asnow):
            eaxis[int(asnow):] = eaxis[int(asnow)] - np.cumsum([elev_snow_compensation]*len(eaxis[int(asnow):])) 
            
        eaxis_arr.append(eaxis)
        
    eaxis_arr = np.column_stack(eaxis_arr)
    return eaxis_arr



def find_level_surface(air_snow_elevation, snow_ice_elevation, elevations_axii):
    df_tmp = pd.DataFrame({'air_snow_elevation':air_snow_elevation})
    df_tmp['elev_quantile'] = pd.qcut(df_tmp['air_snow_elevation'], q=100, labels=False, duplicates='drop')
    min_ind = (df_tmp.groupby('elev_quantile')['air_snow_elevation'].mean().diff().argmin() - 10 , df_tmp.groupby('elev_quantile')['air_snow_elevation'].mean().diff().argmin() + 10)
    level_ice_elevation = df_tmp.groupby('elev_quantile')['air_snow_elevation'].mean().loc[min_ind[0]:min_ind[1]].mean()
    
    air_snow_elevation -= level_ice_elevation
    snow_ice_elevation -= level_ice_elevation
    elevations_axii -= level_ice_elevation

    return air_snow_elevation, snow_ice_elevation, elevations_axii, level_ice_elevation


def get_footprintsize(H, c=speed_of_light, center_frequency=5e9, B=6e9, kt=1.5 ,T=0, v=140, n=16, PRF=1953.125):
    '''
    Equation 4, 5 and 7 from IRSNO1B documentation
    
    H: altitude
    c: speed of light in vacuum
    center_frequency: center frequency (5 GHz)
    B: Bandwidth in radians (?), using GHz right now
    kt: 1.5 (side-lobe reduction something)
    T: depth in ice
    
    '''
    
    lambdac = speed_of_light / center_frequency
    L = (n * v) / (PRF) # from Arttu thesis equation 4.6
    
    along_track_resolution = H * np.tan(np.arcsin(lambdac/ (2 * L)))
    
    across_track_resolution = 2 * np.sqrt((c * kt)/B * (H + (T)/(np.sqrt(3.15))))
    # across_track_resolution = np.array([200] * len(across_track_resolution))
    
    # FRESNEL ZONE, IF SMOOTH (QUASI SPECULAR TARGET) AS E.G. INTERNAL LAYERS (?)
    # across_track_resolution = np.sqrt(2 * lambdac * (H + (T)/(np.sqrt(3.15))))
    
    return across_track_resolution/2, along_track_resolution/2


def perpendicular_line_with_buffer(linestring, p1, across_track_radius, along_track_radius):
    """
    Constructs a rectangle centered at a given point along a LineString.
    
    Args:
        linestring (LineString): The input LineString in meters.
        p1 (tuple): The point (latitude, longitude) where the rectangle is centered.
        across_track_radius (float): The width of the rectangle (in meters).
        along_track_radius (float): The length of the rectangle (in meters).
    
    Returns:
        Polygon: The rectangle as a Polygon.
    """
    # Find the nearest point on the projected LineString to p1
    point_on_line = linestring.interpolate(linestring.project(Point(p1)))

    # Identify the tangent vector of the nearest segment
    coords = np.array(linestring.coords)
    min_distance = float('inf')
    closest_segment = None
    for i in range(len(coords) - 1):
        segment = LineString([coords[i], coords[i + 1]])
        distance = segment.distance(point_on_line)
        if distance < min_distance:
            min_distance = distance
            closest_segment = coords[i], coords[i + 1]

    # Calculate the tangent vector
    (x1, y1), (x2, y2) = closest_segment
    tangent_vector = np.array([x2 - x1, y2 - y1])
    tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)  # Normalize it

    # Compute the perpendicular vector by rotating the tangent vector by 90 degrees
    perp_vector = np.array([-tangent_vector[1], tangent_vector[0]])
    perp_vector = perp_vector / np.linalg.norm(perp_vector)  # Normalize it

    # Scale the perpendicular vector to half the desired width
    half_width_vector = perp_vector * (across_track_radius)
    half_length_vector = tangent_vector * (along_track_radius)

    # Create the four corners of the rectangle
    mid_point = np.array([point_on_line.x, point_on_line.y])
    corner1 = mid_point + half_width_vector + half_length_vector
    corner2 = mid_point + half_width_vector - half_length_vector
    corner3 = mid_point - half_width_vector - half_length_vector
    corner4 = mid_point - half_width_vector + half_length_vector

    # Create the rectangle as a Polygon
    rectangle = Polygon([corner1, corner2, corner3, corner4, corner1])

    return rectangle


def construct_footprints_theoretical(radar_dat, x, y, level_ice_elevation, velocity):
    """
    Constructs footprints at x,y of input DataFrame with given along-track and across-track radius 
    
    Args:
        df (pandas DataFrame): The input LineString in meters.
        along_track_radius (float): Radius [m] of the footprint in along-track direction 
        across_track_radius (float):  Radius [m] of the footprint across along-track direction 
    
    Returns:
        The DataFrame indexed from 1:-1, since we use adjacent coordinates to construct the footprint
        The footprints (shapely.Polygon)
    """
    

    across_track_radius, along_track_radius = get_footprintsize(H=radar_dat.elevation - level_ice_elevation,
                                                                B=radar_dat.bandwidth,
                                                                # center_frequency=radar_dat.center_frequency, #center frequency is weird. It should be (8 - 2) / 2, I guess
                                                                v=velocity,
                                                                n=radar_dat.number_averages,
                                                                PRF=radar_dat.prf
                                                                )
    footprints = []   
    footprints.append(np.nan)
    for i in range(1,len(x)-1):
        p0 = (x[i-1], y[i-1])
        p1 = (x[i], y[i])
        p2 = (x[i+1], y[i+1])

        rectangle = perpendicular_line_with_buffer(LineString([p0,p2]), p1, across_track_radius[i], along_track_radius[i])
        footprints.append(rectangle)
    footprints.append(np.nan)
    
        
    return footprints, across_track_radius, along_track_radius
 

def extract_layers(data_path, picker=algorithms.Wavelet_TN, params=None, dump_results=False, overwrite=True, path='./dump', atm_folder='./'):
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
    
    # Add SnowRadar source file
    
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
    data_src = np.array([radar_dat.file_name] * radar_dat.lat.shape[0])

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
    # lower, upper = radar_dat.get_bounds(m_above=5)
    # radar_sub = radar_dat.data_radar[upper:lower, :]
   
    # Calc or init other necessary params
    params['n_snow'] = np.sqrt((1 + 0.51 * params['snow_density']) ** 3)
    params['null_2_space']  = radar_dat.n2n
    params['delta_fast_time_range'] = radar_dat.dfr
    

    dists = []
    for lon, lat in zip(radar_dat.lon, radar_dat.lat):
        dists.append(haversine.haversine((radar_dat.lat[0],radar_dat.lon[0]), (lat,lon),  unit='m'))
    velocity = dists[-1] / (radar_dat.time_utc[-1] - radar_dat.time_utc[0]).total_seconds()
    
    # Apply the picker to the file, trace by trace
    # return locs_as, locs_as_left, locs_as_right, locs_si, locs_si_left, locs_si_right, sum_log_coefs, sum_lin_coefs #locs_as_1per
    
    if picker.__name__ == 'Wavelet_TN':
        write_type = 'Wavelet'
        airsnow, airsnow_left, airsnow_right, snowice, snowice_left, snowice_right, log_coefs, lin_coefs, snowice_ratio, max_elevation_ind = [], [], [], [], [], [], [], [], [], []
        for trace in radar_dat.data_radar.T:
            asnow, asnow_left, asnow_right, sice, sice_left, sice_right, log_coef, lin_coef, si_ratio = picker(trace, **params)
            airsnow.append(asnow)
            airsnow_left.append(asnow_left)
            airsnow_right.append(asnow_right)
            
            snowice.append(sice)
            snowice_left.append(sice_left)
            snowice_right.append(sice_right)
            snowice_ratio.append(si_ratio)
            
            log_coefs.append(log_coef)
            lin_coefs.append(lin_coef)
            
            max_elevation_ind.append(np.argmax(trace))
            
        airsnow = np.array(airsnow)
        airsnow_left = np.array(airsnow_left)
        airsnow_right = np.array(airsnow_right)
        
        snowice = np.array(snowice)
        snowice_left = np.array(snowice_left)
        snowice_right = np.array(snowice_right)
        snowice_ratio = np.array(snowice_ratio)
        
        log_coefs = np.array(log_coefs).T
        lin_coefs = np.array(lin_coefs).T
    
        max_elevation_ind = np.array(max_elevation_ind)
    
    elif picker.__name__ == 'Peakiness':
        write_type = 'Peakiness'
        airsnow, snowice, max_elevation_ind = [], [], []
        
        for trace in radar_dat.data_radar.T:
            asnow, sice = picker(trace, **params)
            if np.isnan(asnow):
                airsnow.append(np.nan)
            else:
                airsnow.append(int(asnow))
                
            if np.isnan(sice):
                snowice.append(np.nan)
            else:
                snowice.append(int(sice))    
                
            max_elevation_ind.append(np.argmax(trace))
            # snowice.append(int(sice) if sice != np.nan else np.nan)
        airsnow = np.array(airsnow)
        snowice = np.array(snowice)
        max_elevation_ind = np.array(max_elevation_ind)

    # Calc snow depth and remove back picks (ie negative snow depth)
    snow_depth = (snowice - airsnow) * radar_dat.dfr / params['n_snow']
    
    
    # trick to get around the invalid-value runtime warnings
    # props to Jaime: https://stackoverflow.com/a/25346972
    mask = ~np.isnan(snow_depth)
    mask[mask] &= snow_depth[mask] < 0
    snow_depth[mask] = np.nan
    # snowice[mask] = np.nan
    # airsnow[mask] = np.nan
    
    
    if write_type == 'Wavelet':
        elev_corr = lever_arm_compensation(get_phase_center(radar_dat.season), radar_dat)
        radar_dat.elevation = elev_corr
        
        elevation_axis = np.mean(radar_dat.elevation) - np.array(radar_dat.time_fast) * C / 2 #- level_ice_elevation #is only an approximate, assuming that altitude of the aircraft does not change significantly within the file
        elevations_axii = construct_elevation_axis(params, elevation_axis, airsnow)
        
        air_snow_elevation = elevations_axii[airsnow,range(len(airsnow))]
        snow_ice_elevation = elevations_axii[snowice,range(len(snowice))]
        air_snow_elevation, snow_ice_elevation, elevations_axii, level_ice_elevation = find_level_surface(air_snow_elevation, snow_ice_elevation, elevations_axii)
    
        air_snow_left_elevation = elevations_axii[airsnow_left,range(len(airsnow))]
        air_snow_right_elevation = elevations_axii[airsnow_right,range(len(airsnow))]
        snow_ice_left_elevation = elevations_axii[snowice_left,range(len(airsnow))]
        snow_ice_right_elevation = elevations_axii[snowice_right,range(len(airsnow))]
        
        # air_snow_left_elevation -= level_ice_elevation
        # air_snow_right_elevation -= level_ice_elevation
        # snow_ice_left_elevation -= level_ice_elevation
        # snow_ice_right_elevation -= level_ice_elevation
        
        max_elevation = elevations_axii[max_elevation_ind,range(len(max_elevation_ind))]
        
    elif write_type == 'Peakiness':
        elev_corr = lever_arm_compensation(get_phase_center(radar_dat.season), radar_dat)
        radar_dat.elevation = elev_corr
        
        elevation_axis = np.mean(radar_dat.elevation) - np.array(radar_dat.time_fast) * C / 2 #- level_ice_elevation #is only an approximate, assuming that altitude of the aircraft does not change significantly within the file
        elevations_axii = construct_elevation_axis(params, elevation_axis, airsnow)
        
        air_snow_elevation = []
        snow_ice_elevation = []
        max_elevation = []
        
        for i, maxind in enumerate(max_elevation_ind):
            max_elevation.append(elevations_axii[maxind,i])
            
        for i, asnow in enumerate(airsnow):
            if np.isnan(asnow):
                air_snow_elevation.append(np.nan)
            else:
                air_snow_elevation.append(elevations_axii[int(asnow),i])
                
        for i, sice in enumerate(snowice):
            if np.isnan(sice):
                snow_ice_elevation.append(np.nan)
            else:
                snow_ice_elevation.append(elevations_axii[int(sice),i])
        
        air_snow_elevation, snow_ice_elevation, elevations_axii, level_ice_elevation = find_level_surface(air_snow_elevation, snow_ice_elevation, elevations_axii)
        max_elevation -= level_ice_elevation
    
    footprints, across_track_radius, along_track_radius = construct_footprints_theoretical(radar_dat, radar_dat.x, radar_dat.y, level_ice_elevation, velocity)
    # print(f'Footprints constructed: {len(footprints)}')
    # print('ATM folder:', atm_folder)
    if atm_folder != None:
        ATM_data = fetch_atm_data_levelled(radar_dat, atm_folder)
        ATM_classes, ATM_as_interfaces_mean, ATM_as_interfaces_90 = match_atm_data2(ATM_data, footprints)

    
    noise =  radar_dat.data_radar[:100, :].mean(axis=0)
    SNR = 10 * np.log10(radar_dat.data_radar / noise)
    
    if write_type == 'Wavelet':
        
        ds = xr.Dataset(
            data_vars=dict(
                radar_data = (['range_bin','time'], radar_dat.data_radar),
                
                noise = (['time'], noise),
                SNR = (['range_bin','time'], SNR),
                
                altitude=(["time"], radar_dat.elevation),
                roll=(["time"], radar_dat.roll),
                pitch=(["time"], radar_dat.pitch),
                # footprints=(["time"], footprints),
                
                # across_track_radius=(["time"], across_track_radius),
                # along_track_radius=(["time"], along_track_radius),

                wavelet_coefs_log=(['range_bin',"time"], log_coefs),
                wavelet_coefs_lin=(['range_bin',"time"], lin_coefs),
                

                air_snow_index=(["time"], airsnow),
                snow_ice_index=(["time"], snowice),
                
                air_snow_SNR=(["time"], SNR[airsnow, range(len(airsnow))]),
                snow_ice_SNR=(["time"], SNR[snowice, range(len(airsnow))]),
                
                air_snow_elevation=(["time"], air_snow_elevation),
                air_snow_left_elevation=(["time"], air_snow_left_elevation),
                air_snow_right_elevation=(["time"], air_snow_right_elevation),
                
                snow_ice_elevation=(["time"], snow_ice_elevation),
                snow_ice_left_elevation=(["time"], snow_ice_left_elevation),
                snow_ice_right_elevation=(["time"], snow_ice_right_elevation),
                snow_ice_ratio=(["time"], snowice_ratio),
                snow_depth=(["time"], snow_depth),
                
                max_elevation=(["time"], max_elevation),
                # htopo=(["time"], htopo)
                ATM_classes=(["time"], ATM_classes) if atm_folder != None else None,
                ATM_as_interfaces_mean=(["time"], ATM_as_interfaces_mean) if atm_folder != None else None,
                ATM_as_interfaces_90=(["time"], ATM_as_interfaces_90) if atm_folder != None else None  
                

            ),
            coords=dict(
                time=("time",radar_dat.time_utc),
                range_bin=("range_bin",range(np.shape(radar_dat.data_radar)[0])),
                # elevation_axis=("range_bin",elevation_axis),
                elevation_axii=(["range_bin","time"],elevations_axii),
                along_track_distance=("time", dists),
                lon=("time", radar_dat.lon),
                lat=("time", radar_dat.lat),
                
                # x and y should be added in the future (in e.g., epsg:3413) 
                x=("time", transformer.transform(radar_dat.lon,  radar_dat.lat)[0]),
                y=("time", transformer.transform( radar_dat.lon,  radar_dat.lat)[1]),
            ),
            attrs=dict(
                radar_name=radar_dat.radar_name_code,
                src=radar_dat.file_name,
                picker=picker.__name__,
                bandwidth=radar_dat.bandwidth,
                center_frequency=radar_dat.centerfrequency,
                n_snow=params['n_snow'],
                null_2_space=params['null_2_space'],
                delta_fast_time_range=params['delta_fast_time_range'],
                level_ice_elevation=level_ice_elevation,
                number_averages=radar_dat.number_averages,
                PRF=radar_dat.prf,
                velocity=velocity #just a rough estimate of the velocity of the aircraft
                
            )
        )
    elif write_type == 'Peakiness':
        ds = xr.Dataset(
            data_vars=dict(
                # radar_data = (['range_bin','time'], radar_dat.data_radar),
                
                altitude=(["time"], radar_dat.elevation),
                roll=(["time"], radar_dat.roll),
                pitch=(["time"], radar_dat.pitch),
                
                air_snow_index=(["time"], airsnow),
                snow_ice_index=(["time"], snowice),
                
                air_snow_elevation=(["time"], air_snow_elevation),
                snow_ice_elevation=(["time"], snow_ice_elevation),
                snow_depth=(["time"], snow_depth),
                
                max_elevation=(["time"], max_elevation),
                ATM_classes=(["time"], ATM_classes) if atm_folder != None else None,
                ATM_as_interfaces_mean=(["time"], ATM_as_interfaces_mean) if atm_folder != None else None,
                ATM_as_interfaces_90=(["time"], ATM_as_interfaces_90) if atm_folder != None else None  
                
                
            ),
            coords=dict(
                time=("time",radar_dat.time_utc),
                range_bin=("range_bin",range(np.shape(radar_dat.data_radar)[0])),
                # elevation_axis=("range_bin",elevation_axis),
                # elevation_axii=(["range_bin","time"],elevations_axii),
                along_track_distance=("time", dists),
                lon=("time", radar_dat.lon),
                lat=("time", radar_dat.lat),
                
                # x and y should be added in the future (in e.g., epsg:3413) 
                x=("time", transformer.transform(radar_dat.lon,  radar_dat.lat)[0]),
                y=("time", transformer.transform( radar_dat.lon,  radar_dat.lat)[1]),
            ),
            attrs=dict(
                radar_name=radar_dat.radar_name_code,
                src=radar_dat.file_name,
                picker=picker.__name__,
                bandwidth=radar_dat.bandwidth,
                center_frequency=radar_dat.centerfrequency,
                n_snow=params['n_snow'],
                null_2_space=params['null_2_space'],
                delta_fast_time_range=params['delta_fast_time_range'],
                level_ice_elevation=level_ice_elevation,
                number_averages=radar_dat.number_averages,
                PRF=radar_dat.prf,
                velocity=velocity #just a rough estimate of the velocity of the aircraft
                
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


def batch_process(input_sr_data, picker, params, workers=4, dump_results=False, overwrite=True, path='./', atm_folder='./'):
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
            path_triggers,
            [atm_folder] * length

        )
    # If the input parameters vary
    elif isinstance(params, list):
        process_args = zip(
            input_sr_data, 
            picker_args,
            params,
            dump_triggers,
            overwrite_triggers,
            path_triggers,
            [atm_folder] * length
            
        )
    
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(extract_layers, *foo) for foo in process_args]
        results = [f.result() for f in futures]
    
    # return a concatenated dataframe containing results for all input datasets
    return results


def get_atm_filename(row):
    date = row[1]['date']
    k = row[1]['k']
    i = row[1]['i']
    filename = f'{date}_ATM_levelled_{k}_{i}.csv'
    return filename

def fetch_atm_data_levelled(sr, atm_folder):
    
    polygons = pd.read_csv(os.path.join(atm_folder, 'polygons.csv'))
    polygons['bbox'] = polygons['bbox'].apply(lambda x: np.array(x.strip('()').split(',')).astype(float))
    polygons['bbox'] = polygons['bbox'].apply(lambda x: get_atm_bbox(x))
    
    d = sr.day.strftime('%Y%m%d')
    
    relevant_atm_data_files = [
        get_atm_filename(row) for row in polygons.iterrows()
        if row[1]['bbox'].intersects(sr.line_xy) 
        and str(row[1]['date']) == d
    ]
    print(len(relevant_atm_data_files))
    relevant_atm_data = [
        pd.read_csv(os.path.join(atm_folder, f)) 
        for f in relevant_atm_data_files
    ]
    
    
    if len(relevant_atm_data) == 0:
        LOGGER.warning('No ATM data found for %s' % str(sr))
        return
    
    
    # relevant_atm_data.sort(key=lambda x: x.file_name)
    
    for i, data in enumerate(relevant_atm_data):
        if i  == 0:
            df = data
        else:    
            df = pd.concat([df, data])
            
    df.reset_index(drop=True, inplace=True)
    
    return df

def get_atm_bbox(row):
    try:
        return box(*row)
    except:
        return box(0,0,0,0)

# def fetch_atm_data_grid(sr, atm_folder):

#     d = sr.day.strftime('%Y%m%d')
    
#     relevant_atm_data = [
#         xr.open_dataset(os.path.join(r, f)) 
#         for r, ds, fs in os.walk(atm_folder) 
#         for f in fs if 
#         'ATM' in f and 
#         f.endswith('.nc') and 
#         f.split('_')[0] == d
#     ]
    
#     if len(relevant_atm_data) == 0:
#         LOGGER.warning('No ATM data found for %s' % str(sr))
#         return
    
#     relevant_atm_data = [
#         atm for atm in relevant_atm_data
#         if box(*atm.attrs.bbox_xy).intersects(sr.line_xy)
#     ]
    
#     if len(relevant_atm_data) == 0:
#         LOGGER.warning('No ATM data found for %s' % str(sr))
#         return
    
#     relevant_atm_data.sort(key=lambda x: x.file_name)
    
#     for data in relevant_atm_data:
#         if data  == relevant_atm_data[0]:
#             df = data.to_dataframe().reset_index()
#         else:    
#             df = pd.concat([df, data.to_dataframe().reset_index()])
            
#     return df


# def fetch_atm_data(sr, atm_folder):
#     '''
#     Attempt to find and load any locally-available NASA ATM data granules 
#     that share the same day as the passed SnowRadar object

#     Inputs:
#         atm_folder: the local directory where ATM granules should be found
    
#     Outputs:
#         A dataframe containing concatenated ATM data (if multiple local ATM files exist)
#     '''
#     if not os.path.isdir(atm_folder):
#         raise FileNotFoundError('Cannot locate ATM folder: %s' % os.path.abspath(atm_folder))

#     # check for temporal match (same day as current SnowRadar data)
#     d = sr.day.strftime('%Y%m%d')
#     relevant_atm_data = [
#         ATM(os.path.join(r, f)) 
#         for r, ds, fs in os.walk(atm_folder) 
#         for f in fs if 
#         'ATM' in f and 
#         f.endswith('.h5') and 
#         f.split('_')[1] == d
#     ]
#     if len(relevant_atm_data) == 0:
#         LOGGER.warning('No ATM data found for %s' % str(sr))
#         return

#     # check for spatial match (very rough due to simplicity of atm.bbox)
#     relevant_atm_data = [
#         atm for atm in relevant_atm_data
#         if atm.bbox_xy.intersects(sr.line_xy)
#     ]
#     if len(relevant_atm_data) == 0:
#         LOGGER.warning('No ATM data found for %s' % str(sr))
#         return

#     # assuming we still have some ATM data after spatiotemporal filtering, 
#     # we sort by filename and concatenate into one big dataframe
#     relevant_atm_data.sort(key=lambda x: x.file_name)
#     df = pd.concat([
#         pd.DataFrame({
#             'atm_src': [atm.file_name]*len(atm.pitch),
#             'atm_lat': atm.latitude,
#             'atm_lon': atm.longitude,
#             'atm_x': transformer.transform(atm.longitude, atm.latitude)[0],
#             'atm_y': transformer.transform(atm.longitude, atm.latitude)[1],
#             'atm_elev': atm.elevation,
#             'atm_pitch': atm.pitch,
#             'atm_roll': atm.roll,
#             'atm_time_gps': atm.time_gps
#         })
#         for atm in relevant_atm_data
#     ]).reset_index(drop=True)
#     return df


def points_in_poly_list(polygons, points, min_points=1):
    '''
    Function to find the indices of points located within each polygon from a list of polygons.

    Inputs:
    polygons: List of shapely.Polygon objects where points shall be checked.
    points: List of (x, y) tuples representing the points to check.
    min_points: Minimum number of points required within a polygon to include it in the output (default is 1).

    Returns:
    poly_points_indices: Dictionary where each key is the index of the polygon in the input list,
                         and the value is a list of indices of the points located within that polygon.
                         Only includes entries with at least min_points points.
    '''
    
    # Convert polygons to a GeoDataFrame
    gdf_poly = gpd.GeoDataFrame({'geometry': polygons}, geometry='geometry')

    # Convert points to a GeoDataFrame
    df_points = pd.DataFrame()
    df_points['points'] = points
    df_points['points'] = df_points['points'].apply(Point)
    gdf_points = gpd.GeoDataFrame(df_points, geometry='points')

    # Perform spatial join between points and polygons
    sjoin = gpd.sjoin(gdf_points, gdf_poly, predicate="within", how='inner')

    # Initialize dictionary with all polygon indices
    poly_points_indices = {i: [] for i in range(len(polygons))}

    # Group by polygon index and filter by min_points
    grouped = sjoin.groupby('index_right').apply(lambda x: x.index.tolist())
    for idx, pts in grouped.items():
        if len(pts) >= min_points:
            poly_points_indices[idx] = pts
    return poly_points_indices



# def match_atm_data(ATM_data, footprints):
    
#     points = [Point(x,y) for x,y in zip(ATM_data['x'], ATM_data['y'])]
#     poly_points_indices = points_in_poly_list(footprints, points)
    
#     elevations = []
#     for key in poly_points_indices.keys():
        
#         elevations.append(ATM_data.loc[poly_points_indices[key],'elev_levelled'])
        
#     return pd.DataFrame({'elevations': elevations})
    
   
def match_atm_data2(ATM_data, footprints, deformation_threshold=20):
    
    points = [Point(x,y) for x,y in zip(ATM_data['x'], ATM_data['y'])]
    poly_points_indices = points_in_poly_list(footprints, points)
    
    classes = []
    ATM_as_interfaces_mean = []
    ATM_as_interfaces_90 = []
    
    for key in poly_points_indices.keys():
        
        if len(poly_points_indices[key]) > 0:
            deform_percent = len(ATM_data.loc[poly_points_indices[key],'classes'].loc[ATM_data.loc[poly_points_indices[key],'classes']== 1] ) / len(ATM_data.loc[poly_points_indices[key]]) * 100
            classes.append(1 if deform_percent > deformation_threshold else 0)
            
            ATM_as_interfaces_mean.append(ATM_data.loc[poly_points_indices[key],'elev_levelled'].mean())
            ATM_as_interfaces_90.append(np.nanquantile(ATM_data.loc[poly_points_indices[key],'elev_levelled'], .9))
            
        else:
            classes.append(np.nan)
            ATM_as_interfaces_mean.append(np.nan)
            ATM_as_interfaces_90.append(np.nan)

    return classes, ATM_as_interfaces_mean, ATM_as_interfaces_90
     
# def calculate_htopo(ATM_elevations,length):
    
#     if ATM_elevations is None:
#         return [np.nan] * length
    
#     htopo = []
#     for i, elevs in enumerate(ATM_elevations):
#         htopo.append(np.nanquantile(elevs, .95) - np.nanquantile(elevs, .05))   
#     return htopo


