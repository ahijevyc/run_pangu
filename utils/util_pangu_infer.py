import scipy.ndimage
import numpy as np
import copy
import xarray as xr
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
import os

pangu_abbreviation_list = [
    'z1000', 'z925', 'z850', 'z700', 'z600', 'z500', 'z400', 'z300', 'z250', 'z200', 'z150', 'z100', 'z50',
    'q1000', 'q925', 'q850', 'q700', 'q600', 'q500', 'q400', 'q300', 'q250', 'q200', 'q150', 'q100', 'q50',
    't1000', 't925', 't850', 't700', 't600', 't500', 't400', 't300', 't250', 't200', 't150', 't100', 't50',
    'u1000', 'u925', 'u850', 'u700', 'u600', 'u500', 'u400', 'u300', 'u250', 'u200', 'u150', 'u100', 'u50',
    'v1000', 'v925', 'v850', 'v700', 'v600', 'v500', 'v400', 'v300', 'v250', 'v200', 'v150', 'v100', 'v50',
    'msl', 'u10m', 'v10m', 't2m'
]

# Define the channel mapping
variable_mapping = {
    'specific_humidity': 'q',
    'temperature': 't',
    'geopotential': 'z',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
    'mean_sea_level_pressure': 'msl',
    '10m_u_component_of_wind': 'u10m',
    '10m_v_component_of_wind': 'v10m',
    '2m_temperature': 't2m'
}

class PanguUpscaler:
    def __init__(self, basemap_products=None, LAT_SOUTH=20, LAT_NORTH=56, LON_WEST=-132, LON_EAST=-60):
        if basemap_products is None:
            self.download_raw= True
            raise ValueError("Please provide basemap products")
        else:
            self.download_raw = False
            self.NN_GRID_PTS = basemap_products[0]
            self.interp_lons = basemap_products[1]
            self.interp_lats = basemap_products[2]
            self.in_lons_proj = basemap_products[3]
            self.in_lats_proj = basemap_products[4]
            self.x81 = basemap_products[5]
            self.y81 = basemap_products[6]
            
            self.LAT_SOUTH = LAT_SOUTH
            self.LAT_NORTH = LAT_NORTH
            self.LON_WEST = LON_WEST
            self.LON_EAST = LON_EAST

            self.ds_out = xr.Dataset(
                {
                    "lat": (["lat"], self.interp_lats[:, 0]),
                    "lon": (["lon"], self.interp_lons[0, :]),
                }
            )
        
    def idw_interpolation(self, source_points, source_values, target_points, p=2):
        tree = cKDTree(source_points)
        distances, indices = tree.query(target_points, k=2)
        weights = 1 / (distances**p)
        weights /= weights.sum(axis=1, keepdims=True)
        return np.sum(source_values[indices] * weights, axis=1)

    def idw_interpolation_haversine(self, source_points, source_values, target_points, radius=120000, std_dev=25000):
        # Convert lat/lon to radians
        source_points_rad = np.radians(source_points)
        target_points_rad = np.radians(target_points)

        # Create BallTree
        tree = BallTree(source_points_rad, metric='haversine')

        # Query the tree
        earth_radius = 6371000  # meters
        indices, distances = tree.query_radius(target_points_rad, r=radius/earth_radius, 
                                            return_distance=True, sort_results=True)

        # Convert distances back to meters
        distances = [d * earth_radius for d in distances]

        interpolated_values = np.zeros(len(target_points))

        for i, (idx, dist) in enumerate(zip(indices, distances)):
            if len(idx) > 0:
                # Gaussian weights calculation
                weights = np.exp(-0.5 * (np.array(dist) / std_dev) ** 2)
                weights /= np.sum(weights)
                interpolated_values[i] = np.sum(source_values[idx] * weights)
            else:
                # If no points within radius, assign NaN
                interpolated_values[i] = np.nan
        return interpolated_values

    def pangu_upscale_forecast(self, data_dir, upscaled_model_fields, upscaled_derived_fields, get_date, fhr, ic, interp=True):
        print('upscaling', fhr)
 
        this_upscaled_fields = copy.deepcopy(upscaled_model_fields)
        this_upscaled_derived_fields = copy.deepcopy(upscaled_derived_fields)
        combined_this_upscaled_fields = {**this_upscaled_fields, **this_upscaled_derived_fields}
        assert self.download_raw != interp, "The interpolation flag should not be the same as the download_raw flag"
        try:
            #data_dir_lower = data_dir.lower()
            #if 'gfs' in data_dir_lower:
            #    ds = xr.open_dataarray(os.path.join(data_dir, get_date, 'pangu_hresan_forecast_data', f'pangu_gfs_pred_{str(fhr).zfill(3)}.nc')).isel(lat=slice(None, None, -1))
            #elif 'era5' in data_dir_lower:
            #    ds = xr.open_dataarray(os.path.join(data_dir, get_date, 'pangu_hresan_forecast_data', f'pangu_era5_pred_{str(fhr).zfill(3)}.nc')).isel(lat=slice(None, None, -1))
            #elif 'hresan' in data_dir_lower:
            #    ds = xr.open_dataarray(os.path.join(data_dir, get_date, 'pangu_hresan_forecast_data', f'pangu_{ic}_pred_{str(fhr).zfill(3)}.nc')).isel(lat=slice(None, None, -1))
                
            ds = xr.open_dataarray(os.path.join(data_dir, get_date, ic, 'pangu_forecast_data', f'pangu_{ic}_pred_{str(fhr).zfill(3)}.nc')).isel(lat=slice(None, None, -1))

            ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
            ds = ds.sortby(ds.lon)
            ds = ds.sel(lat=slice(self.LAT_SOUTH, self.LAT_NORTH), lon=slice(self.LON_WEST, self.LON_EAST))

            grid_lons, grid_lats = np.meshgrid(ds.lon.values, ds.lat.values)
            source_points = np.column_stack((grid_lats.flatten(), grid_lons.flatten()))
            target_points = np.column_stack((self.interp_lats.flatten(), self.interp_lons.flatten()))            
            for f in this_upscaled_fields.keys():
                if interp==True:
                    this_field = self.idw_interpolation_haversine(source_points, ds.isel(init_time=0).sel(channel=f).values.flatten(), target_points, radius=120000, std_dev=25000)
                    this_field = this_field.reshape(self.interp_lats.shape)   
                else:
                    this_field = ds.isel(init_time=0).sel(channel=f).values       
                this_upscaled_fields[f] = this_field                
                         
            for f in this_upscaled_derived_fields.keys():
                if f == 'LR75':
                    t500 = ds.isel(init_time=0).sel(channel='t500').values
                    t700 = ds.isel(init_time=0).sel(channel='t700').values
                    ht500 = ds.isel(init_time=0).sel(channel='z500').values/9.81
                    ht700 = ds.isel(init_time=0).sel(channel='z700').values/9.81
                    this_field = -(t700-t500)/(ht700-ht500)
                elif f == 'SHEAR500':
                    ushr = ds.isel(init_time=0).sel(channel='u500').values - ds.isel(init_time=0).sel(channel='u10m').values
                    vshr = ds.isel(init_time=0).sel(channel='v500').values - ds.isel(init_time=0).sel(channel='v10m').values
                    this_field = np.sqrt(ushr**2 + vshr**2)
                elif f == 'SHEAR850':
                    ushr = ds.isel(init_time=0).sel(channel='u850').values - ds.isel(init_time=0).sel(channel='u10m').values
                    vshr = ds.isel(init_time=0).sel(channel='v850').values - ds.isel(init_time=0).sel(channel='v10m').values
                    this_field = np.sqrt(ushr**2 + vshr**2)
                elif f == 'UV10':
                    u = ds.isel(init_time=0).sel(channel='u10m').values
                    v = ds.isel(init_time=0).sel(channel='v10m').values
                    this_field = np.sqrt(u**2 + v**2)

                if interp==True:
                    this_field = self.idw_interpolation_haversine(source_points, this_field.flatten(), target_points, radius=120000, std_dev=25000)
                    this_field = this_field.reshape(self.interp_lats.shape)      

                this_upscaled_derived_fields[f] = this_field
                            
            combined_this_upscaled_fields = {**this_upscaled_fields, **this_upscaled_derived_fields}        
        
        except Exception as e:
            print(e)
            return (fhr, combined_this_upscaled_fields)
        
        return {fhr: combined_this_upscaled_fields}
