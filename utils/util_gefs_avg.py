import scipy.ndimage
import numpy as np
import copy
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
import xarray as xr

class GEFSAVGUpscaler:
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

    def gefs_upscale_forecast(self, datasource, upscaled_model_fields, upscaled_derived_fields, get_date, fhr):
        this_upscaled_fields = copy.deepcopy(upscaled_model_fields)
        this_upscaled_derived_fields = copy.deepcopy(upscaled_derived_fields)
        combined_this_upscaled_fields = {**this_upscaled_fields, **this_upscaled_derived_fields}
        
        try:
            ds = datasource(get_date)
            ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
            ds = ds.sortby(ds.lon).isel(lat=slice(None, None, -1))
            ds = ds.sel(lat=slice(self.LAT_SOUTH, self.LAT_NORTH), lon=slice(self.LON_WEST, self.LON_EAST))

            grid_lons, grid_lats = np.meshgrid(ds.lon.values, ds.lat.values)
            source_points = np.column_stack((grid_lats.flatten(), grid_lons.flatten()))
            target_points = np.column_stack((self.interp_lats.flatten(), self.interp_lons.flatten()))            
            for f in this_upscaled_fields.keys():
                this_field = self.idw_interpolation_haversine(source_points, ds.isel(time=0).sel(channel=f).values.flatten(), target_points, radius=120000, std_dev=25000)
                this_field = this_field.reshape(self.interp_lats.shape)                
                this_upscaled_fields[f] = this_field
            
            for f in this_upscaled_derived_fields.keys():
                if f == 'LR75':
                    t500 = ds.isel(time=0).sel(channel='t500').values
                    t700 = ds.isel(time=0).sel(channel='t700').values
                    ht500 = ds.isel(time=0).sel(channel='z500').values/9.80665
                    ht700 = ds.isel(time=0).sel(channel='z700').values/9.90665
                    this_field = -(t700-t500)/(ht700-ht500)
                elif f == 'SHEAR500':
                    ushr = ds.isel(time=0).sel(channel='u500').values - ds.isel(time=0).sel(channel='u10m').values
                    vshr = ds.isel(time=0).sel(channel='v500').values - ds.isel(time=0).sel(channel='v10m').values
                    this_field = np.sqrt(ushr**2 + vshr**2)
                elif f == 'SHEAR850':
                    ushr = ds.isel(time=0).sel(channel='u850').values - ds.isel(time=0).sel(channel='u10m').values
                    vshr = ds.isel(time=0).sel(channel='v850').values - ds.isel(time=0).sel(channel='v10m').values
                    this_field = np.sqrt(ushr**2 + vshr**2)
                elif f == 'UV10':
                    u = ds.isel(time=0).sel(channel='u10m').values
                    v = ds.isel(time=0).sel(channel='v10m').values
                    this_field = np.sqrt(u**2 + v**2)
                
                this_field = self.idw_interpolation_haversine(source_points, this_field.flatten(), target_points, radius=120000, std_dev=25000)
                this_field = this_field.reshape(self.interp_lats.shape)                
                
                this_upscaled_derived_fields[f] = this_field
            
            combined_this_upscaled_fields = {**this_upscaled_fields, **this_upscaled_derived_fields}
        
        except Exception as e:
            print('error: ',e)
            return (fhr, combined_this_upscaled_fields)
        
        return {fhr: combined_this_upscaled_fields}