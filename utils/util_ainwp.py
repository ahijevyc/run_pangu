import numpy as np
import copy
import xarray as xr
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

wxformer_channel = ['qL137',
 'qL136',
 'qL130',
 'qL120',
 'qL110',
 'qL105',
 'qL100',
 'qL95',
 'qL90',
 'qL80',
 'qL70',
 'qL60',
 'qL50',
 'qL40',
 'qL30',
 'qL10',
 'uL137',
 'uL136',
 'uL130',
 'uL120',
 'uL110',
 'uL105',
 'uL100',
 'uL95',
 'uL90',
 'uL80',
 'uL70',
 'uL60',
 'uL50',
 'uL40',
 'uL30',
 'uL10',
 'vL137',
 'vL136',
 'vL130',
 'vL120',
 'vL110',
 'vL105',
 'vL100',
 'vL95',
 'vL90',
 'vL80',
 'vL70',
 'vL60',
 'vL50',
 'vL40',
 'vL30',
 'vL10',
 'tL137',
 'tL136',
 'tL130',
 'tL120',
 'tL110',
 'tL105',
 'tL100',
 'tL95',
 'tL90',
 'tL80',
 'tL70',
 'tL60',
 'tL50',
 'tL40',
 'tL30',
 'tL10',
 'sp',
 't2m',
 'z500',
 'u500',
 'q500',
 'v500',
 't500',]
# Define the channel mapping
variable_mapping = {
    'specific_humidity': 'q',
    'temperature': 't',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
    'surface_pressure': 'sp',
    '2m_temperature': 't2m',
    'v500': 'v500',
    'u500': 'u500',
    't500': 't500',
    'z500': 'z500',
    'q500': 'q500',
}

class AinwpUpscaler:
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

    def ainwp_upscale_forecast(self, datasource, upscaled_model_fields, upscaled_derived_fields, get_date, fhr):
        this_upscaled_fields = copy.deepcopy(upscaled_model_fields)
        this_upscaled_derived_fields = copy.deepcopy(upscaled_derived_fields)
        combined_this_upscaled_fields = {**this_upscaled_fields, **this_upscaled_derived_fields}
        
        try:
            ds = datasource(get_date, fhr)
            ds = ds.compute()
            
            ### below is the part to format the data ####            
            # Create an empty dictionary to store DataArrays
            data_arrays = {} 
                                
            # Populate the data arrays
            for var_name, abbr in variable_mapping.items():
                if abbr in ['sp', 'v500', 'u500', 'z500','t500','q500','t2m',]:
                    # Surface variables
                    data_arrays[abbr] = xr.DataArray(ds[var_name].values, dims=['lat', 'lon'])
                else:
                    # Variables with levels
                    for level in [137, 136, 130, 120, 110, 105, 100,  95,  90,  80,  70,  60,  50,  40, 30,  10]:
                        channel = f"{abbr}L{level}"
                        if channel in list(upscaled_model_fields.keys()):
                            data_arrays[channel] = xr.DataArray(ds[var_name].sel(level=level).values, dims=['lat', 'lon'])
            
            # Create coordinates
            lats = ds.latitude.values
            lons = ds.longitude.values
            time = get_date
            grid_lons, grid_lats = np.meshgrid(lons, lats)
            source_points = np.column_stack((grid_lats.flatten(), grid_lons.flatten()))
            target_points = np.column_stack((self.interp_lats.flatten(), self.interp_lons.flatten()))
            # Create the dataset
            ds = xr.Dataset(data_arrays)

            # Add coordinates
            ds = ds.assign_coords(time=time, lat=lats, lon=lons)

            # Expand the time dimension and transpose
            ds = ds.expand_dims('time').transpose('time', 'lat', 'lon')
            ### End of the part to format the data ####
            # print(ds)
            for f in this_upscaled_fields.keys():                
                this_field = self.idw_interpolation_haversine(source_points, ds.isel(time=0)[f].values.flatten(), target_points, radius=120000, std_dev=25000)
                this_field = this_field.reshape(self.interp_lats.shape)                
                this_upscaled_fields[f] = this_field        
            for f in this_upscaled_derived_fields.keys():
                if f == 'LR75':
                    raise NotImplementedError()
                    t500 = ds.isel(time=0)['t500'].values
                    t700 = ds.isel(time=0)['tL105'].values
                    ht500 = ds.isel(time=0)['z500'].values/9.80665
                    ht700 = ds.isel(time=0)['z700'].values/9.80665
                    this_field = -(t700-t500)/(ht700-ht500)
                elif f == 'SHEAR500':
                    ushr = ds.isel(time=0)['u500'].values - ds.isel(time=0)['uL137'].values
                    vshr = ds.isel(time=0)['v500'].values - ds.isel(time=0)['vL137'].values
                    this_field = np.sqrt(ushr**2 + vshr**2)
                elif f == 'SHEARL110':
                    ushr = ds.isel(time=0)['uL110'].values - ds.isel(time=0)['uL137'].values
                    vshr = ds.isel(time=0)['vL110'].values - ds.isel(time=0)['vL137'].values
                    this_field = np.sqrt(ushr**2 + vshr**2)
                elif f == 'UVL137':
                    u = ds.isel(time=0)['uL137'].values
                    v = ds.isel(time=0)['vL137'].values
                    this_field = np.sqrt(u**2 + v**2)
                
                this_field = self.idw_interpolation_haversine(source_points, this_field.flatten(), target_points, radius=120000, std_dev=25000)
                this_field = this_field.reshape(self.interp_lats.shape)   
                this_upscaled_derived_fields[f] = this_field
            
            combined_this_upscaled_fields = {**this_upscaled_fields, **this_upscaled_derived_fields}
        
        except Exception as e:
            print(e)
            return (fhr, combined_this_upscaled_fields)
        
        return {fhr: combined_this_upscaled_fields}
