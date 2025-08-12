import numpy as np
from datetime import *
import math
import time, os, sys
import pickle as pickle
from scipy import spatial
from mpl_toolkits.basemap import *

def readNCLcm(name):
    '''Read in NCL colormap for use in matplotlib'''
    import os
    rgb, appending = [], False
    rgb_dir_ys = '/glade/apps/opt/ncl/6.2.0/intel/12.1.5/lib/ncarg/colormaps'
    rgb_dir_ch = '/glade/u/apps/ch/opt/ncl/6.4.0/intel/16.0.3/lib/ncarg/colormaps'
    if os.path.isdir(rgb_dir_ys): fh = open('%s/%s.rgb'%(rgb_dir_ys,name), 'r')
    else: fh = open('%s/%s.rgb'%(rgb_dir_ch,name), 'r')

    for line in fh.read().splitlines():
        if appending: rgb.append(map(float,line.split()))
        if ''.join(line.split()) in ['#rgb',';RGB']: appending = True
    maxrgb = max([ x for y in rgb for x in y ])
    if maxrgb > 1: rgb = [ [ x/255.0 for x in a ] for a in rgb ]
    return rgb

def get_closest_gridbox(model = None, model_path='./', data=None, lats=None, lons=None):
    if model is None:
        raise ValueError("model must be specified")
    gpfname = 'nngridpts_80km_%s'%model
    abs_path = os.path.abspath(model_path)    
    nngridpts_path = os.path.join(abs_path, gpfname)
    if os.path.exists(nngridpts_path) and data is None:
        try:
            with open(nngridpts_path, 'rb') as f:
                nngridpts, new_lons, new_lats, in_lons_proj, in_lats_proj, x81, y81 = pickle.load(f)
                return nngridpts, new_lons, new_lats, in_lons_proj, in_lats_proj, x81, y81
        except:
            raise ValueError("File does not exist or is corrupted")
    else:
        if lats is None or lons is None:
            raise ValueError("Please provide both lats and lons.")
        awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, 
                        urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, 
                        resolution=None, area_thresh=10000.)
    
        # Generate a grid of 93x65 points
        grid81 = awips.makegrid(93, 65, returnxy=True)
        x81, y81 = awips(grid81[0], grid81[1])
        
        # Calculate lats and lons for the new grid
        new_lons, new_lats = awips(x81, y81, inverse=True)
        
        # Convert lat/lon of original data to map projection coordinates
        lons_2d, lats_2d = np.meshgrid(lons, lats)
        xy = awips(lons_2d, lats_2d)
        
        tree = spatial.KDTree(list(zip(xy[0].ravel(),xy[1].ravel())))
        nngridpts = tree.query(list(zip(x81.ravel(),y81.ravel())))
        
        # Save nngridpts, new_lons, and new_lats
        with open('nngridpts_80km_%s'%model, 'wb') as f:
            pickle.dump((nngridpts, new_lons, new_lats, xy[0], xy[1], x81, y81), f)
    
    return nngridpts, new_lons, new_lats, xy[0], xy[1], x81, y81

def generate_date_list(start_input_date, end_date, hourinterval=24):
    # Convert to datetime object
    start_date = datetime.strptime(start_input_date, '%Y%m%d%H')
    end_date_dt = datetime.strptime(end_date, '%Y%m%d%H')

    # Generate list of datetime objects in 12-hour steps
    date_list = []
    while start_date <= end_date_dt:
        date_list.append(start_date.strftime('%Y%m%d%H'))
        start_date += timedelta(hours=hourinterval)

    return date_list


