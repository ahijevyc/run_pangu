import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.basemap import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap,BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np
import datetime as dt
import sys, pickle
#import pygrib

def get_obs():
    ds = xr.open_mfdataset('/glade/work/sobash/NSC_objects/HRRR_new/obs_new/ml_obs_dist_2024050100.nc')
    time_win, space_win = 0, 40

    if hazard == 'all': 
        labels   = ((ds['hailone_rptdist_%dhr'%time_win] < space_win) & (ds['hailone_rptdist_%dhr'%time_win] > 0)) |  \
                   ((ds['wind_rptdist_%dhr'%time_win] < space_win) & (ds['wind_rptdist_%dhr'%time_win] > 0)) | \
                   ((ds['torn_rptdist_%dhr'%time_win] < space_win) & (ds['torn_rptdist_%dhr'%time_win] > 0))
    elif hazard == 'wind':    labels  =  ((ds['wind_rptdist_%dhr'%time_win] < space_win) & (ds['wind_rptdist_%dhr'%time_win] > 0))
    elif hazard == 'hailone': labels  =  ((ds['hailone_rptdist_%dhr'%time_win] < space_win) & (ds['hailone_rptdist_%dhr'%time_win] > 0))
    elif hazard == 'torn':    labels  =  ((ds['torn_rptdist_%dhr'%time_win] < space_win) & (ds['torn_rptdist_%dhr'%time_win] > 0))
    elif hazard == 'sighail': labels  =  ((ds['sighail_rptdist_%dhr'%time_win] < space_win) & (ds['sighail_rptdist_%dhr'%time_win] > 0))
    elif hazard == 'sigwind': labels  =  ((ds['sigwind_rptdist_%dhr'%time_win] < space_win) & (ds['sigwind_rptdist_%dhr'%time_win] > 0))

    # retrieve specific forecast hour range and apply mask (obs files are unmasked)
    labels = labels.sel(hr=slice(1,204))
    labels = labels.stack(pt=("y", "x"))
    labels = labels.where( xr.DataArray(thismask, dims=("pt")) , drop=True)
    return labels

def readNCLcm(name):
    '''Read in NCL colormap for use in matplotlib'''
    rgb, appending = [], False
    rgb_dir_ch = '/glade/u/apps/opt/ncl/6.6.2/lib/ncarg/colormaps/'
    fh = open('%s/%s.rgb'%(rgb_dir_ch,name), 'r')

    for line in list(fh.read().splitlines()):
        if appending: rgb.append(list(map(float,line.split())))
        if ''.join(line.split()) in ['#rgb',';RGB']: appending = True
    maxrgb = max([ x for y in rgb for x in y ])
    if maxrgb > 1: rgb = [ [ x/255.0 for x in a ] for a in rgb ]
    return rgb

def brier_skill_score(obs, preds):
    bs = K.mean((preds - obs) ** 2)
    obs_climo = K.mean(obs, axis=0) # use each observed class frequency instead of 1/nclasses. Only matters if obs is multiclass.
    bs_climo = K.mean((obs - obs_climo) ** 2)
    bss = 1.0 - (bs/bs_climo+K.epsilon())
    return bss

def grid_data(field):
    # convert 1d array into 4d array with shape (num_dates, num_fhr, 65, 93)
    gridded_field = np.zeros((num_dates,num_fhr,65*93), dtype=np.float64)
    field = field.reshape((num_dates, num_fhr, -1))

    for i, dt in enumerate(unique_forecasts):
        for j, f in enumerate(unique_fhr):
            gridded_field[i,j,thismask] = field[i,j,:]

    return gridded_field.reshape((num_dates, num_fhr, 65, 93))

def make_gridded_forecast(predictions, labels, dates, fhr):
    ### reconstruct into grid by day (mask makes things more complex than a simple reshape)
    gridded_predictions = np.zeros((num_dates,num_fhr,65*93), dtype=np.float64)
    gridded_labels      = np.zeros((num_dates,num_fhr,65*93), dtype=np.float64)

    # just grid predictions for this class
    predictions = predictions.reshape((num_dates, num_fhr, -1))
    labels      = labels.reshape((num_dates, num_fhr, -1))

    for i, dt in enumerate(unique_forecasts):
        for j, f in enumerate(unique_fhr):
            gridded_predictions[i,j,thismask] = predictions[i,j,:]
            gridded_labels[i,j,thismask]      = labels[i,j,:]
        #print(dt, gridded_predictions[i,:].max())

    # return only predictions for US points
    return (gridded_predictions.reshape((num_dates, num_fhr, 65, 93)), gridded_labels.reshape((num_dates, num_fhr, 65, 93)))

def smooth_gridded_forecast(predictions_gridded):
    smoothed_predictions = []
    dim = predictions_gridded.shape
    for k,s in enumerate(smooth_sigma):
        if len(dim) == 4: smoothed_predictions.append(gaussian_filter(predictions_gridded, sigma=[0,0,s,s]))
        if len(dim) == 3: smoothed_predictions.append(gaussian_filter(predictions_gridded, sigma=[0,s,s]))

    # return only predictions for US points
    return np.array(smoothed_predictions)

def plot_forecast(data2d, fname='forecast.png'):
    #fig, axes, m = pickle.load(open('../rt2015_ch_CONUS.pk', 'rb'))
    #xgrid, ygrid = m(lons, lats) #80 km grid points on new map projection   
    data2d = data2d.flatten()[thismask]
    print(data2d.max())

    fig = plt.figure(figsize=(10,6))
    #axes_proj = ccrs.PlateCarree(central_longitude=180)
    axes_proj = ccrs.LambertConformal(central_longitude=-100, central_latitude=37)
    axes = plt.axes(projection=axes_proj)

    axes.set_extent([-121,-74,25,50], ccrs.PlateCarree())
    axes.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.2)
    axes.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.1)

    these_lons, these_lats = lons.flatten()[thismask], lats.flatten()[thismask]
    pts = axes_proj.transform_points(ccrs.PlateCarree(), these_lons, these_lats)
    xgrid, ygrid, zgrid = pts[:,0], pts[:,1], pts[:,2]

    # turn into 2d field
    gridded_field = np.zeros((65*93), dtype=np.float32)
    gridded_field[thismask] = data2d
    gridded_field = gridded_field.reshape((65,93))

    #xcorner = (x[1:,1:] + x[:-1,:-1])/2.0
    #ycorner = (y[1:,1:] + y[:-1,:-1])/2.0
    
    if hazard=='torn':
        test = readNCLcm('MPL_Greys')[50::] + [[1,1,1]] + readNCLcm('MPL_Reds')[20::]
        cmap = ListedColormap(test)
        norm = BoundaryNorm(np.arange(0,0.5,0.05), ncolors=cmap.N, clip=True)
        plot_thresh = 0.02
    else:
        test = readNCLcm('MPL_Greys')[35::] + [[1,1,1]] + readNCLcm('MPL_Reds')[20::]
        cmap = ListedColormap(test)
        norm = BoundaryNorm(np.arange(0,1.1,0.1), ncolors=cmap.N, clip=True)
        plot_thresh = 0.05

    xflat, yflat= xgrid.flatten(), ygrid.flatten()

    print(data2d.shape)
    for i,b in enumerate(data2d.flatten()):
        color = cmap(norm([b])[0])

        #if not np.isnan(b) and not np.isinf(b) and thismask[i] and b>plot_thresh:
        if not np.isnan(b) and not np.isinf(b) and b>plot_thresh:
            val = int(round(b*100))
            val = int(b*100)
            #a = axes.text(xflat[i], yflat[i], val, fontsize=8, ha='center', va='center', family='monospace', color='0.2', fontweight='bold')
            a = axes.text(xflat[i], yflat[i], val, fontsize=6, ha='center', va='center', family='monospace', color=color, fontweight='bold')
            #a = axes.text(lons[:1308][i], lats[:1308][i], val, fontsize=8, ha='center', va='center', family='monospace', color=color, fontweight='bold', transform=ccrs.PlateCarree())
        #if labels_all[i]>0: a = axes.scatter(xflat[i], yflat[i], s=25, color='black', marker='o', alpha=0.5)

    if hazard in ['torn','sigwind','sighail']:
            contour_colors = np.array([(0,0,0),(61,128,40),(124,83,60),(248,206,75),(235,55,40),(234,67,247)])/255.0
            a = axes.contour(lons, lats, gridded_field, levels=[0.005,0.02,0.05,0.1,0.15,0.3], colors=contour_colors, \
                           linewidths=1.5, transform=ccrs.PlateCarree())
    else:
            #contour_colors = np.array([(0,0,0),(124,83,60),(248,206,75),(235,55,40),(222,63,233),(135,38,203)])/255.0
            contour_colors = np.array([(124,83,60),(0,0,0),(248,206,75),(0,0,0),(0,0,0),(235,55,40),(0,0,0),(0,0,0),(222,63,233),(0,0,0),(0,0,0),(135,38,203)])/255.0
            linewidths = [1.5,0.5,1.5,0.5,0.5,1.5,0.5,0.5,1.5,0.5,0.5,1.5]
            #a = axes.contour(lons, lats, gridded_field, levels=[0.005,0.05,0.15,0.3,0.45,0.60], colors=contour_colors, \
            a = axes.contour(lons, lats, gridded_field, levels=[0.05,0.10,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.60], colors=contour_colors, \
                           linewidths=linewidths, transform=ccrs.PlateCarree())
           
    plt.tight_layout()
    plt.savefig(fname, dpi=150)

def plot_forecast_grid(mean_predictions, fname='test.png'):
    hazards = ['sighail', 'sigwind', 'hail', 'wind', 'torn', 'all', 'cg'] 
    plot_thresh = [0.02, 0.02, 0.05, 0.05, 0.02, 0.05, 0.05]
    axes_proj = ccrs.LambertConformal(central_longitude=-100, central_latitude=37)
    
    fig, axes = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(10,7), subplot_kw={'projection': axes_proj})
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    axes = axes.flatten()
    
    pts = axes_proj.transform_points(ccrs.PlateCarree(), lons.flatten()[thismask], lats.flatten()[thismask])
    xgrid, ygrid, zgrid = pts[:,0], pts[:,1], pts[:,2]
    xflat, yflat= xgrid.flatten(), ygrid.flatten()

    for n in range(len(hazards)):
        hazard = hazards[n]

        # grab data for this hazard and compute 24-hr maximum
        predictions_max_flat = np.amax( mean_predictions[0,12:36,:,n], axis=0 ).flatten()
        max_prob = predictions_max_flat.max()
   
        # turn into 2d field 
        gridded_field = np.zeros((65*93), dtype=np.float32)
        gridded_field[thismask] = predictions_max_flat
        gridded_field = gridded_field.reshape((65,93))

        # smooth field if desired
        gridded_field = gaussian_filter(gridded_field, sigma=[0.75,0.75])

        ax = axes[n]
        #axes = plt.axes(projection=axes_proj)
        a = ax.text(0.05, 0.1, hazard, fontsize=10, ha='left', va='center', family='monospace', color='k', fontweight='bold', transform=ax.transAxes)
        a = ax.text(0.03, 0.02, 'max prob: %d'%(100*max_prob), fontsize=5, ha='left', va='center', family='monospace', color='k', fontweight='bold', transform=ax.transAxes)

        ax.set_extent([-121,-74,25,50], ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.2)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.1)

        if hazard=='torn1':
            test = readNCLcm('MPL_Greys')[50::] + [[1,1,1]] + readNCLcm('MPL_Reds')[20::]
            cmap = ListedColormap(test)
            norm = BoundaryNorm(np.arange(0,0.5,0.05), ncolors=cmap.N, clip=True)
        else:
            test = readNCLcm('MPL_Greys')[35::] + [[1,1,1]] + readNCLcm('MPL_Reds')[20::]
            cmap = ListedColormap(test)
            norm = BoundaryNorm(np.arange(0,1.1,0.1), ncolors=cmap.N, clip=True)

        for i,b in enumerate(predictions_max_flat):
            color = cmap(norm([b])[0])

            if not np.isnan(b) and not np.isinf(b) and b>plot_thresh[n]:
                val = int(b*100)
                # plot numbers
                a = ax.text(xflat[i], yflat[i], val, fontsize=5, ha='center', va='center', family='monospace', color=color, fontweight='bold')

                # plot dots
                #a = ax.scatter(xflat[i], yflat[i], s=5, color=color)

        if hazard in ['torn','sigwind','sighail']:
            contour_colors = np.array([(0,0,0),(61,128,40),(124,83,60),(248,206,75),(235,55,40),(234,67,247)])/255.0
            a = ax.contour(lons, lats, gridded_field, levels=[0.005,0.02,0.05,0.1,0.15,0.3], colors=contour_colors, \
                           linewidths=1.5, transform=ccrs.PlateCarree())
        else:
            contour_colors = np.array([(0,0,0),(124,83,60),(248,206,75),(235,55,40),(222,63,233),(135,38,203)])/255.0
            a = ax.contour(lons, lats, gridded_field, levels=[0.005,0.05,0.15,0.3,0.45,0.60], colors=contour_colors, \
                           linewidths=1.5, transform=ccrs.PlateCarree())
           
        for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(0.1)
 

    plt.tight_layout()
    plt.savefig(fname, dpi=300)

##################################
mask  = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))
#mask  = pickle.load(open('../usamask_gefs.pk', 'rb'))
thismask = mask.flatten()

awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
lons, lats = awips.makegrid(93, 65, returnxy=False)

hazard = 'any'
day = 1
thisdate = dt.datetime.strptime(sys.argv[1], '%Y%m%d%H')
yyyymmddhh = thisdate.strftime('%Y%m%d%H')
mem = sys.argv[2]

all_ds = []
for mem in [ 'ens%d'%d for d in range(0,51) ]:
    this_ds = xr.open_dataset('/glade/derecho/scratch/sobash/pangu_realtime/%s/%s/forecast_%s_%s.nc'%(yyyymmddhh,mem,mem,yyyymmddhh))
    all_ds.append(this_ds)
ds = xr.concat(all_ds, dim='initialization_time')
ds = ds.mean(dim='initialization_time')

#ds = xr.open_dataset('/glade/derecho/scratch/sobash/pangu_realtime/%s/%s/forecast_%s_%s.nc'%(thisdate.strftime('%Y%m%d%H'),mem,mem,thisdate.strftime('%Y%m%d%H')))
print(ds)
lons, lats = ds['lon'].values, ds['lat'].values

#for hazard in ['all', 'wind', 'wind50', 'torn', 'hailone', 'cg', 'sigtor']:
for hazard in ['any']:
    for day in [1,2,3,4,5,6,7,8]:
        print(hazard, day)
        #plot_forecast( ds['prob'].isel(initialization_time=0).sel(hazard=hazard,day=day).values, 'predictions_%s_day%d_%s_%s.png'%(mem,day,hazard,thisdate.strftime('%Y%m%d%H')) )
        plot_forecast( ds['prob'].sel(hazard=hazard,day=day).values, 'predictions_%s_day%d_%s_%s.png'%(mem,day,hazard,thisdate.strftime('%Y%m%d%H')) )

#mean_predictions = ds['prob'].values
#plot_forecast_grid( mean_predictions, 'predictions_grid_%s.png'%(thisdate.strftime('%Y%m%d%H')) )
