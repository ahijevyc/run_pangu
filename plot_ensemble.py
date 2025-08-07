import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap,BoundaryNorm
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import sys, pickle
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plotting for AI NWP ensemble severe output')
    parser.add_argument('itime', help='initialization time')
    parser.add_argument('model', default='panguweather', help='model')
    return parser.parse_args()

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

def plot_forecast(data2d, ofile='forecast.png'):
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
    plt.savefig(ofile, dpi=150)

def plot_forecast_grid(args, ds, plotdays=range(9), ofile=None):
    axes_proj = ccrs.LambertConformal(central_longitude=-100, central_latitude=37)
    
    fig_width = 9
    fig, axes = plt.subplots(3, 3, figsize=(fig_width,fig_width/1.45), subplot_kw={'projection': axes_proj})
    fig.subplots_adjust(hspace=0.05, wspace=0.06, bottom=0.05,top=0.95,left=0.05,right=0.95)
    axes = axes.flatten()

    ds = ds.sel(latitude=slice(53,22), longitude=slice(360-129, 360-65))
    lons, lats = ds['longitude'], ds['latitude']
    ds = ds["z"] / 9.81
    coastline = cfeature.COASTLINE.with_scale('50m')
    states = cfeature.STATES.with_scale('50m')
    for n, ax in tqdm(enumerate(axes[:-1]), total=len(plotdays)):
        day = n+1
        if day not in plotdays:
            continue
        for spine in ax.spines.values(): spine.set_linewidth(0.25)

        daytext1 = (ds.init_time.data + pd.Timedelta(days=day)).strftime('00 UTC %a %d %b')
        #daytext2 = (ds.init_time.data + pd.Timedelta(days=day)).strftime('12 UTC %a %d %b')
        a = ax.text(0, 1, r'$\bf{Day\ %d:}$ %s'%(day, daytext1), fontsize=6, ha='left', va='bottom', color='k', fontweight='normal', transform=ax.transAxes)
        #a = ax.text(0.01, 0.03, 'max:%d%%'%(100*max_prob), fontsize=6, ha='left', va='center', color='k', transform=ax.transAxes)

        ax.set_extent([-121,-74,25,50], ccrs.PlateCarree())
        ax.add_feature(coastline, linewidth=0.1, edgecolor="gray")
        ax.add_feature(states, linewidth=0.1, edgecolor="gray")
        #contour_colors = np.array([(128,128,128),(124,83,60),(248,206,75),(235,55,40),(222,63,233),(135,38,203)])/255.0
        #contour_colors = np.array([(128,128,128),(124,83,60),(248,206,75),(235,55,40),(234,51,237),(135,25,203)])/255.0

        contour_colors = ['blue', 'green', 'red']

        for m in tqdm(ds.mem):
            this_mem = ds.sel(mem=m, step=pd.Timedelta(days=day))
            ax.contour(lons, lats, this_mem, levels=[5520,5760,5880], colors=contour_colors, \
                       linewidths=[0.5], transform=ccrs.PlateCarree(), alpha=0.3)
        mean_field = ds.sel(step=pd.Timedelta(days=day)).mean(dim="mem")
        ax.contour(lons, lats, mean_field, levels=[5520,5760,5880], colors='k', \
                   linewidths=[1.5], transform=ccrs.PlateCarree())

    for spine in axes[8].spines.values(): spine.set_linewidth(0)
    #cax = axes[8].inset_axes([0.05, 0.9, 0.9, 0.05])
    #cbar = fig.colorbar(a, cax=cax, orientation='horizontal')
    #cbar.ax.tick_params(labelsize=6, length=0, width=0, color='k') 
    #cbar.ax.set_xticklabels(['0.5%','5%','15%','30%','45%','60%','']) 

    ictext = args.ic + ' ICs'

    daytext1 = ds.init_time.dt.strftime('00 UTC %A %d %B %Y').item()
    axes[8].text(0.02,0.7, 'Init: %s'%daytext1, fontsize=8, ha='left', va='center', color='k', transform=axes[8].transAxes)
    axes[8].text(0.02,0.63, 'Model: %s, IC: %s'%(args.model.upper(),ictext), fontsize=6, ha='left', va='center', color='k', transform=axes[8].transAxes)

    if ofile:
        plt.savefig(ofile, dpi=150, bbox_inches='tight')
        print(ofile)

    return fig

##################################
mask  = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))
thismask = mask.flatten()

import re
def parsemem(ds):
    filename = ds.encoding["source"]
    # get member substring
    match = re.search(r'/[cp]\d\d/', filename)
    mem = match.group(0).strip('/')
    # assign mem to coordinate
    ds = ds.assign_coords(mem=[mem])
    return ds

def main():
    args = parse_arguments()

    day = 1
    itime = pd.to_datetime(args.itime)
    yyyymmddhh = itime.strftime("%Y%m%d%H")
    model = args.model
    print('making graphics for', itime, model)

    odir = Path(f'/glade/derecho/scratch/ahijevyc/ai-models/output/{model}/{yyyymmddhh}')
    ifiles = list(odir.glob('ge[pc][0-9][0-9].grib'))
    if len(ifiles):
        print(f"open {len(ifiles)} grib files")
        ds = xr.open_mfdataset(
            ifiles,
            engine="cfgrib",
            filter_by_keys={
                "typeOfLevel": "isobaricInhPa",
                "level": 500,
            },
            backend_kwargs={"indexpath": ""},
            concat_dim="mem",
            combine="nested",
        )
        args.ic = "GEFS"
    else:
        ifiles = odir.glob("[cp]*/*.nc")
        def daymultiple(f):
            fhr = f.name[-6:-3]  # fhr part
            return int(fhr) % 24 == 0
        ifiles = [f for f in ifiles if daymultiple(f)]
        print(f"open {len(ifiles)} nc files")
        ds = xr.open_mfdataset(ifiles, preprocess=parsemem).rename(lat="latitude", lon="longitude", prediction_timedelta="step").squeeze(dim="init_time")
        ds = ds.sel(channel="z500").rename(__xarray_dataarray_variable__="z")
        args.ic = "ECMWF"

    plot_forecast_grid(args, ds, ofile = odir / f'predictions_grid_{model}_spag_z500_{yyyymmddhh}.png')



#for hazard in ['any']:
#    for day in [1,2,3,4,5,6,7,8]:
#        print(hazard, day)
#        plot_forecast( ds['prob'].sel(hazard=hazard,day=day).values, 'predictions_%s_%s_day%d_%s_%s.png'%(model,mem,day,hazard,yyyymmddhh) )


if __name__ == '__main__':
    main()
