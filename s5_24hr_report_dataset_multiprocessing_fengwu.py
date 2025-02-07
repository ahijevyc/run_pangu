import pandas as pd
import numpy as np
import sys, os, math
from datetime import *
import sqlite3, cartopy, pickle
from mpl_toolkits.basemap import *
#from get_osr_gridded_new import *
from cartopy.geodesic import Geodesic
import scipy.ndimage.filters
import xarray as xr
import pygrib
from scipy import spatial
from mpl_toolkits.basemap import *
import scipy.ndimage as ndimage
import pickle as pickle
import matplotlib.pyplot as plt
from utils.util import get_closest_gridbox, generate_date_list
import shutil
import yaml
import argparse
from pathlib import Path
import multiprocessing as mp
from functools import partial

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_closest_report_distances(grid_lats, grid_lons, thisdate, get_fhrs, df, report_types, geo, gmt2cst):
    sdate, edate = thisdate + timedelta(hours=int(get_fhrs[0])) - gmt2cst, thisdate + timedelta(hours=int(get_fhrs[-1])) - gmt2cst
    conn = sqlite3.connect('/glade/u/home/sobash/2013RT/REPORTS/reports_v20240820.db')
    c = conn.cursor()
    for type in report_types:
        if (type=='nonsigwind'):c.execute("SELECT slat, slon, datetime FROM reports_wind WHERE datetime > '%s' AND datetime <= '%s' AND mag < 65 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='nonsighail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' AND size < 2.00 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='sigwind'):c.execute("SELECT slat, slon, datetime FROM reports_wind WHERE datetime > '%s' AND datetime <= '%s' AND mag >= 65 AND mag <= 999 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='sighail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' AND size >= 2.00 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='wind'):c.execute("SELECT slat, slon, datetime FROM reports_wind WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (sdate,edate))
        elif (type=='hail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (sdate,edate))
        elif (type=='hailone'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' AND size >= 1.00 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='torn'):c.execute("SELECT slat, slon, datetime FROM reports_torn WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (sdate,edate))
        elif (type=='torn-one-track'):c.execute("SELECT slat, slon, datetime FROM reports_torn WHERE datetime > '%s' AND datetime <= '%s' AND sg == 1 ORDER BY datetime asc" % (sdate,edate))

        elif (type=='wind50'):c.execute("SELECT slat, slon, datetime FROM reports_wind WHERE datetime > '%s' AND datetime <= '%s' AND mag >= 50 AND mag <= 999 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='windmg'):c.execute("SELECT slat, slon, datetime FROM reports_wind WHERE datetime > '%s' AND datetime <= '%s' AND mag >= 50 AND mag <= 999 AND type == 'MG' ORDER BY datetime asc" % (sdate,edate))
        elif (type=='sigtor'):c.execute("SELECT slat, slon, datetime FROM reports_torn WHERE datetime > '%s' AND datetime <= '%s' AND rating >= 2 AND rating <= 5 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='torwarn'):c.execute("SELECT latcent, loncent, starttime FROM warnings WHERE starttime > '%s' AND starttime <= '%s' AND type == 'TO' ORDER BY starttime asc" % (sdate,edate))
        elif (type=='svrwarn'):c.execute("SELECT latcent, loncent, starttime FROM warnings WHERE starttime > '%s' AND starttime <= '%s' AND type == 'SV' ORDER BY starttime asc" % (sdate,edate))
        rpts = c.fetchall()

        # extract both start/end lat/lon (doesnt use end time)
        if type == 'torn':
            c.execute("SELECT slat, slon, datetime FROM reports_torn WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (sdate,edate))
            rpts_start = c.fetchall()
            c.execute("SELECT elat, elon, datetime FROM reports_torn WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (sdate,edate))
            rpts_end = c.fetchall()
            rpts = rpts_start + rpts_end    
        if len(rpts) > 0:
            report_lats, report_lons, report_times = list(zip(*rpts))
            report_times = np.array(report_times, dtype=np.datetime64) + np.timedelta64(6, "h")
            report_lats, report_lons = np.array(report_lats), np.array(report_lons)
        
        for time_tolerance in [24]:
            if len(rpts) == 0:
                df['%s_rptdist_%dhr'%(type,time_tolerance)] = -1
                continue

            all_distances = []
            for i in range(len(grid_lats)):
                these_report_lons, these_report_lats = report_lons, report_lats

                reports = list(zip(these_report_lons, these_report_lats))
                if len(reports) > 0:
                    t = geo.inverse((grid_lons[i], grid_lats[i]), reports)
                    t = np.asarray(t)

                    distances_km = t[:,0]/1000.0
                    closest_report_distance = np.amin(distances_km)
                else:
                    closest_report_distance = -1

                all_distances.append(int(closest_report_distance))

            df['%s_rptdist_%dhr'%(type,time_tolerance)] = all_distances
    
def maximum_filter_ignore_nan(data, footprint):
    nans = np.isnan(data)
    replaced = np.where(nans, -np.inf, data) #need to ignore nans, this *should* do it
    return scipy.ndimage.maximum_filter(replaced, footprint=footprint, mode='nearest')

def minimum_filter_ignore_nan(data, footprint):
    nans = np.isnan(data)
    replaced = np.where(nans, np.inf, data)
    return scipy.ndimage.minimum_filter(replaced, footprint=footprint, mode='nearest')

def process_single_date(thisdate, config, geo, ic):
    # Extract necessary variables from config
    output_rootdir = config['output_rootdir']
    nc_rootdir = config['nc_rootdir']
    dataset_folder = config['dataset_folder']
    output_folder_name = config['output_folder_name']
    start_fcst_day = config['start_fcst_day']
    end_fcst_day = config['end_fcst_day']
    fcst_hour_interval = config['fcst_hour_interval']
    report_types = config['report_types']
    fields_selected = config['fields_selected']
    gmt2cst = timedelta(hours=6)

    yyyymmdd = thisdate.strftime('%Y%m%d')
    yyyymmddhh = thisdate.strftime('%Y%m%d%H')
    init = thisdate.hour
    print(thisdate)
    fnames = []
    
    for i, dtdir in enumerate(dataset_folder):
        #fname = f'{nc_rootdir}/{dtdir}/{thisdate.strftime("%Y")}/{thisdate.strftime("%Y%m%d%H")}_{models[i]}_upscaled.nc'
        fname = f'{nc_rootdir}/{thisdate.strftime("%Y%m%d%H")}/{ic}/{thisdate.strftime("%Y%m%d%H")}_FengWu_{ic}_upscaled.nc'
        print(fname)
        if os.path.exists(fname):
            fnames.append(fname)
    
    if len(fnames) > 0:
        dss = [xr.open_dataset(fname) for fname in fnames]
        df_all = []
        for day in range(start_fcst_day, end_fcst_day+1):
            #get_fhrs = np.arange(12,37,fcst_hour_interval) + 24*(day-1)

            # 12z forecasts may not work?
            if day == 1 and init == 12: get_fhrs = np.array([6,12,18,24])
            elif day == 1 and init == 18: get_fhrs = np.array([6,12,18])
            else: get_fhrs = np.arange(12-init,37-init,fcst_hour_interval) + 24*(day-1)

            print(day, get_fhrs)
            
            upscaled_fields = {}
            for model_idx, (model, fields) in enumerate(fields_selected.items()):
                if len(fields) > 0:
                    for field in fields:
                        try:
                            print(f"Upscaling {model_idx} {model} {field}")
                            upscaled_fields[f"{model}_{field}"] = dss[model_idx][field].sel(fhr=get_fhrs).values
                        except KeyError:
                            print(f"Warning: Field '{field}' not found in dataset for model '{model}'. Skipping.")

            # Update the simple_mean_fields processing
            print('~~~ computing time and space averages ~~~')
            for k in simple_mean_fields:
                # print(k)
                for model in models:
                    if k in fields_selected[model]:
                        field_key = f"{model}_{k}"
                        if field_key in upscaled_fields:
                            for x in [5]:
                                for t in [1]:
                                    nanmask = np.isnan(upscaled_fields[field_key])
                                    kernel = np.ones((t,x,x))
                                    denom = scipy.ndimage.convolve(np.logical_not(nanmask).astype(np.float32), kernel, mode='nearest')
                                    upscaled_fields[f"{field_key}-N{x}T{t}-mean"] = scipy.ndimage.convolve(np.where(nanmask, 0, upscaled_fields[field_key]), kernel, mode='nearest') / denom                            
                                    # upscaled_fields[f"{field_key}-N{x}T{t}-max"] = maximum_filter_ignore_nan(np.where(nanmask, np.nan, upscaled_fields[field_key]), kernel)
                                    # upscaled_fields[f"{field_key}-N{x}T{t}-min"] = minimum_filter_ignore_nan(np.where(nanmask, np.nan, upscaled_fields[field_key]), kernel)
        
            print('#### masking ####')
            # only use grid points within mask, and remove first forecast hour (to match OSRs)
            for k2 in upscaled_fields:
                upscaled_fields[k2] = upscaled_fields[k2][:,mask]

            upscaled_fields['xind'] = x_ind[mask][np.newaxis,:]
            upscaled_fields['yind'] = y_ind[mask][np.newaxis,:]
            upscaled_fields['lat'] = lats[mask][np.newaxis,:]
            upscaled_fields['lon'] = lons[mask][np.newaxis,:]
            upscaled_fields['day'] = np.ones((1,rows))*day

            upscaled_fields_all, columns = [], []
            for k3 in upscaled_fields.keys():
                upscaled_fields_all.append( upscaled_fields[k3].T )
                if k3 not in ['xind', 'yind', 'lat', 'lon', 'day']: 
                    columns.extend( [ '%s-idx%d'%(k3,i) for i,h in enumerate(get_fhrs) ] )
                else: 
                    columns.extend( [ k3 ] )
                    
            upscaled_fields_all = np.hstack(upscaled_fields_all) #(conus80km_rows,numfeatures)

            # create pandas dataframe for all fields
            df = pd.DataFrame(upscaled_fields_all, columns=columns)
            df['Date'] = thisdate.strftime('%Y-%m-%d %H:%M:%S')

            print('#### obs ####')
            # Call get_closest_report_distances with necessary parameters
            get_closest_report_distances(df['lat'].values, df['lon'].values, thisdate, get_fhrs, df, report_types, geo, gmt2cst)
            # #get_lightning_obs()
            df_all.append(df)
    
        df = pd.concat(df_all)
        # df.dropna(subset=[col for col in df.columns if col.endswith('-idx0')], inplace=True)
        df.dropna(subset=[col for col in df.columns], inplace=True)
        
        #write_dir = f'{output_rootdir}{output_folder_name}/{thisdate.strftime("%Y")}'
        write_dir = f'{output_rootdir}/{thisdate.strftime("%Y%m%d%H")}/{ic}/'

        #outfile = os.path.join(write_dir,f'grid_data_{output_folder_name}_avg_d{str(start_fcst_day).zfill(2)}_24hr_{yyyymmddhh}-0000.par')
        outfile = os.path.join(write_dir,f'grid_data_fengwu_{ic}_avg_d{str(start_fcst_day).zfill(2)}_24hr_{yyyymmddhh}-0000.par')
        
        if len(df.index) > 0:
            print(outfile)
            os.makedirs(write_dir, exist_ok=True) 
            df.to_parquet(outfile)
            return True
    
    return False

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process configuration file')
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYYMMDD format')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYYMMDD format')
    parser.add_argument('--ic', type=str, required=True, help='Initial condition')

    args = parser.parse_args()

    # Get the absolute path of the script
    script_dir = Path(__file__).parent.absolute()

    # Check if the provided path is absolute
    if os.path.isabs(args.config_path):
        config_path = args.config_path
    else:
        # If it's a relative path, join it with the script directory
        config_path = os.path.join(script_dir, args.config_path)

    # Load the configuration
    config = load_config(config_path)

    # Access the variables
    output_rootdir = config['output_rootdir']
    nc_rootdir = config['nc_rootdir']
    dataset_folder = config['dataset_folder']
    output_folder_name = config['output_folder_name']
    nfr = config['nfr']

    #start_input_date = config['start_input_date']
    #end_input_date = config['end_input_date']
    start_input_date = args.start_date
    end_input_date = args.end_date
    assert (start_input_date[-2:] in ['00', '12']) and (end_input_date[-2:] in ['00', '12'])

    start_fcst_day = config['start_fcst_day']
    end_fcst_day = config['end_fcst_day']
    fcst_hour_interval = config['fcst_hour_interval']
    report_types = config['report_types']
    fields_selected = config['fields_selected']
    
    gmt2cst = timedelta(hours=6)

    # start and end date of the current script has to be identicle
    startdate = datetime.strptime(start_input_date, '%Y%m%d%H')
    enddate = datetime.strptime(end_input_date, '%Y%m%d%H')

    simple_mean_fields = []
    models = []
    # track unique entries for simple_mean_fields
    unique_fields = set()  
    for k, v in fields_selected.items():
        for field in v:
            if field not in unique_fields:
                unique_fields.add(field)  
                simple_mean_fields.append(field)
        models.append(k)
    # make sure these are 1s in the masked area if we want to pull out these values
    mask  = pickle.load(open('usmask_80km.pk', 'rb'))
    # mask = np.logical_not(mask)
    mask = mask.reshape((65,93)) 
    rows = mask.sum()
    assert rows != 0
            

    ## Take the mean lat lon of multiple datasets
    lons_list = []
    lats_list = []
    for k, v in fields_selected.items():
        print(k , v, len(v))
        if len(v)>0:
            _,lons,lats,_,_,_,_ = get_closest_gridbox(model = k, model_path='./nngridpts')
            lats_list.append(lats)
            lons_list.append(lons)
    lats = np.mean(np.dstack(lats_list),axis=-1)
    lons = np.mean(np.dstack(lons_list),axis=-1)
    y_ind, x_ind = np.indices(lats.shape)  
    
    # Initialize Geodesic object
    geo = Geodesic()

    # Generate list of dates to process
    date_list = []
    thisdate = startdate
    while thisdate <= enddate:
        date_list.append(thisdate)
        thisdate += timedelta(days=1)

    # Set up multiprocessing pool
    num_processes = 1  # Use all available CPU cores
    pool = mp.Pool(processes=num_processes)

    # Use partial to pass the config and geo to the process_single_date function
    process_func = partial(process_single_date, config=config, geo=geo, ic=args.ic)

    # Process dates in parallel
    results = pool.map(process_func, date_list)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    # Write the config file (only once, after all processing is done)
    #os.makedirs(f'{output_rootdir}{output_folder_name}', exist_ok=True) 
    #config_file_path = os.path.join(f'{output_rootdir}{output_folder_name}', 'preprocess_cfg.yaml')
    config_file_path = os.path.join(f'{output_rootdir}', startdate.strftime('%Y%m%d%H'), args.ic, 'preprocess_cfg.yaml')
    with open(config_file_path, 'w') as file:
        yaml.safe_dump(config, file, sort_keys=False)
    print(f'Wrote config file to {config_file_path}')
    # Count successful processes
    forecasts_processed = sum(results)
    print('forecasts processed', forecasts_processed)

