from ecmwf.opendata import Client
import datetime, sys, os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Download and process data for a date range')
    parser.add_argument('--start_date', type=str, required=True, 
                        help='Start date in YYYYMMDD format')
    parser.add_argument('--end_date', type=str, required=True,
                        help='End date in YYYYMMDD format')
    parser.add_argument('--output_dir', type=str, default='input_data',
                        help='Base output directory for saving files')
    parser.add_argument('--ic', type=str, default='hres',
                        help='Initial condition')
    return parser.parse_args()

args = parse_arguments()
thisdate = datetime.datetime.strptime(args.start_date, '%Y%m%d%H')
yyyymmddhh = thisdate.strftime('%Y%m%d%H')
outdir = args.output_dir

client = Client(source="ecmwf")

if args.ic == 'hres':
    client.retrieve(
        date=thisdate.strftime('%Y%m%d'),
        time=thisdate.strftime('%H'),
        stream="oper",
        type="fc",
        param=["gh","q","t","u","v","msl","2t","10u","10v"],
        step=0,
        target="%s/hres_analysis_%s.grib2"%(outdir,yyyymmddhh),
    )

elif args.ic[0:3] == 'ens':
    mem = int(args.ic[3:])
    if mem < 1:
        client.retrieve(
            date=thisdate.strftime('%Y%m%d'),
            time=thisdate.strftime('%H'),
            stream="enfo",
            type="cf",
            param=["gh","q","t","u","v","msl","2t","10u","10v"],
            step=0,
            target="%s/%s_analysis_%s.grib2"%(outdir,args.ic,yyyymmddhh),
        )
    else:
        client.retrieve(
            date=thisdate.strftime('%Y%m%d'),
            time=thisdate.strftime('%H'),
            stream="enfo",
            type="pf",
            number=mem,
            param=["gh","q","t","u","v","msl","2t","10u","10v"],
            step=0,
            target="%s/%s_analysis_%s.grib2"%(outdir,args.ic,yyyymmddhh),
        )
