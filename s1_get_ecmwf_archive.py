import datetime, sys, os
from ecmwf.opendata import Client
from ecmwfapi import ECMWFService

os.environ['ECMWF_API_KEY'] = "bbfde4b05540eefc786ab64c9f4ea8b7"
os.environ['ECMWF_API_URL'] = "https://api.ecmwf.int/v1"
os.environ['ECMWF_API_EMAIL'] = "sobash@ucar.edu"
#{
#    "url"   : "https://api.ecmwf.int/v1",
#    "key"   : "bbfde4b05540eefc786ab64c9f4ea8b7",
#    "email" : "sobash@ucar.edu"
#}

thisdate = datetime.datetime.strptime(sys.argv[1], '%Y%m%d%H')
ic_source = sys.argv[2]
yyyymmddhh = thisdate.strftime('%Y%m%d%H')
outdir = os.getenv('SCRATCH') + '/pangu_realtime/%s/%s/'%(yyyymmddhh,ic_source)


if ic_source == 'hres':
    server = ECMWFService("mars")
    server.execute( {
        "class": "od",
        "date": thisdate.strftime('%Y%m%d'),
        "time": thisdate.strftime('%H'),
        "stream": "oper",
        "type": "fc",
        "expver": "1",
        "param": ["gh","q","t","u","v","msl","2t","10u","10v"],
        "step": "0"
    },
        "%s/hres_analysis_%s.grib2"%(outdir,yyyymmddhh),
    )

elif ic_source[0:3] == 'ens':
    client = Client(source="ecmwf")
    mem = int(ic_source[3:])
    if mem < 1:
        client.retrieve(
            date=thisdate.strftime('%Y%m%d'),
            time=thisdate.strftime('%H'),
            stream="enfo",
            type="cf",
            param=["gh","q","t","u","v","msl","2t","10u","10v"],
            step=0,
            target="%s/%s_analysis_%s.grib2"%(outdir,ic_source,yyyymmddhh),
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
            target="%s/%s_analysis_%s.grib2"%(outdir,ic_source,yyyymmddhh),
        )
