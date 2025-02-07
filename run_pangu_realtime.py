#!/usr/bin/env python

from datetime import *
from subprocess import *
import os, time, sys

members = ['hres'] + [ 'ens%d'%d for d in range(0,51) ]

thisdate = datetime.strptime(sys.argv[1], '%Y%m%d%H')
yyyymmddhh = thisdate.strftime('%Y%m%d%H')

for mem in members:
    geyser_script = '/glade/work/sobash/run_pangu/run_pangu_severe_forecast.sh'

    command = "/opt/pbs/bin/qsub -v yyyymmddhh=%s,ic=%s %s"%(yyyymmddhh, mem, geyser_script)
    command = command.split()

    fout = open('/dev/null', 'w')
    #pid = Popen(execute, stdout=fout, stderr=fout).pid
    print(time.ctime(time.time()),':', 'Running', ' '.join(command))
    call(command, stdout=fout, stderr=fout)
