#!/bin/csh

set yyyymmddhh=$1
set model=$2
set hh=`echo $yyyymmddhh | cut -c9-10`
set odir=$SCRATCH/ai-models/output/$model/$yyyymmddhh
mkdir -p $odir
foreach mem (`seq -w 00 30`)
    if ($mem == 00) then
        set mem=c00
    else
        set mem=p$mem
    endif

    set ofile=$odir/ge$mem.grib
    if (-e $ofile) continue
    ai-models --input file --file $SCRATCH/ai-models/input/$yyyymmddhh/$mem/ge$mem.t${hh}z.pgrb.0p25.f000 --assets $SCRATCH/ai-models --output file --path $ofile $model

end
