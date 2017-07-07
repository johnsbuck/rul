#!/bin/sh

python import_data.py --d CMAPSS_TRAIN_2 --r 

maxmultiplier=$(sed -n 1p data/maxmultiplier.csv)
maxmultiplier=${maxmultiplier%.*}
echo $maxmultiplier

for i in $(seq 1 5 $maxmultiplier)
do
	python change_maxmul.py $i
	file_count=$(printf %03d $(((i/5)+1)))
	th src/doall.lua -batchSize 10 -epoch 200 -save frun_13_$file_count
done
