#!/bin/bash
conda activate hgt 
if [ -n $1 ] ; then
    resultL=$1
else
    resultL="./"
fi
if [ -n $2 ] ; then
    gpuN=$2
else
    gpuN="2"
fi
echo $resultL
echo $gpuN
for i in {0..4}
do
    for j in {0..9}
    do
        python trainACM.py --device $gpuN --seed $j --label ../data/ACM/labels_5_fold_cross_validation_$i.pkl --result $resultL
    done
done
for i in {0..4}
do
    for j in {0..9}
    do
        python trainDBLP.py --device $gpuN --seed $j --label ../data/DBLP/labels_5_fold_cross_validation_$i.pkl --result $resultL
    done
done        
for i in {0..4}
do
    for j in {0..9}
    do
        python trainIMDB.py --device $gpuN --seed $j --label ../data/IMDB/labels_5_fold_cross_validation_$i.pkl --result $resultL
    done
done    
conda deactivate