#!/bin/bash
conda activate magnn 
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
for p in 0.1 0.25 0.5
do
    mkdir -p "$resultL$p/"
    for i in {0..4}
    do
        for j in {0..9}
        do
            python run_ACM_ours_DBLP.py --device $gpuN --seed $j --label ../data/ACM/$p/${p}_labels_5_fold_cross_validation_$i.pkl --result "$resultL$p/"
        done
    done
    for i in {0..4}
    do
        for j in {0..9}
        do
            python run_DBLP.py --device $gpuN --seed $j --label ../data/DBLP/$p/${p}_labels_5_fold_cross_validation_$i.pkl --result "$resultL$p/"
        done
    done        
    for i in {0..4}
    do
        for j in {0..9}
        do
            python run_IMDB_ours.py --device $gpuN --seed $j --label ../data/IMDB/$p/${p}_labels_5_fold_cross_validation_$i.pkl --result "$resultL$p/"
        done
    done    
done
conda deactivate