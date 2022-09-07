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
for p in 0.1 0.25 0.5
do
    mkdir -p "$resultL$p/"
    for i in {0..4}
    do
        for j in {0..9}
        do
            python trainACM.py --device $gpuN --seed $j --label ../data/ACM/$p/${p}_labels_5_fold_cross_validation_$i.pkl --result "$resultL$p/"
        done
    done
    for i in {0..4}
    do
        for j in {0..9}
        do
            python trainDBLP.py --device $gpuN --seed $j --label ../data/DBLP/$p/${p}_labels_5_fold_cross_validation_$i.pkl --result "$resultL$p/"
        done
    done        
    for i in {0..4}
    do
        for j in {0..9}
        do
            python trainIMDB.py --device $gpuN --seed $j --label ../data/IMDB/$p/${p}_labels_5_fold_cross_validation_$i.pkl --result "$resultL$p/"
        done
    done    
done
conda deactivate