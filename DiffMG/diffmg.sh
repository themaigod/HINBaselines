#!/bin/bash
conda activate diffmg
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
        python train_search_ACM.py --gpu $gpuN --seed $j --label ../data/ACM/labels_5_fold_cross_validation_$i.pkl --result $resultL
        python train_ACM.py --gpu $gpuN --seed $j --label ../data/ACM/labels_5_fold_cross_validation_$i.pkl --result $resultL
    done
done
for i in {0..4}
do
    for j in {0..9}
    do
        python train_search_DBLP.py --gpu $gpuN --seed $j --label ../data/DBLP/labels_5_fold_cross_validation_$i.pkl --result $resultL
        python train_DBLP.py --gpu $gpuN --seed $j --label ../data/DBLP/labels_5_fold_cross_validation_$i.pkl --result $resultL
    done
done        
for i in {0..4}
do
    for j in {0..9}
    do
        python train_search_IMDB.py --gpu $gpuN --seed $j --label ../data/IMDB/labels_5_fold_cross_validation_$i.pkl --result $resultL
        python train_IMDB.py --gpu $gpuN --seed $j --label ../data/IMDB/labels_5_fold_cross_validation_$i.pkl --result $resultL
    done
done    
conda deactivate