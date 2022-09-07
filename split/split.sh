#!/bin/bash
conda activate han
python try_split.py --label_path ../data/IMDB/labels.pkl --output_label_path_style ../data/IMDB/labels_{k}_fold_cross_validation_{number}.pkl
python try_split.py --label_path ../data/ACM/labels.pkl --output_label_path_style ../data/ACM/labels_{k}_fold_cross_validation_{number}.pkl
python try_split.py --label_path ../data/DBLP/labels.pkl --output_label_path_style ../data/DBLP/labels_{k}_fold_cross_validation_{number}.pkl
conda deactivate