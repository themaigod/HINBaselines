import argparse
import os.path
import pickle
import random

import numpy as np


def load(labels):
    with open(labels, "rb") as f:
        label_original = pickle.load(f)
        f.close()
    return label_original


def cat(labels):
    train_idx = np.array(labels[0])[:, 0]
    train_target = np.array(labels[0])[:, 1]
    valid_idx = np.array(labels[1])[:, 0]
    valid_target = np.array(labels[1])[:, 1]

    idx = np.concatenate([train_idx, valid_idx])
    target = np.concatenate([train_target, valid_target])

    return idx, target


def percentage_length(length, percentage):
    length *= percentage
    val_len = int(length / 3)
    train_len = int(length - val_len)
    return train_len, val_len


def reorder(idx, target, length):
    index = list(range(length))
    random.shuffle(index)
    idx_new = idx[index]
    target_new = target[index]
    return idx_new, target_new


def split(idx_new, target_new, lengths):
    train_idx = idx_new[:lengths[0]]
    train_target = target_new[:lengths[0]]
    valid_idx = idx_new[lengths[0]: (lengths[0] + lengths[1])]
    valid_target = target_new[lengths[0]: (lengths[0] + lengths[1])]

    train_labels = np.vstack((train_idx, train_target)).transpose().tolist()
    valid_labels = np.vstack((valid_idx, valid_target)).transpose().tolist()

    return [train_labels, valid_labels]


def save(result, style, ori_name, percentage):
    current_dir = os.path.dirname(style.format(**{"percentage": percentage, 'ori_name': ori_name}))
    if not os.path.isdir(current_dir):
        os.makedirs(current_dir)
    with open(style.format(**{"percentage": percentage, 'ori_name': ori_name}), "wb") as f:
        pickle.dump(result, f)
        f.close()


def default_process(seed, label_path, percentage, output_label_path_style):
    random.seed(seed)
    label_ori_default = load(label_path)
    label_idx_default, label_target_default = cat(label_ori_default)
    label_train_len_default, label_val_len_default = percentage_length(len(label_idx_default), percentage)
    label_idx_new_default_default, label_target_new_default = reorder(label_idx_default, label_target_default,
                                                                      len(label_idx_default))
    split_result_default = split(label_idx_new_default_default, label_target_new_default,
                                 [label_train_len_default, label_val_len_default])
    split_result_default.append(label_ori_default[2])
    file_name_default = os.path.basename(label_path)
    save(split_result_default, output_label_path_style, file_name_default, percentage)


def default_dataset_process():
    seed = 0
    dataset_path = "../data"
    datasets = ["ACM", "DBLP", "IMDB"]
    k = 5
    ori_label_style = "labels_" + str(k) + "_fold_cross_validation_{i}.pkl"
    output_label_path_style = "{percentage}/{percentage}_{ori_name}"
    percentages = [0.1, 0.25, 0.5]

    for dataset in datasets:
        current_dataset_path = os.path.join(dataset_path, dataset)
        for i in range(k):
            current_ori_label = os.path.join(current_dataset_path, ori_label_style.format(**{"i": i}))
            current_output_label_path_style = os.path.join(current_dataset_path, output_label_path_style)
            for percentage in percentages:
                default_process(seed, current_ori_label, percentage, current_output_label_path_style)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--default", type=bool, default=True,
                        help="if True, it will process all the default datasets with 5 fold and default path, you do "
                             "not need to provide other args")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--label_path", type=str, default="../data/IMDB/labels_5_fold_cross_validation_0.pkl",
                        help="label path")
    parser.add_argument("--output_label_path_style", type=str,
                        default="../data/IMDB/{percentage}/{percentage}_{ori_name}",
                        help="it is output label path style, you need to give {k}, {number} in the style")
    parser.add_argument("--percentage", type=int, default=0.1, help="percentage for training set")
    args = parser.parse_args()
    if not args.default:
        random.seed(args.seed)

        label_ori = load(args.label_path)
        label_idx, label_target = cat(label_ori)
        label_train_len, label_val_len = percentage_length(len(label_idx), args.percentage)
        label_idx_new, label_target_new = reorder(label_idx, label_target, len(label_idx))
        split_result = split(label_idx_new, label_target_new, [label_train_len, label_val_len])
        split_result.append(label_ori[2])
        file_name = os.path.basename(args.label_path)
        save(split_result, args.output_label_path_style, file_name, args.percentage)

    else:
        default_dataset_process()
