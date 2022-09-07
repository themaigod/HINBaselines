import argparse
import csv
import collections

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, default="../pygcn/gcnfix.csv", help='csv file path')
parser.add_argument('--stable', type=list, default=['0'], help='which axis is needed to stable for sort')
parser.add_argument('--ignore', type=list, default=['1', '2'],
                    help='(optional) which axis is ignored to stable for sort')
parser.add_argument('--statistic', type=list, default=['3', '4'], help='which axis is needed to calculate for sort')
parser.add_argument('--std', type=bool, default=True, help='calculate the std value')
parser.add_argument('--mean', type=bool, default=True, help='calculate the mean value')
parser.add_argument('--output', type=str, default="../pygcn/gcnresult.csv", help='output csv file')
parser.add_argument('--start', type=int, default=None, help='the start line (start with 0)')
parser.add_argument('--end', type=int, default=None, help='the end line (not include)')
parser.add_argument('--reformat', type=bool, default=True, help='reformat the result')
args = parser.parse_args()
print(args)

with open(args.csv, newline='') as csvfile:
    reader = csv.reader(csvfile)
    csv_file = np.array(list(reader))

if args.start and args.end:
    csv_file = csv_file[args.start: args.end]
elif args.start and not args.end:
    csv_file = csv_file[args.start:]
elif args.end and not args.start:
    csv_file = csv_file[:args.end]

args.stable = [int(i) for i in args.stable]
args.ignore = [int(i) for i in args.ignore]
args.statistic = [int(i) for i in args.statistic]

Reorder = collections.defaultdict(list)
for i in csv_file:
    Reorder[tuple(i[args.stable])].append([float(j) for j in i[args.statistic]])

std = {}
if args.std:
    for key, value in Reorder.items():
        value = np.array(value)
        std[key] = np.std(value, axis=0, ddof=1).tolist()

mean = {}
if args.std:
    for key, value in Reorder.items():
        value = np.array(value)
        mean[key] = np.mean(value, axis=0).tolist()

with open(args.output, 'w', newline='') as f:
    writer = csv.writer(f)
    for key in Reorder.keys():
        if not args.reformat:
            writer.writerow(list(key) + mean[key] + std[key])
        else:
            reformat_list = [mean[key][0], std[key][0], mean[key][1], std[key][1]]
            reformat_list = [round(i * 100, 3) for i in reformat_list]
            writer.writerow(list(key)+ reformat_list + mean[key] + std[key])
