import os
import fnmatch
import argparse
import numpy as np
import math
import random
import pandas as pd

TRAIN_FOLDER = "/home/cunjun/moped_data/summit/train/data/"
VAL_FOLDER = "/home/cunjun/moped_data/summit/val/data/"

cap = 10

train_txt_files = list([])
for root, dirnames, filenames in os.walk(TRAIN_FOLDER):

    for filename in fnmatch.filter(filenames, '*.csv'):
        # append the absolute path for the file
        train_txt_files.append(os.path.join(root, filename))
print("%d files found in %s" % (len(train_txt_files), TRAIN_FOLDER))

val_txt_files = list([])
for root, dirnames, filenames in os.walk(VAL_FOLDER):

    for filename in fnmatch.filter(filenames, '*.csv'):
        # append the absolute path for the file
        val_txt_files.append(os.path.join(root, filename))
print("%d files found in %s" % (len(val_txt_files), VAL_FOLDER))

#1. Must be unique in train_txt_files
assert len(train_txt_files) == len(set(train_txt_files))
print(f"Passing condition 1")

assert len(val_txt_files) == len(set(val_txt_files))
print(f"Passing condition 2")

for f in val_txt_files:
    if f in train_txt_files:
        print(f"File {f} is same in both")
        break


print(f"Passing condition 3")

