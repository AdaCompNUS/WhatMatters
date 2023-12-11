import os
import fnmatch
import pandas as pd
import numpy as np

TRAIN_FOLDER_SUMMIT = "/home/cunjun/moped_data/summit/train/data/"
TRAIN_FOLDER_ARGOVERSE = "/home/cunjun/moped_data/argoverse/train/data"

cap = 10

summit_len = []

for root, dirnames, filenames in os.walk(TRAIN_FOLDER_SUMMIT):
    for filename in fnmatch.filter(filenames, '*.csv'):
        fpath = os.path.join(root, filename)
        data = pd.read_csv(fpath)
        track_id = pd.unique(data['TRACK_ID'])
        summit_len.append(len(track_id))


argoverse_len = []

for root, dirnames, filenames in os.walk(TRAIN_FOLDER_ARGOVERSE):
    for filename in fnmatch.filter(filenames, '*.csv'):
        fpath = os.path.join(root, filename)
        data = pd.read_csv(fpath)
        track_id = pd.unique(data['TRACK_ID'])
        argoverse_len.append(len(track_id))

print(np.mean(summit_len))
print(np.mean(argoverse_len))
