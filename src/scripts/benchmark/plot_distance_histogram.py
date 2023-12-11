import os
import fnmatch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#TRAIN_FOLDER_SUMMIT = "/home/cunjun/driving_data/result/summit_process/train/" #"/home/cunjun/moped_data/summit/train/data/"
TRAIN_FOLDER_SUMMIT = "/home/cunjun/moped_data/summit/train/data/"

TRAIN_FOLDER_ARGOVERSE = "/home/cunjun/moped_data/argoverse/train/data"

cap = 10

summit_len = []
summit_agent_len = []

limit = 0

for root, dirnames, filenames in os.walk(TRAIN_FOLDER_SUMMIT):

    for filename in fnmatch.filter(filenames, '*.csv'):
        limit += 1
        if limit >= 100:
            break

        xxx = 0

        fpath = os.path.join(root, filename)
        data = pd.read_csv(fpath)
        av_trajectory = data[data['OBJECT_TYPE'] == 'AGENT'][['X', 'Y']].values
        other_trajectory = data[data['OBJECT_TYPE'] != 'AGENT']

        track_id = pd.unique(other_trajectory['TRACK_ID'])
        other_trajectory_list = []
        for track in track_id:
            temp = other_trajectory.groupby(['TRACK_ID']).get_group(track)[['X', 'Y']].values
            if len(temp) >= 20:
                summit_len.append(float(np.mean(np.abs(av_trajectory[19] - temp[19]))))
                xxx += 1
        summit_agent_len.append(xxx)

        print(f'summit {limit}')


limit = 0


argoverse_len = []
argoverse_agent_len = []

for root, dirnames, filenames in os.walk(TRAIN_FOLDER_ARGOVERSE):

    for filename in fnmatch.filter(filenames, '*.csv'):
        limit += 1
        if limit >= 100:
            break

        xxx = 0

        fpath = os.path.join(root, filename)
        data = pd.read_csv(fpath)
        av_trajectory = data[data['OBJECT_TYPE'] == 'AGENT'][['X', 'Y']].values
        other_trajectory = data[data['OBJECT_TYPE'] != 'AGENT']

        track_id = pd.unique(other_trajectory['TRACK_ID'])
        other_trajectory_list = []
        for track in track_id:
            temp = other_trajectory.groupby(['TRACK_ID']).get_group(track)[['X', 'Y']].values
            if len(temp) >= 20:
                argoverse_len.append(float(np.mean(np.abs(av_trajectory[19] - temp[19]))))
                xxx += 1

        argoverse_agent_len.append(xxx)

        print(f'argoverse {limit}')


print(np.mean(summit_len))
print(np.mean(argoverse_len))


print(np.mean(summit_agent_len))
print(np.mean(argoverse_agent_len))

fig, axs = plt.subplots(2, 2, figsize=(20,10))

axs[0][0].hist(summit_len)
axs[0][0].title.set_text("Histogram of distance between AGENT and others in SUMMIT")
axs[0][0].set_xlim(0, 150)

axs[0][1].hist(summit_agent_len)
axs[0][1].title.set_text("Histogram of number of other agents near to AGENT in SUMMIT")
axs[0][1].set_xlim(0, 60)

axs[1][0].hist(argoverse_len, label="Mean of number of others is 16")
axs[1][0].title.set_text("Histogram of distance between agent and others in Argoverse")
axs[1][0].set_xlim(0, 150)


axs[1][1].hist(argoverse_agent_len)
axs[1][1].title.set_text("Histogram of number of other agents near to AGENT in Argoverse")
axs[1][1].set_xlim(0, 60)

plt.legend()
plt.show()
