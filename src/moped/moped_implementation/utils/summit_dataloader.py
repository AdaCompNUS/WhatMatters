"""This module is used for computing social and map features for motion forecasting baselines.
Example usage:
    $ python compute_features.py --data_dir ~/val/data
        --feature_dir ~/val/features --mode val
"""

import os
import time
from typing import Any, List
from typing import Any, Dict, List, Optional, Tuple, Union

import argparse
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from utils.social_features_utils import SocialFeaturesUtils
from utils.map_features_utils import MapFeaturesUtils
from utils.baseline_config import RAW_DATA_FORMAT, _FEATURES_SMALL_SIZE
from utils.baseline_utils import get_relative_distance, get_normalized_traj


from utils.baseline_config import (
    BASELINE_INPUT_FEATURES,
    BASELINE_OUTPUT_FEATURES,
    FEATURE_FORMAT,
)

from tqdm import tqdm


class SummitBaselineDataset():
    def __init__(self, split: str, obs_len: int, pred_len: int,
                 raw_train_dir: str,
                 raw_val_dir: str,
                 raw_test_dir: str,
                 mode: str = None,
                 processed_dir: str = None):
        '''
        split: train/test/val: to process the correct directory

        mode: to process the map_features. In almost of the case, mode = split. However in KNN, for split='val',
        mode need to be 'test' so that the candidate_centerlines are poped up

        processed_dir is the place to store the process files. Because KNN needs different data for split='val',
        this processed_dir should be provided explicitely.
        '''
        self.mode = split if mode is None else mode

        if split == 'train':
            self._directory = raw_train_dir
        elif split == 'val':
            self._directory = raw_val_dir
        elif split == 'test':
            self._directory = raw_test_dir
        else:
            raise ValueError(split + ' is not valid')

        # if summit, path should be summit/train/data. if argoverse, path: argoverse/train/data/
        self.raw_dir = os.path.join(self._directory, "data")
        self.raw_file_names = os.listdir(self.raw_dir)

        self.processed_dir = os.path.join(self._directory, 'baseline_processed' if processed_dir is None else processed_dir)
        os.makedirs(self.processed_dir, exist_ok=True)

        # processed_file_names got from raw_file_names and not in processed_dir to ensure all files are processed
        self.processed_file_names = [os.path.splitext(f)[0] + '.pkl' for f in self.raw_file_names]
        self.processed_paths = [os.path.join(self.processed_dir, f) for f in self.processed_file_names]

        self.obs_len = obs_len
        self.pred_len = pred_len

        self.process()

    def compute_features_single_file(self, seq_path: str,
                            social_features_utils_instance: SocialFeaturesUtils,
                            map_features_utils_instance: MapFeaturesUtils,) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

        df = pd.read_csv(seq_path, dtype={"TIMESTAMP": str})
        
        maxwidth = df['TIMESTAMP'].astype(str).str.len().max()
        assert maxwidth == 2, "Otherwise, this dataset is not summit"
        df['TIMESTAMP'] = df['TIMESTAMP'].str.pad(side='left', fillchar='0', width=maxwidth)

        # padding timestamp
        maxwidth = df['TIMESTAMP'].astype(str).str.len().max()
        assert maxwidth == 2, "Otherwise, this dataset is not summit"
        df['TIMESTAMP'] = df['TIMESTAMP'].str.pad(side='left', fillchar='0', width=maxwidth)

        # Get social and map features for the agent
        agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values

        # Social features are computed using only the observed trajectory
        social_features = social_features_utils_instance.compute_social_features(
            df, agent_track, self.obs_len, self.obs_len + self.pred_len,
            RAW_DATA_FORMAT)

        # agent_track will be used to compute n-t distances for future trajectory,
        # using centerlines obtained from observed trajectory
        map_features, map_feature_helpers = map_features_utils_instance.compute_map_features(
            agent_track,
            self.obs_len,
            self.obs_len + self.pred_len,
            RAW_DATA_FORMAT,
            self.mode,
        )

        # Combine social and map features

        # If track is of OBS_LEN (i.e., if it's in test mode), use agent_track of full SEQ_LEN,
        # But keep (OBS_LEN+1) to (SEQ_LEN) indexes having None values
        if agent_track.shape[0] == self.obs_len:
            agent_track_seq = np.full(
                (self.obs_len + self.pred_len, agent_track.shape[1]), None)
            agent_track_seq[:self.obs_len] = agent_track
            merged_features = np.concatenate(
                (agent_track_seq, social_features, map_features), axis=1)
        else:
            merged_features = np.concatenate(
                (agent_track, social_features, map_features), axis=1)

        return merged_features, map_feature_helpers

    def process(self) -> None:
        """Load sequences, compute features, and save them.

        Args:
            start_idx : Starting index of the current batch
            sequences : Sequence file names
            save_dir: Directory where features for the current batch are to be saved
            map_features_utils_instance: MapFeaturesUtils instance
            social_features_utils_instance: SocialFeaturesUtils instance
        """

        # All files of processed_paths must exist, otherwise process again
        if files_exist(self.processed_paths):
            return

        social_features_utils_instance = SocialFeaturesUtils()
        map_features_utils_instance = MapFeaturesUtils()

        sequences = os.listdir(self.raw_dir)

        # Enumerate over the batch starting at start_idx
        for seq in tqdm(sequences):

            if not seq.endswith(".csv"):
                continue

            file_path = os.path.join(self.raw_dir, seq)
            seq_id = seq.split(".")[0]

            # Compute social and map features
            features, map_feature_helpers = self.compute_features_single_file(file_path,
                                    social_features_utils_instance,
                                     map_features_utils_instance)

            data = [[
                seq_id,
                features,
                map_feature_helpers["CANDIDATE_CENTERLINES"],
                map_feature_helpers["ORACLE_CENTERLINE"],
                map_feature_helpers["CANDIDATE_NT_DISTANCES"],
                ]
            ]
            data_df = pd.DataFrame(
                data,
                columns=[
                    "SEQUENCE",
                    "FEATURES",
                    "CANDIDATE_CENTERLINES",
                    "ORACLE_CENTERLINE",
                    "CANDIDATE_NT_DISTANCES",
                ],
            )

            save_path = os.path.join(self.processed_dir, seq_id+".pkl")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data_df.to_pickle(save_path)

    def merge_saved_features(self, temp_file_path: Union[str, None]) -> pd.DataFrame:
        """Merge features saved by parallel jobs.

        Args:
            batch_save_dir: Directory where features for all the batches are saved.

        """
        # path must be None or if string, must end with .pkl
        assert (temp_file_path == None) or (temp_file_path.endswith(".pkl"))
        all_features = []
        for feature_file in self.processed_paths:
            if not feature_file.endswith(".pkl") :
                continue
            file_path = feature_file
            df = pd.read_pickle(file_path)
            all_features.append(df)


        all_features_df = pd.concat(all_features, ignore_index=True)

        # Save the features for all the sequences into a single file
        if temp_file_path:
            all_features_df.to_pickle(temp_file_path)

        return all_features_df

    def __len__(self) -> int:
        return len(self.raw_file_names)

    def get(self, idx: int) -> str:
        '''

        :param idx:
        :return: return path to zipfile containing all csv files for this idx
        '''
        return self.processed_paths[idx]

def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([os.path.exists(f) for f in files])
