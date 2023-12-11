import numpy as np
from typing import Dict
from mmcv import Config
import time
from utils.baseline_config import FEATURE_FORMAT
from utils.nn_utils import MopedRegressor
import pandas as pd
from utils.logger import Logger
import logging
import pickle as pkl
import utils.baseline_utils as baseline_utils
import logging

from model.base_model import MotionPrediction

class SummitKNearestNeighbor(MotionPrediction):
    def __init__(self, config: Dict, logger: Logger):
        self.config = config
        self.logger = logger

        # 1. Set up configuration

        self.train_features_path = self.config.data.train.features
        self.val_features_path = self.config.data.val.features
        
        self.traj_normalize = self.config.model.normalize
        self.use_delta = self.config.model.use_delta
        self.use_map = self.config.model.use_map
        self.use_social = self.config.model.use_social
        self.obs_len = self.config.model.obs_len
        self.pred_len = self.config.model.pred_len
        self.n_neigh = self.config.model.n_neigh
        self.checkpoint = self.config.model.checkpoint
        

    def train(self):
        '''
        :return:
        '''
        print('Starting to training KNearestNeighbor\n')


        # Create args to parse
        args = Config(dict(
            use_map=self.use_map,
            use_social=self.use_social,
            normalize=self.traj_normalize,
            use_delta=self.use_delta,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            train_features=self.train_features_path,
            test_features=None,
            val_features=None
        ))
        
        if not baseline_utils.validate_args(args):
            return

        np.random.seed(100)

        # Get features
        if args.use_map and args.use_social:
            baseline_key = "map_social"
        elif args.use_map:
            baseline_key = "map"
        elif args.use_social:
            baseline_key = "social"
        else:
            baseline_key = "none"

        # Get data
        data_dict = baseline_utils.get_data(args, baseline_key)

        # Perform experiments
        start = time.time()
        model = MopedRegressor()
        train_input = data_dict["train_input"]
        train_output = data_dict["train_output"]
        train_helpers = data_dict["train_helpers"]

        if args.use_map:
            print("####  Training Nearest Neighbor in NT frame  ###")
            grid_search = model.train_map(
                train_input,
                train_output,
                len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]),
                args,
            )

        else:
            print("####  Training Nearest Neighbor in absolute map frame  ###")
            grid_search = model.train_absolute(
                train_input,
                train_output,
                len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]),
                args,
            )

        self.logger.dump_pkl_model(grid_search, model_path=self.config.model.train_cfg.save_model_path)

        end = time.time()
        print(f"Completed experiment in {(end - start) / 60.0} mins")

    def test(self):
        pass

    def validate(self):
        print('Starting to validate KNearestNeighbor\n')

        # Create args to parse
        args = Config(dict(
            use_map=self.use_map,
            use_social=self.use_social,
            normalize=self.traj_normalize,
            use_delta=self.use_delta,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            n_neigh=self.n_neigh,
            train_features=self.train_features_path,
            test_features=self.val_features_path, # Swap val for test
            val_features=None
        ))
        if not baseline_utils.validate_args(args):
            return

        np.random.seed(100)

        # Get features
        if args.use_map and args.use_social:
            baseline_key = "map_social"
        elif args.use_map:
            baseline_key = "map"
        elif args.use_social:
            baseline_key = "social"
        else:
            baseline_key = "none"
        # # Get data
        data_dict = baseline_utils.get_data(args, baseline_key)

        # # Perform experiments
        start = time.time()
        with open(self.checkpoint, "rb") as f:
            grid_search = pkl.load(f)
        print(f"## Loaded {self.checkpoint} ....")
        
        val_input = data_dict["test_input"]
        val_output = data_dict["test_output"]
        val_helpers = data_dict["test_helpers"]
        train_output = data_dict["train_output"]


        if args.use_map:
            print("####  Validating Nearest Neighbor in NT frame  ###")
            pred_trajectories = MopedRegressor().val_map(
                grid_search,
                train_output,
                val_helpers,
                len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]),
                args,
            )

        else:
            print("####  Validating Nearest Neighbor in absolute map frame  ###")
            pred_trajectories = MopedRegressor().val_absolute(
                grid_search,
                train_output=train_output,
                test_input=val_input,
                test_helpers=val_helpers,
                num_features=len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]),
                args=args,
            )

        forecasted_trajectories = {}
        for id, val in pred_trajectories.items():
            forecasted_trajectories[id] = [pred_trajectories[id][i]
                                                  for i in range(len(pred_trajectories[id])) if i < 6]

        # # To get ground-truth
        val_df = pd.read_pickle(self.val_features_path)
        data_features_frame = val_df
        feature_idx = [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]]
        seq_id = data_features_frame["SEQUENCE"].values

        gt_trajectory = np.stack(
            data_features_frame["FEATURES"].values)[:, self.obs_len:self.obs_len + self.pred_len,
                        feature_idx].astype("float")

        groundtruth_trajectories = {}
        for i in range(gt_trajectory.shape[0]):
            groundtruth_trajectories[seq_id[i]] = gt_trajectory[i]  # No list for ground-truth

        # Probabilities of each trajectories
        forecasted_probabilities = {k: np.ones(shape=len(v)) / len(v) for k, v in forecasted_trajectories.items()}

        self.evaluation_trajectories(groundtruth_trajectories=groundtruth_trajectories,
                                     forecasted_trajectories=forecasted_trajectories,
                                     features_df=data_features_frame, args=self.config,
                                     output_path=self.logger.get_val_log_dir(),
                                     forecasted_probabilities=forecasted_probabilities)

        end = time.time()
        print(f"Completed experiment in {(end - start) / 60.0} mins")



