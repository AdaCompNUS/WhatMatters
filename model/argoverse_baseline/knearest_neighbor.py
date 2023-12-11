import numpy as np
from typing import List, Tuple, Dict, Union, Any
from mmcv import Config
import time
import os

from utils.baseline_config import FEATURE_FORMAT
from utils import baseline_utils
from utils.nn_utils import Regressor
import pandas as pd
import pickle as pkl
from argoverse.evaluation.competition_util import generate_forecasting_h5

from utils.eval_forecasting_helper import evaluation
from model.base_model import MotionPrediction

class KNearestNeighbor(MotionPrediction):
    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        self.set_up_config()

    def set_up_config(self):
        self.train_features = self.config.data.train.features
        self.test_features = self.config.data.test.features
        self.val_features = self.config.data.val.features
        self.traj_normalize = self.config.model.normalize
        self.use_delta = self.config.model.use_delta
        self.use_map = self.config.model.use_map
        self.use_social = self.config.model.use_social
        self.obs_len = self.config.model.obs_len
        self.pred_len = self.config.model.pred_len
        self.n_neigh = self.config.model.n_neigh
        self.joblib_batch_size = self.config.data.joblib_batch_size
        self.output_dir = self.config.model.model_path

    def perform_k_nn_train(self,
            data_dict: Dict[str, Union[np.ndarray, pd.DataFrame, None]],
            baseline_key: str,
            args: Any) -> None:
        """Perform various experiments using K Nearest Neighbor Regressor.

        Args:
            data_dict (dict): Dictionary of train/val/test data
            baseline_key: Key for obtaining features for the baseline

        """

        # Get model object for the baseline
        model = Regressor()

        train_input = data_dict["train_input"]
        train_output = data_dict["train_output"]
        train_helpers = data_dict["train_helpers"]


        if args.use_map:
            print("####  Training Nearest Neighbor in NT frame  ###")
            model.train_map(
                train_input,
                train_output,
                len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]),
                args,
            )

        else:
            print("####  Training Nearest Neighbor in absolute map frame  ###")
            model.train_absolute(
                train_input,
                train_output,
                len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]),
                args,
            )

    def train(self):
        '''
        :return:
        '''
        print('Starting to traing KNearestNeighbor\n')
        # Create args to parse
        args = Config(dict(
            train_features=self.train_features,
            val_features=self.val_features, # Use both train + val for cross-validation
            test_features=None,
            use_map=self.use_map,
            use_social=self.use_social,
            normalize=self.traj_normalize,
            use_delta=self.use_delta,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            joblib_batch_size=self.joblib_batch_size,
            model_path=os.path.join(self.output_dir, 'model.pkl'),
            test=False
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
        self.perform_k_nn_train(data_dict, baseline_key, args)
        end = time.time()
        print(f"Completed experiment in {(end - start) / 60.0} mins")

    def test(self):
        print('Starting to traing KNearestNeighbor\n')
        # Create args to parse
        args = Config(dict(
            train_features=self.train_features,
            val_features=None,
            test_features=self.test_features,
            use_map=self.use_map,
            use_social=self.use_social,
            normalize=self.traj_normalize,
            use_delta=self.use_delta,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            n_neigh=self.n_neigh,
            joblib_batch_size=self.joblib_batch_size,
            model_path=os.path.join(self.output_dir, 'model.pkl'),
            traj_save_path=os.path.join(os.path.join(self.output_dir, 'competition_files'), 'nn_test_pred.pkl'),
            test=True
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

        model = Regressor()

        train_input = data_dict["train_input"]
        train_output = data_dict["train_output"]
        train_helpers = data_dict["train_helpers"]

        test_input = data_dict["test_input"]
        test_output = data_dict["test_output"]
        test_helpers = data_dict["test_helpers"]

        if args.use_map:
            print("####  Validating Nearest Neighbor in NT frame  ###")
            pred_trajectories = model.infer_map_toplevel(
                train_input,
                train_output,
                test_helpers,
                len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]),
                args,
            )

        else:
            print("####  Validating Nearest Neighbor in absolute map frame  ###")
            pred_trajectories = model.infer_absolute_toplevel(
                train_input,
                train_output,
                test_input,
                test_helpers,
                len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]),
                args,
            )

        forecasted_trajectories = {}
        for id, val in pred_trajectories.items():
            forecasted_trajectories[id] = [pred_trajectories[id][i, :, :]
                                           for i in range(pred_trajectories[id].shape[0])]

        # Probabilities
        forecasted_probabilities = {k: np.ones(shape=len(v)) / len(v) for k, v in forecasted_trajectories.items()}

        self.output_path = os.path.join(self.output_dir, "competition_files/")
        generate_forecasting_h5(forecasted_trajectories, self.output_path, filename="argoverse_forecasting",
                                probabilities=forecasted_probabilities)

        print(f'\nDone generating forecasting sequences for competitions.'
              f'There are {len(forecasted_trajectories)} keys in output. Each value of key is a list of 6 ndarray has shape'
              f' {forecasted_trajectories[id][0].shape}')

        end = time.time()
        print(f"Completed experiment in {(end - start) / 60.0} mins")

    def validate(self):
        print('Starting to traing KNearestNeighbor\n')
        # Create args to parse
        
        args = Config(dict(
            train_features=self.train_features,
            val_features=None if self.use_map else self.val_features,
            test_features=self.val_features if self.use_map else None,
            use_map=self.use_map,
            use_social=self.use_social,
            normalize=self.traj_normalize,
            use_delta=self.use_delta,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            n_neigh=self.n_neigh,
            joblib_batch_size=self.joblib_batch_size,
            model_path=os.path.join(self.output_dir, 'model.pkl'),
            traj_save_path=os.path.join(os.path.join(self.output_dir, 'validation_files'), 'nn_validation_pred.pkl'),
            test= True
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

        model = Regressor()

        train_input = data_dict["train_input"]
        train_output = data_dict["train_output"]
        train_helpers = data_dict["train_helpers"]
        forecasted_trajectories = {}

        if args.use_map:
            val_input = data_dict["test_input"]
            val_output = data_dict["test_output"]
            val_helpers = data_dict["test_helpers"]
            
            print("####  Validating Nearest Neighbor in NT frame  ###")
            pred_trajectories = model.infer_map_toplevel(
                train_input,
                train_output,
                val_helpers,
                len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]),
                args,
            )
            for id, val in pred_trajectories.items():
                forecasted_trajectories[id] = pred_trajectories[id]
        else:
            val_input = data_dict["val_input"]
            val_output = data_dict["val_output"]
            val_helpers = data_dict["val_helpers"]

            print("####  Validating Nearest Neighbor in absolute map frame  ###")
            pred_trajectories = model.infer_absolute_toplevel(
                train_input,
                train_output,
                val_input,
                val_helpers,
                len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]),
                args,
            )

            for id, val in pred_trajectories.items():
                forecasted_trajectories[id] = [pred_trajectories[id][i, :, :] for i in range(pred_trajectories[id].shape[0])]
           

        # To get ground-truth
        data_features_frame = pd.read_pickle(self.val_features)
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

        if args.use_map:
            self.config.evaluation.val_eval.prune_n_guesses = 6

        self.evaluation_trajectories(groundtruth_trajectories=groundtruth_trajectories,
                                    forecasted_trajectories=forecasted_trajectories,
                                    features_df=data_features_frame, args=self.config,
                                    output_path=self.output_dir,
                                    forecasted_probabilities=forecasted_probabilities)

        end = time.time()
        print(f"Completed experiment in {(end - start) / 60.0} mins")
