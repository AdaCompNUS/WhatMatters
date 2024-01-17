#!/home/cunjun/anaconda3/envs/conda38/bin/python
# A shebang line to run different python version

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Any
import numpy as np
import pandas as pd
import os
from utils.eval_forecasting_helper import evaluation
from utils.baseline_config import FEATURE_FORMAT

class MotionPrediction(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    def evaluation_trajectories(self, groundtruth_trajectories: Dict[int, np.ndarray],
                                forecasted_trajectories: Dict[int, List[np.ndarray]],
                                features_df: pd.DataFrame, args:Any, output_path: str,
                                forecasted_probabilities: Optional[Dict[int, List[float]]]= None):

        prune_n_guesses = args.evaluation.val_eval.prune_n_guesses
        n_cl = args.evaluation.val_eval.n_cl
        max_neighbors_cl = args.evaluation.val_eval.max_neighbors_cl
        n_guesses_cl = args.evaluation.val_eval.n_guesses_cl
        miss_threshold = args.evaluation.val_eval.miss_threshold
        horizon = args.model.pred_len
        obs_len = args.model.obs_len

        text_file = open(os.path.join(output_path, "metric.txt"), "w")

        metrics = {}
        for max_n_guesses in [1, 3, 6]:
            metric = evaluation(gt_trajectories=groundtruth_trajectories,
                                forecasted_trajectories=forecasted_trajectories,
                                features_df=features_df,
                                prune_n_guesses=prune_n_guesses,
                                n_cl=n_cl, max_n_guesses=max_n_guesses,
                                max_neighbors_cl=max_neighbors_cl, n_guesses_cl=n_guesses_cl,
                                horizon=horizon, obs_len=obs_len,
                                miss_threshold=miss_threshold,
                                forecasted_probabilities=forecasted_probabilities,
                                viz=False, viz_seq_id=None)

            metrics[max_n_guesses] = metric

            text_file.write("------------------------------------------------\n")
            text_file.write(f"Prediction Horizon : {horizon}, Max #guesses (K): {max_n_guesses}\n")
            text_file.write("------------------------------------------------\n")
            text_file.write(str(metric) + "\n")
            text_file.write("------------------------------------------------\n")

        text_file.close()
        return metrics

    def output_competition_files(self):
        pass

    def get_gt_trajectories(self, feature_files_path: str, obs_len: int, pred_len: int) -> Dict[int, np.ndarray]:


        data_features_frame = pd.read_pickle(feature_files_path)
        feature_idx = [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]]
        seq_id = data_features_frame["SEQUENCE"].values

        gt_trajectory = np.stack(
            data_features_frame["FEATURES"].values)[:, obs_len:obs_len + pred_len,
                        feature_idx].astype("float")

        groundtruth_trajectories = {}
        for i in range(gt_trajectory.shape[0]):
            groundtruth_trajectories[seq_id[i]] = gt_trajectory[i]  # No list for ground-truth

        return groundtruth_trajectories

