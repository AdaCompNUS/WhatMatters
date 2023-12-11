
import numpy as np
from typing import Tuple, Dict
from argoverse.evaluation.competition_util import generate_forecasting_h5
import os
import atexit
import tempfile

from utils.baseline_config import FEATURE_FORMAT
import pandas as pd

from model.base_model import MotionPrediction
from utils.logger import Logger


def _get_mean_velocity_and_acceleration(coords: np.ndarray, avg_history_points: int) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get mean velocity of the observed trajectory.

    Args:
        coords: Coordinates for the trajectory, shape [num_trajectories, obs_len, features]
    Returns:
        Mean velocity along x and y

    """
    vx, vy = (
        np.zeros((coords.shape[0], avg_history_points-1)),
        np.zeros((coords.shape[0], avg_history_points-1)),
    )

    ax, ay = (
        np.zeros((coords.shape[0], avg_history_points-2)),
        np.zeros((coords.shape[0], avg_history_points-2)),
    )

    for i in range(1, avg_history_points, 1):
        vx[:, i - 1] = (coords[:, -i, 0] - coords[:, -(i + 1), 0]) / 0.1
        vy[:, i - 1] = (coords[:, -i, 1] - coords[:, -(i + 1), 1]) / 0.1

    for i in range(1, avg_history_points-1, 1):
        ax[:, i - 1] = (vx[:, -i] - vx[:, -(i + 1)]) / 0.1
        ay[:, i - 1] = (vy[:, -i] - vy[:, -(i + 1)]) / 0.1

    vx = np.mean(vx, axis=1)
    vy = np.mean(vy, axis=1)
    ax = np.mean(ax, axis=1)
    ay = np.mean(ay, axis=1)

    return vx, vy, ax, ay

def _predict(obs_trajectory: np.ndarray, vx: np.ndarray, vy: np.ndarray,
             ax: np.ndarray, ay: np.ndarray, pred_len: int) -> np.ndarray:
    """Predict future trajectory given mean velocity.

    Args:
        obs_trajectory: Observed Trajectory, num_trajectories, obs_len, 2]
        vx: Mean velocity along x
        vy: Mean velocity along y
        args: Arguments to the baseline
    Returns:
        pred_trajectory: Future trajectory

    """
    pred_trajectory = np.zeros((obs_trajectory.shape[0], pred_len, 2))

    prev_coords = obs_trajectory[:, -1, :]
    for i in range(pred_len):
        pred_trajectory[:, i, 0] = prev_coords[:, 0] + vx * 0.1 + 1/2 * ax * 0.1**2
        pred_trajectory[:, i, 1] = prev_coords[:, 1] + vy * 0.1 + 1/2 * ay * 0.1**2
        prev_coords = pred_trajectory[:, i]

    return pred_trajectory

def _get_multi_predictions(trajectory: np.ndarray, pred_len: int, avg_point_list: Tuple):
    '''

    :param trajectory: history trajectories, shape should be [num_trajectories, obs_len, 2], where 2 means x, y position
    :param num_points: number of future points that are going to predicted.
    :param avg_point_list: List of points, where each point is number of history trajectories to predict a point
                            in future
    :return: trajectories that has length [len(avg_point_list), num_trajectories, n + num_points, 2]
    '''
    predict_traj_list = []
    for i in avg_point_list:
        vx, vy, ax, ay = _get_mean_velocity_and_acceleration(trajectory, avg_history_points=i)
        pred_traj = _predict(trajectory, vx=vx, vy=vy, ax=ax, ay=ay, pred_len=pred_len)
        predict_traj_list.append(pred_traj)
    return np.array(predict_traj_list)

class ConstantAcceleration(MotionPrediction):
    def __init__(self, config: Dict, logger: Logger):
        self.config = config
        self.logger = logger

        #1. Set up configuration

        self.test_features_path = config.data.test.features
        self.val_features_path = config.data.val.features

        self.obs_len = config.model.obs_len
        self.pred_len = config.model.pred_len
        self.avg_points = config.model.avg_points


    def get_mean_velocity_and_acceleration(self, coords: np.ndarray, avg_history_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get mean velocity of the observed trajectory.

        Args:
            coords: Coordinates for the trajectory, shape [num_trajectories, obs_len, features]
        Returns:
            Mean velocity along x and y

        """
        return _get_mean_velocity_and_acceleration(coords, avg_history_points)

    def predict(self, obs_trajectory: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                ax: np.ndarray, ay: np.ndarray, pred_len: int) -> np.ndarray:
        """Predict future trajectory given mean velocity.

        Args:
            obs_trajectory: Observed Trajectory, num_trajectories, obs_len, 2]
            vx: Mean velocity along x
            vy: Mean velocity along y
            args: Arguments to the baseline
        Returns:
            pred_trajectory: Future trajectory

        """
        return _predict(obs_trajectory, vx, vy, ax, ay, pred_len)

    def get_multi_predictions(self, trajectory: np.ndarray, pred_len: int, avg_point_list: Tuple):
        '''

        :param trajectory: history trajectories, shape should be [num_trajectories, obs_len, 2], where 2 means x, y position
        :param num_points: number of future points that are going to predicted.
        :param avg_point_list: List of points, where each point is number of history trajectories to predict a point
                                in future
        :return: trajectories that has length [len(avg_point_list), num_trajectories, n + num_points, 2]
        '''
        return _get_multi_predictions(trajectory, pred_len, avg_point_list)

    def train(self):
        '''
        Constant Velocity does not need training
        :return:
        '''
        print('Constant Acceleration does not need training. Done\n')

    def test(self):
        if self.test_features_path == "" or self.test_features_path == None:
            print("This configuration does not have test features path, thus stopping here")
            return

        data_features_frame = pd.read_pickle(self.test_features_path)
        print(f'Loading test data sets with number of sequences: {len(data_features_frame)}')

        feature_idx = [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]]
        seq_id = data_features_frame["SEQUENCE"].values

        obs_trajectory = np.stack(
            data_features_frame["FEATURES"].values)[:, :self.obs_len, feature_idx].astype("float")

        # Output has shape [K, num_trajectories, pred_len, 2]
        pred_trajectories = self.get_multi_predictions(obs_trajectory, pred_len=self.pred_len,
                                                  avg_point_list=self.avg_points)

        # Dict[int, List[np.ndarray]] is shape of forecasted_trajectories
        forecasted_trajectories = {}
        for i in range(pred_trajectories.shape[1]):
            forecasted_trajectories[seq_id[i]] = pred_trajectories[:, i,: ,:]

        forecasted_probabilities = {k: np.ones(shape=len(v)) / len(v) for k, v in forecasted_trajectories.items()}

        generate_forecasting_h5(forecasted_trajectories, self.logger.get_test_log_dir(), filename="argoverse_forecasting",
                                probabilities=forecasted_probabilities)

        self.logger.info(f'\nDone generating forecasting sequences for competitions.'
              f'There are {len(forecasted_trajectories)} keys in output. Each value of key is ndarray has shape'
              f' {forecasted_trajectories[seq_id[0]].shape}')

    def validate(self):
        if self.val_features_path == "" or self.val_features_path == None:
            print("This configuration does not have test features path, thus stopping here")
            return

        data_features_frame = pd.read_pickle(self.val_features_path)
        self.logger.info(f'Loading validation data sets with number of sequences: {len(data_features_frame)}')

        feature_idx = [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]]
        seq_id = data_features_frame["SEQUENCE"].values

        obs_trajectory = np.stack(
            data_features_frame["FEATURES"].values)[:, :self.obs_len, feature_idx].astype("float")

        # Output has shape [K, num_trajectories, pred_len, 2]
        pred_trajectories = self.get_multi_predictions(obs_trajectory, pred_len=self.pred_len,
                                                       avg_point_list=self.avg_points)

        # Dict[int, List[np.ndarray]] is shape of forecasted_trajectories
        forecasted_trajectories = {}
        for i in range(pred_trajectories.shape[1]):
            forecasted_trajectories[seq_id[i]] = [pred_trajectories[z, i, :, :]
                                                  for z in range(pred_trajectories[:, i, :, :].shape[0])]

        # Probabilities of each trajectories
        forecasted_probabilities = {k: np.ones(shape=len(v)) / len(v) for k, v in forecasted_trajectories.items()}
        self.logger.dump_forecasted_trajectories(forecasted_trajectories=forecasted_trajectories,
                                                 forecasted_probs=forecasted_probabilities)
        self.logger.info(f'Done generating forecasting sequences for validation.')

        # Dict[int, np.ndarray] is shape of groundtruth_trajectories
        groundtruth_trajectories = self.get_gt_trajectories(self.val_features_path,
                                                            obs_len=self.obs_len, pred_len=self.pred_len)
        self.logger.info(f'Done generating ground-truth sequences for validation')

        self.evaluation_trajectories(groundtruth_trajectories=groundtruth_trajectories,
                                     forecasted_trajectories=forecasted_trajectories,
                                     features_df=data_features_frame, args=self.config,
                                     output_path=self.logger.get_val_log_dir(), forecasted_probabilities=forecasted_probabilities)


class ConstantAccelerationPlanner():
    def __int__(self):
        pass

    @staticmethod
    def predict(trajectory: np.ndarray, pred_len: int, avg_point_list: Tuple):
        '''

        :param trajectory: history trajectories, shape should be [num_trajectories, obs_len, 2], where 2 means x, y position
        :param num_points: number of future points that are going to predicted.
        :param avg_point_list: List of points, where each point is number of history trajectories to predict a point
                                in future
        :return: trajectories that has length [len(avg_point_list), num_trajectories, n + num_points, 2]
        '''
        return _get_multi_predictions(trajectory, pred_len, avg_point_list)

if __name__ == "__main__":
    trajectories = np.random.randn(2, 10,2)
    temp = ConstantAccelerationPlanner.predict(trajectories, 3, (5,))
    print(temp)
    print(temp.shape)
