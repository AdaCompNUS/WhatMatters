import numba.core.types
import numpy as np
from typing import Tuple, Dict
from utils.baseline_config import FEATURE_FORMAT
import pandas as pd
from model.base_model import MotionPrediction
from utils.logger import Logger
from numba import jit, types, njit
from numba.experimental import jitclass

@jit(nopython=True)
def _get_mean_velocity(coords: np.ndarray, avg_history_points: int) -> Tuple[np.ndarray, np.ndarray]:
    vx, vy = (
        np.zeros((coords.shape[0], avg_history_points-1)),
        np.zeros((coords.shape[0], avg_history_points-1)),
    )

    for i in range(1, avg_history_points , 1):
        vx[:, i - 1] = (coords[:, -i, 0] - coords[:, -(i + 1), 0]) / 0.1
        vy[:, i - 1] = (coords[:, -i, 1] - coords[:, -(i + 1), 1]) / 0.1

    mean_x, mean_y = (
        np.zeros(coords.shape[0]),
        np.zeros(coords.shape[0])
    )
    for i in range(vx.shape[0]):
        mean_x[i] = vx[i].mean()
        mean_y[i] = vy[i].mean()


    return mean_x, mean_y


@jit(nopython=True)
def _predict(obs_trajectory: np.ndarray, vx: np.ndarray, vy: np.ndarray, pred_len: int) -> np.ndarray:
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
        pred_trajectory[:, i, 0] = prev_coords[:, 0] + vx * 0.1
        pred_trajectory[:, i, 1] = prev_coords[:, 1] + vy * 0.1
        prev_coords = pred_trajectory[:, i]

    return pred_trajectory

@jit(nopython=True)
def _get_multi_predictions(trajectory: np.ndarray, pred_len: int, avg_history_points: int):
    '''

    :param trajectory: history trajectories, shape should be [num_trajectories, obs_len, 2], where 2 means x, y position
    :param num_points: number of future points that are going to predict.
    :param avg_history_points:  history observation
    :return: trajectories that has length [num_trajectories, pred_len, 2]
    '''
    vx, vy = _get_mean_velocity(trajectory, avg_history_points=avg_history_points)
    pred_traj = _predict(trajectory, vx=vx, vy=vy, pred_len=pred_len)

    return pred_traj


@jitclass([])
class ConstantVelocityPlannerJIT():
    def __init__(self):
        return

    def predict(self, trajectory: np.ndarray, pred_len: int, avg_history_points: int):
        '''

        :param trajectory: history trajectories, shape should be [num_trajectories, obs_len, 2], where 2 means x, y position
        :param num_points: number of future points that are going to predict.
        :param avg_history_points:  history observation
        :return: trajectories that has length [num_trajectories, pred_len, 2]
        '''
        return _get_multi_predictions(trajectory, pred_len, avg_history_points)
