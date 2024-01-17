
"""This module evaluates the forecasted trajectories against the ground truth.
Author : Official Argoverse Forecasting Website
Website: https://github.com/jagjeet-singh/argoverse-forecasting
"""

import argparse
from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd
import pickle as pkl

from utils.eval_forecasting import compute_forecasting_metrics
from argoverse.map_representation.map_api import ArgoverseMap
from utils.baseline_config import FEATURE_FORMAT
from utils.baseline_utils import viz_predictions


def get_city_names_from_features(features_df: pd.DataFrame) -> Dict[int, str]:
    """Get sequence id to city name mapping from the features.

    Args:
        features_df: DataFrame containing the features
    Returns:
        city_names: Dict mapping sequence id to city name

    """
    city_names = {}
    for index, row in features_df.iterrows():
        city_names[row["SEQUENCE"]] = row["FEATURES"][0][
            FEATURE_FORMAT["CITY_NAME"]]
    return city_names


def get_pruned_guesses(
        forecasted_trajectories: Dict[int, List[np.ndarray]],
        city_names: Dict[int, str],prune_n_guesses: int) -> Dict[int, List[np.ndarray]]:
    """Prune the number of guesses using map.

    Args:
        forecasted_trajectories: Trajectories forecasted by the algorithm.
        city_names: Dict mapping sequence id to city name.
        gt_trajectories: Ground Truth trajectories.

    Returns:
        Pruned number of forecasted trajectories.

    """
    avm = ArgoverseMap()

    pruned_guesses = {}

    for seq_id, trajectories in forecasted_trajectories.items():

        city_name = city_names[seq_id]
        da_points = []
        for trajectory in trajectories:
            raster_layer = avm.get_raster_layer_points_boolean(
                trajectory, city_name, "driveable_area")
            da_points.append(np.sum(raster_layer))

        sorted_idx = np.argsort(da_points)[::-1]
        pruned_guesses[seq_id] = [
            trajectories[i] for i in sorted_idx[:prune_n_guesses]
        ]

    return pruned_guesses


def get_m_trajectories_along_n_cl(
        forecasted_trajectories: Dict[int, List[np.ndarray]], n_cl: int, max_neighbors_cl: int, n_guesses_cl: int
) -> Dict[int, List[np.ndarray]]:
    """Given forecasted trajectories, get <args.n_guesses_cl> trajectories along each of <args.n_cl> centerlines.

    Args:
        forecasted_trajectories: Trajectories forecasted by the algorithm.

    Returns:
        <args.n_guesses_cl> trajectories along each of <args.n_cl> centerlines.

    """
    selected_trajectories = {}
    for seq_id, trajectories in forecasted_trajectories.items():
        curr_selected_trajectories = []
        max_predictions_along_cl = min(len(forecasted_trajectories[seq_id]),
                                       n_cl * max_neighbors_cl)
        for i in range(0, max_predictions_along_cl, max_neighbors_cl):
            for j in range(i, i + n_guesses_cl):
                curr_selected_trajectories.append(
                    forecasted_trajectories[seq_id][j])
        selected_trajectories[seq_id] = curr_selected_trajectories
    return selected_trajectories


def viz_predictions_helper(
        forecasted_trajectories: Dict[int, List[np.ndarray]],
        gt_trajectories: Dict[int, np.ndarray],
        features_df: pd.DataFrame,
        obs_len: int,
        viz_seq_id: Union[None, List[int]],
) -> None:
    """Visualize predictions.

    Args:
        forecasted_trajectories: Trajectories forecasted by the algorithm.
        gt_trajectories: Ground Truth trajectories.
        features_df: DataFrame containing the features
        viz_seq_id: Sequence ids to be visualized

    """
    seq_ids = gt_trajectories.keys() if viz_seq_id is None else viz_seq_id
    for seq_id in seq_ids:
        gt_trajectory = gt_trajectories[seq_id]
        curr_features_df = features_df[features_df["SEQUENCE"] == seq_id]
        input_trajectory = (
            curr_features_df["FEATURES"].values[0]
            [:obs_len, [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]]].astype(
                "float"))
        output_trajectories = forecasted_trajectories[seq_id]
        candidate_centerlines = curr_features_df[
            "CANDIDATE_CENTERLINES"].values[0]
        city_name = curr_features_df["FEATURES"].values[0][
            0, FEATURE_FORMAT["CITY_NAME"]]

        gt_trajectory = np.expand_dims(gt_trajectory, 0)
        input_trajectory = np.expand_dims(input_trajectory, 0)
        output_trajectories = np.expand_dims(np.array(output_trajectories), 0)
        candidate_centerlines = np.expand_dims(np.array(candidate_centerlines),
                                               0)
        city_name = np.array([city_name])
        viz_predictions(
            input_trajectory,
            output_trajectories,
            gt_trajectory,
            candidate_centerlines,
            city_name,
            show=True,
        )


def evaluation(gt_trajectories: Dict[int, np.ndarray], forecasted_trajectories: Dict[int, List[np.ndarray]],
               features_df: pd.DataFrame, prune_n_guesses: int, n_cl: int, max_n_guesses: int,
               max_neighbors_cl:int, n_guesses_cl: int, horizon:int, miss_threshold: float, obs_len: int,
               viz:bool, viz_seq_id: Union[None, List[int]],
               forecasted_probabilities: Optional[Dict[int, List[float]]] = None):

    city_names = get_city_names_from_features(features_df)

    # Get displacement error and dac on multiple guesses along each centerline
    if not prune_n_guesses and n_cl:
        forecasted_trajectories = get_m_trajectories_along_n_cl(
            forecasted_trajectories, n_cl=n_cl, max_neighbors_cl=max_neighbors_cl, n_guesses_cl=n_guesses_cl)
        num_trajectories = n_cl * n_guesses_cl

    # Get displacement error and dac on pruned guesses
    elif prune_n_guesses:
        forecasted_trajectories = get_pruned_guesses(
            forecasted_trajectories, city_names, prune_n_guesses)
        num_trajectories = prune_n_guesses

    # Normal case
    else:
        num_trajectories = max_n_guesses

    metric_results = compute_forecasting_metrics(
        forecasted_trajectories,
        gt_trajectories,
        city_names,
        num_trajectories,
        horizon,
        miss_threshold,
        forecasted_probabilities
    )

    if viz:
        viz_predictions_helper(forecasted_trajectories, gt_trajectories,
                               features_df, obs_len, viz_seq_id)

    return metric_results