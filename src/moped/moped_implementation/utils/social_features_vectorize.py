"""This module is used for computing social features for motion forecasting baselines."""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

from utils.baseline_config import (
    PADDING_TYPE,
    STATIONARY_THRESHOLD,
    VELOCITY_THRESHOLD,
    EXIST_THRESHOLD,
    DEFAULT_MIN_DIST_FRONT_AND_BACK,
    NEARBY_DISTANCE_THRESHOLD,
    FRONT_OR_BACK_OFFSET_THRESHOLD,
)


class SocialFeaturesUtilsVectorize:
    """Utils class for computation of social features."""
    def __init__(self):
        """Initialize class."""
        self.PADDING_TYPE = PADDING_TYPE
        self.STATIONARY_THRESHOLD = STATIONARY_THRESHOLD
        self.VELOCITY_THRESHOLD = VELOCITY_THRESHOLD
        self.EXIST_THRESHOLD = EXIST_THRESHOLD
        self.DEFAULT_MIN_DIST_FRONT_AND_BACK = DEFAULT_MIN_DIST_FRONT_AND_BACK
        self.NEARBY_DISTANCE_THRESHOLD = NEARBY_DISTANCE_THRESHOLD

    def get_is_front_or_back_vec(self,
            track: np.ndarray,
            neigh_x: np.ndarray,
            neigh_y: np.ndarray,
            raw_data_format: Dict[str, int],
    ):
        """

        Args:
            track: [20, num_features]
            neigh_x: [num_neigh, 20]
            neigh_y: [num_neigh, 20]

        Returns:
            [num_neigh] array of front/back

        """

        obs_len = 20 # Always 20 as this function accepts track whose length is equal to obs_len = 20
        track = track[:, [raw_data_format["X"], raw_data_format["Y"]]].astype(float)


        # Step 1. Find the different between the consecutive coordinates (x,y) in the track
        diff_xy = np.diff(track, axis=0)  # shape [19,2]
        # Step 2. Find the first non-zero different in either x or y (as long as either x or y is non-zero then it is valid)
        diff_xy_index = np.any(diff_xy != 0, axis=1)  # shape [19]
        # Step 4, building a lower triangular matrix for each
        lower_triangular = np.tril(np.ones((obs_len - 1, obs_len - 1), dtype=np.int8))
        # Step 5. Broadcasting index to lower_triangular
        # At any row that is not having 1' index, we return None for that row
        # otherwise, the last 1 at each row in the index
        index_array = np.einsum('ij,j->ij', lower_triangular, diff_xy_index)  # shape (19,19)
        # Add top row same as first row so two first row now are represented as at obs = 0 and obs = 1
        index_array = np.concatenate([index_array[0][None], index_array], axis=0)  # shape [20, 19]

        # Create a masked_index so that every row contains at least 1's so that we can calculate p1, p2, p3
        index_array_fake = np.copy(index_array)
        index_array_fake[:, 0] = 1

        # 2d array list where each row is i,j of 1's in index_array_fake
        ## TODO, now I can only to make it loop as i do not know how to use np
        ones_ij = np.argwhere(index_array_fake == 1)
        # out = []
        # for row in range(ones_ij.shape[0]):
        #     if len(out) == 0:
        #         out.append(ones_ij[row])
        #     elif out[-1][0] == ones_ij[row][0]: # same row with latest column, use this
        #         out.pop()
        #         out.append(ones_ij[row])
        #     elif out[-1][0] != ones_ij[row][0]:
        #         out.append(ones_ij[row])
        # out = np.stack(out, axis=0)

        temp = ones_ij[np.lexsort((-ones_ij[:, 1], ones_ij[:, 0]))]
        temp2 = np.unique(temp[:, 0], return_index=True)
        temp3 = temp[temp2[1]]

        # assert np.isclose(temp3, out).all()
        out = temp3

        p1 = track[out[:, 1]]  # (obs, 2)
        p2 = track[out[:, 1] + 1]  # (obs,2)
        p3 = np.stack([neigh_x, neigh_y], axis=2)  # [neigh, obs,2]

        proj_dist = np.abs(np.cross(p2 - p1,
                                    p1 - p3)) / np.linalg.norm(p2 - p1, axis=1)  # [neigh, obs]

        interested_neighbors = proj_dist < FRONT_OR_BACK_OFFSET_THRESHOLD

        track_replacement = np.concatenate([track[1][None], track[1:]], axis=0)
        dist_from_end_of_track = np.sqrt(
            (track_replacement[:, 0] - neigh_x) ** 2 +
            (track_replacement[:, 1] - neigh_y) ** 2)  # [neigh,obs]
        dist_from_start_of_track = np.sqrt(
            (track[0, 0] - neigh_x) ** 2 +
            (track[0, 1] - neigh_y) ** 2)  # [neigh,obs]
        dist_start_end = np.sqrt((track_replacement[:, 0] -
                                  track[0, 0]) ** 2 +
                                 (track_replacement[:, 1] -
                                  track[0, 1]) ** 2)  # [obs]

        frontback = np.where(np.logical_and(dist_from_end_of_track < dist_from_start_of_track,
                                            dist_from_start_of_track > dist_start_end) == True, "front",
                             "back")  # [neigh, obs]

        result = np.where(interested_neighbors == True, frontback, None)  # [neigh, obs]
        # Add back the index mask
        index_mask = np.any(index_array == 1, axis=1).astype(int)  # [obs]
        index_mask = np.repeat(index_mask[None], axis=0, repeats=neigh_x.shape[0])  # [neigh, obs]

        result2 = np.where(index_mask == True, result, None)

        return result2

    def get_min_distance_front_and_back_vec(self,
            agent_track: np.ndarray,
            social_tracks: np.ndarray,
            obs_len: int,
            raw_data_format: Dict[str, int],
    ) -> np.ndarray:
        """Get minimum distance of the tracks in front and in back.

        Args:
            agent_track (numpy array): Data for the agent track
            social_tracks (numpy array): Array of relevant tracks
            obs_len (int): Length of the observed trajectory
            raw_data_format (Dict): Format of the sequence
            viz (bool): Visualize tracks
        Returns:
            min_distance_front_and_back (numpy array): obs_len x 2, minimum front and back distances

        """

        # print(f"Shape {agent_track.shape} and {social_tracks.shape}")

        agent_x_vec = agent_track[:, raw_data_format["X"]].astype(float)  # [20]
        agent_y_vec = agent_track[:, raw_data_format["Y"]].astype(float)   # [20]
        neigh_x_vec1 = social_tracks[:, :, raw_data_format["X"]].astype(float)   # [neighbor, 20]
        neigh_y_vec1 = social_tracks[:, :, raw_data_format["Y"]].astype(float)   # [neighbor, 20]

        instant_distance_vec1 = np.sqrt((agent_x_vec - neigh_x_vec1) ** 2 +
                                        (agent_y_vec - neigh_y_vec1) ** 2)  # [neighbor, 20]

        instant_distance_vec_mask1 = np.where(instant_distance_vec1 < NEARBY_DISTANCE_THRESHOLD,
                                              1, np.inf)  # [neigh, obs]

        is_front_or_back = self.get_is_front_or_back_vec(
            agent_track,
            neigh_x_vec1,
            neigh_y_vec1,
            raw_data_format
        )  # [neigh, obs]

        if len(is_front_or_back) == 0:
            return np.full((obs_len, 2), self.DEFAULT_MIN_DIST_FRONT_AND_BACK)

        front_min_val = np.min(np.einsum('no,no->no', np.where(is_front_or_back == "front",
                                                               instant_distance_vec1, np.inf),
                                         instant_distance_vec_mask1), axis=0)
        back_min_val = np.min(np.einsum('no,no->no', np.where(is_front_or_back == "back",
                                                              instant_distance_vec1, np.inf),
                                        instant_distance_vec_mask1), axis=0)

        min_distance_front = np.minimum(front_min_val, DEFAULT_MIN_DIST_FRONT_AND_BACK)
        min_distance_back = np.minimum(back_min_val, DEFAULT_MIN_DIST_FRONT_AND_BACK)

        min_distance_front_and_back = np.stack([min_distance_front, min_distance_back], axis=1)


        return min_distance_front_and_back

    def get_num_neighbors_vec(self,
            agent_track: np.ndarray,
            social_tracks: np.ndarray,
            obs_len: int,
            raw_data_format: Dict[str, int],
    ) -> np.ndarray:
        """Get minimum distance of the tracks in front and back.

        Args:
            agent_track (numpy array): Data for the agent track [20,2]
            social_tracks (numpy array): Array of relevant tracks [19,20,2]
            obs_len (int): Length of observed trajectory
            raw_data_format (Dict): Format of the sequence
        Returns:
            num_neighbors (numpy array): Number of neighbors at each timestep

        """

        agent_x_vec = agent_track[:, raw_data_format["X"]].astype(float)  # [20]
        agent_y_vec = agent_track[:, raw_data_format["Y"]].astype(float)  # [20]

        neigh_x_vec = social_tracks[:, :, raw_data_format["X"]].astype(float)  # [neigh,20]
        neigh_y_vec = social_tracks[:, :, raw_data_format["Y"]].astype(float)  # [neigh,20]

        instant_distance_vec = np.sqrt((agent_x_vec - neigh_x_vec) ** 2 +
                                       (agent_y_vec - neigh_y_vec) ** 2)  #

        instant_distance_vec_mask = instant_distance_vec < NEARBY_DISTANCE_THRESHOLD
        instant_distance_vec_mask = instant_distance_vec_mask.astype(int)
        num_neighbors_vec = np.sum(instant_distance_vec_mask, axis=0).reshape(obs_len, 1)

        num_neighbors = num_neighbors_vec

        return num_neighbors
    def get_is_track_stationary(self, social_track) -> bool:
        vel_x, vel_y = zip(*[(
            social_track[i][0] - social_track[i-1][0],
            social_track[i][1] - social_track[i-1][1],
        ) for i in range(1, 20)])
        vel = [np.sqrt(x**2 + y**2) for x, y in zip(vel_x, vel_y)]
        sorted_vel = sorted(vel)
        threshold_vel = sorted_vel[self.STATIONARY_THRESHOLD]
        return True if threshold_vel < self.VELOCITY_THRESHOLD else False

    def filter_tracks(self, seq_df, obs_len: int,
                      raw_data_format: Dict[str, int]) -> np.ndarray:
        
        index = []
        for i in range(len(seq_df)-1,-1,-1):
            if len(np.unique(seq_df[i],axis=0)) < self.EXIST_THRESHOLD:
                continue
            if self.get_is_track_stationary(seq_df[i]):
                continue
            index.append(i)
        return index

    def compute_social_features_vec(self, trajectory: np.ndarray, social_index: list,
                                        agent_index: int, obs_len: int):
        """

        Args:
            trajectory: shape [agents, obs_len, 2], where 2 means x,y position, obs_len = 20
            agent_index: the index of the agent in the trajectory
            obs_len: the length of the observed trajectory

        Returns: social features for the agent shape [20, 3], 3 means [distance_front, distance_back, num_neighbors]

        """
        # Spte 0. Get agent track
        agent_track_obs = trajectory[agent_index, :obs_len, :]
        raw_data_format = {"X": 0, "Y": 1}
        
        social_tracks_obs = trajectory[social_index]
        assert social_tracks_obs.shape == (len(social_index), trajectory.shape[1], trajectory.shape[2])

        # Step 2. Get minimum following distance in front and back
        min_distance_front_and_back_obs = self.get_min_distance_front_and_back_vec(
            agent_track_obs,
            social_tracks_obs,
            obs_len,
            raw_data_format)

        num_neighbors_obs = self.get_num_neighbors_vec(agent_track_obs,
                                                   social_tracks_obs, obs_len,
                                                   raw_data_format)

        # Agent track with social features
        social_features_obs = np.concatenate(
            (min_distance_front_and_back_obs, num_neighbors_obs), axis=1)
        social_features = np.full((obs_len, social_features_obs.shape[1]), #####TODO, this is different from the original code, I use obs_len instead of seq_len
                                  None)
        social_features[:obs_len] = social_features_obs
        social_features = social_features.astype(float)

        return social_features
