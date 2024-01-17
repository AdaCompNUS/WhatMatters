
import numpy as np
import random
from typing import Dict, Optional
import time

from baseline_config import (
    PADDING_TYPE,
    STATIONARY_THRESHOLD,
    VELOCITY_THRESHOLD,
    EXIST_THRESHOLD,
    DEFAULT_MIN_DIST_FRONT_AND_BACK,
    NEARBY_DISTANCE_THRESHOLD,
    FRONT_OR_BACK_OFFSET_THRESHOLD,
)

from temp_testing import get_is_front_or_back, get_min_distance_front_and_back, get_num_neighbors

def get_is_front_or_back_vec(
        track: np.ndarray,
        neigh_x: np.ndarray,
        neigh_y: np.ndarray
):
    """

    Args:
        track: [20, 2]
        neigh_x: [num_neigh, 20]
        neigh_y: [num_neigh, 20]

    Returns:
        [num_neigh] array of front/back

    """

    obs_len = 20

    # Step 1. Find the different between the consecutive coordinates (x,y) in the track
    diff_xy = np.diff(track, axis=0)# shape [19,2]
    # Step 2. Find the first non-zero different in either x or y (as long as either x or y is non-zero then it is valid)
    diff_xy_index = np.any(diff_xy != 0, axis=1) # shape [19]
    # Step 4, building a lower triangular matrix for each
    lower_triangular = np.tril(np.ones((obs_len-1, obs_len-1), dtype=np.int8))
    # Step 5. Broadcasting index to lower_triangular
    # At any row that is not having 1' index, we return None for that row
    # otherwise, the last 1 at each row in the index
    index_array = np.einsum('ij,j->ij', lower_triangular, diff_xy_index) # shape (19,19)
    # Add top row same as first row so two first row now are represented as at obs = 0 and obs = 1
    index_array = np.concatenate([index_array[0][None], index_array], axis=0) # shape [20, 19]

    # Create a masked_index so that every row contains at least 1's so that we can calculate p1, p2, p3
    index_array_fake = np.copy(index_array)
    index_array_fake[:,0] = 1

    # 2d array list where each row is i,j of 1's in index_array_fake
    ## TODO, now I can only to make it loop as i do not know how to use np
    ones_ij = np.argwhere(index_array_fake==1)
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

    #assert np.isclose(temp3, out).all()
    out = temp3

    p1 = track[out[:, 1]] # (obs, 2)
    p2 = track[out[:, 1] + 1] # (obs,2)
    p3 = np.stack([neigh_x, neigh_y], axis=2) # [neigh, obs,2]

    proj_dist = np.abs(np.cross(p2 - p1,
                                p1 - p3)) / np.linalg.norm(p2 - p1, axis=1) # [neigh, obs]

    interested_neighbors = proj_dist < FRONT_OR_BACK_OFFSET_THRESHOLD

    track_replacement = np.concatenate([track[1][None], track[1:]], axis=0)
    dist_from_end_of_track = np.sqrt(
        (track_replacement[:, 0] - neigh_x) ** 2 +
        (track_replacement[:, 1] - neigh_y) ** 2) #[neigh,obs]
    dist_from_start_of_track = np.sqrt(
        (track[0, 0] - neigh_x) ** 2 +
        (track[0, 1] - neigh_y) ** 2) #[neigh,obs]
    dist_start_end = np.sqrt((track_replacement[:, 0] -
                              track[0, 0]) ** 2 +
                             (track_replacement[:, 1] -
                              track[0, 1]) ** 2) #[obs]

    frontback = np.where(np.logical_and(dist_from_end_of_track < dist_from_start_of_track,
            dist_from_start_of_track > dist_start_end) == True, "front", "back") #[neigh, obs]

    result = np.where(interested_neighbors == True, frontback, None) #[neigh, obs]
    # Add back the index mask
    index_mask = np.any(index_array==1, axis=1).astype(int) #[obs]
    index_mask = np.repeat(index_mask[None], axis=0, repeats = neigh_x.shape[0])#[neigh, obs]

    result2 = np.where(index_mask==True, result, None)

    return result2

def get_min_distance_front_and_back_vec(
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

    #print(f"Shape {agent_track.shape} and {social_tracks.shape}")

    agent_x_vec = agent_track[:, 0] # [20]
    agent_y_vec = agent_track[:, 1]# [20]
    neigh_x_vec1 = social_tracks[:, :, 0] # [neighbor, 20]
    neigh_y_vec1 = social_tracks[:, :, 1] # [neighbor, 20]

    instant_distance_vec1 = np.sqrt((agent_x_vec - neigh_x_vec1) ** 2 +
                                    (agent_y_vec - neigh_y_vec1) ** 2) # [neighbor, 20]

    instant_distance_vec_mask1 = np.where(instant_distance_vec1 < NEARBY_DISTANCE_THRESHOLD,
                                         1, np.inf) #[neigh, obs]

    is_front_or_back = get_is_front_or_back_vec(
        agent_track,
        neigh_x_vec1,
        neigh_y_vec1
    ) #[neigh, obs]

    front_min_val = np.min(np.einsum('no,no->no', np.where(is_front_or_back == "front",
                                                 instant_distance_vec1, np.inf), instant_distance_vec_mask1), axis=0)
    back_min_val = np.min(np.einsum('no,no->no', np.where(is_front_or_back == "back",
                                                instant_distance_vec1, np.inf), instant_distance_vec_mask1), axis=0)

    min_distance_front = np.minimum(front_min_val, DEFAULT_MIN_DIST_FRONT_AND_BACK)
    min_distance_back = np.minimum(back_min_val, DEFAULT_MIN_DIST_FRONT_AND_BACK)

    min_distance_front_and_back = np.stack([min_distance_front, min_distance_back], axis=1)

    min_distance_front_and_backx = np.full(
        (obs_len, 2), DEFAULT_MIN_DIST_FRONT_AND_BACK)

    ## To test
    # for i in range(obs_len):
    #
    #     # Agent coordinates
    #     agent_x, agent_y = (
    #         agent_track[i, raw_data_format["X"]],
    #         agent_track[i, raw_data_format["Y"]],
    #     )
    #
    #     # Compute distances for all the social tracks
    #     neigh = 0
    #     for social_track in social_tracks[:, i, :]:
    #
    #         neigh_x = social_track[raw_data_format["X"]]
    #         neigh_y = social_track[raw_data_format["Y"]]
    #
    #         # Distance between agent and social
    #         instant_distance = np.sqrt((agent_x - neigh_x) ** 2 +
    #                                    (agent_y - neigh_y) ** 2)
    #
    #         assert instant_distance == instant_distance_vec1[neigh][i]
    #
    #         # If not a neighbor, continue
    #         if instant_distance > NEARBY_DISTANCE_THRESHOLD:
    #             continue
    #
    #         # Check if the social track is in front or back
    #         is_front_or_backx = get_is_front_or_back(
    #             agent_track[:2, :] if i == 0 else agent_track[:i + 1, :],
    #             neigh_x,
    #             neigh_y,
    #             raw_data_format,
    #         )
    #
    #         assert is_front_or_backx == is_front_or_back[neigh][i], f"first {is_front_or_backx} second " \
    #                                                                 f"{is_front_or_back[neigh][i]}"
    #
    #         if is_front_or_backx == "front":
    #             min_distance_front_and_backx[i, 0] = min(
    #                 min_distance_front_and_backx[i, 0], instant_distance)
    #
    #         elif is_front_or_backx == "back":
    #             min_distance_front_and_backx[i, 1] = min(
    #                 min_distance_front_and_backx[i, 1], instant_distance)
    #
    #         neigh += 1

    #assert np.isclose(min_distance_front_and_back, min_distance_front_and_backx).all()

    return min_distance_front_and_back

def get_num_neighbors_vec(
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

        agent_x_vec = agent_track[:, 0] # [20]
        agent_y_vec = agent_track[:, 1] # [20]

        neigh_x_vec = social_tracks[:,:, 0] # [neigh,20]
        neigh_y_vec = social_tracks[:,:, 1] # [neigh,20]

        instant_distance_vec = np.sqrt((agent_x_vec - neigh_x_vec) ** 2 +
                                   (agent_y_vec - neigh_y_vec) ** 2) #

        instant_distance_vec_mask = instant_distance_vec < NEARBY_DISTANCE_THRESHOLD
        instant_distance_vec_mask = instant_distance_vec_mask.astype(int)
        num_neighbors_vec = np.sum(instant_distance_vec_mask, axis=0).reshape(obs_len, 1)

        num_neighbors = num_neighbors_vec

        return num_neighbors

def compute_social_features_speedup(trajectory: np.ndarray,
                                        agent_index: int, obs_len: int):
        """

        Args:
            trajectory: shape [agents, obs_len, 2], where 2 means x,y position, obs_len = 20
            agent_index: the index of the agent in the trajectory
            obs_len: the length of the observed trajectory

        Returns: social features for the agent shape [20, 3], 3 means [distance_front, distance_back, num_neighbors]

        """

        # Spte 0. Get agent track
        import time
        a = time.time()
        agent_track_obs = trajectory[agent_index, :obs_len, :]
        raw_data_format = {"X": 0, "Y": 1}

        # Step 1. Filter track (no need to filter anything as we will predict all agents)
        social_tracks_obs = []
        for i in range(trajectory.shape[0]):
            if i != agent_index:
                social_tracks_obs.append(trajectory[i])
        # Must have shape [num_agents-1, obs_len, 2]
        social_tracks_obs = np.array(social_tracks_obs)
        assert social_tracks_obs.shape == (trajectory.shape[0]-1, trajectory.shape[1], 2)
        b = time.time()
        # Step 2. Get minimum following distance in front and back
        min_distance_front_and_back_obs = get_min_distance_front_and_back_vec(
            agent_track_obs,
            social_tracks_obs,
            obs_len,
            raw_data_format)
        c = time.time()

        # min_distance_front_and_back_obs1 = get_min_distance_front_and_back(
        #     agent_track_obs,
        #     social_tracks_obs,
        #     obs_len,
        #     raw_data_format)

        #assert np.isclose(min_distance_front_and_back_obs, min_distance_front_and_back_obs1).all()

        num_neighbors_obs = get_num_neighbors_vec(agent_track_obs,
                                                   social_tracks_obs, obs_len,
                                                   raw_data_format)

        num_neighbors_obs1 = get_num_neighbors_vec(agent_track_obs,
                                                   social_tracks_obs, obs_len,
                                                   raw_data_format)

        assert np.isclose(num_neighbors_obs, num_neighbors_obs1).all()

        d = time.time()

        # Agent track with social features
        social_features_obs = np.concatenate(
            (min_distance_front_and_back_obs, num_neighbors_obs), axis=1)
        social_features = np.full((obs_len, social_features_obs.shape[1]), #####TODO, this is different from the original code, I use obs_len instead of seq_len
                                  None)
        social_features[:obs_len] = social_features_obs
        e = time.time()
        #print(f"Time for social features: ea: {e-a} ba {b-a} cb {c-b} dc {d-c} ed {e-d}")

        return social_features

random.seed(42)
np.random.seed(42)

trajectories = np.random.randn(25,20,2)
#print(trajectories)
a = time.time()

agregated_list = []

for agent_index in range(trajectories.shape[0]):
    social_features = compute_social_features_speedup(
        trajectories, agent_index, 20
    )

    agregated_list.append(social_features)

b = time.time()
print(f"Time taken {b-a}")
#print(f"agreegated_llist shape for each {[x.shape for x in agregated_list]}")

a = time.time()
outputs = [compute_social_features_speedup(trajectories, i, 20) for i in range(trajectories.shape[0])]
b = time.time()
print(f"Time taken {b-a}")