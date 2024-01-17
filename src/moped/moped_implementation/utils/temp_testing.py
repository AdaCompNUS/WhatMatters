
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




def get_is_front_or_back(
        track: np.ndarray,
        neigh_x: float,
        neigh_y: float,
        raw_data_format: Dict[str, int],
) -> Optional[str]:
    """Check if the neighbor is in front or back of the track.

    Args:
        track (numpy array): Track data, shape [min(2, current_observation_index+1), 2]
        neigh_x (float): Neighbor x coordinate
        neigh_y (float): Neighbor y coordinate
    Returns:
        _ (str): 'front' if in front, 'back' if in back

    """
    # We don't have heading information. So we need at least 2 coordinates to determine that.
    # Here, front and back is determined wrt to last 2 coordinates of the track
    x2 = track[-1, raw_data_format["X"]]
    y2 = track[-1, raw_data_format["Y"]]

    # Keep taking previous coordinate until first distinct coordinate is found.
    idx1 = track.shape[0] - 2
    while idx1 > -1:
        x1 = track[idx1, raw_data_format["X"]]
        y1 = track[idx1, raw_data_format["Y"]]
        if x1 != x2 or y1 != y2:
            break
        idx1 -= 1

    # If all the coordinates in the track are the same, there's no way to find front/back
    if idx1 < 0:
        return None

    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    p3 = np.array([neigh_x, neigh_y])
    proj_dist = np.abs(np.cross(p2 - p1,
                                p1 - p3)) / np.linalg.norm(p2 - p1)

    # Interested in only those neighbors who are not far away from the direction of travel
    if proj_dist < FRONT_OR_BACK_OFFSET_THRESHOLD:

        dist_from_end_of_track = np.sqrt(
            (track[-1, raw_data_format["X"]] - neigh_x) ** 2 +
            (track[-1, raw_data_format["Y"]] - neigh_y) ** 2)
        dist_from_start_of_track = np.sqrt(
            (track[0, raw_data_format["X"]] - neigh_x) ** 2 +
            (track[0, raw_data_format["Y"]] - neigh_y) ** 2)
        dist_start_end = np.sqrt((track[-1, raw_data_format["X"]] -
                                  track[0, raw_data_format["X"]]) ** 2 +
                                 (track[-1, raw_data_format["Y"]] -
                                  track[0, raw_data_format["Y"]]) ** 2)

        return ("front"
                if dist_from_end_of_track < dist_from_start_of_track
                   and dist_from_start_of_track > dist_start_end else "back")

    else:
        return None

def get_min_distance_front_and_back(
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
    min_distance_front_and_back = np.full(
        (obs_len, 2), DEFAULT_MIN_DIST_FRONT_AND_BACK)

    print(f"Shape {agent_track.shape} and {social_tracks.shape}")

    # Compute distances for each timestep in the sequence
    for i in range(obs_len):

        # Agent coordinates
        agent_x, agent_y = (
            agent_track[i, raw_data_format["X"]],
            agent_track[i, raw_data_format["Y"]],
        )

        # Compute distances for all the social tracks
        for social_track in social_tracks[:, i, :]:

            neigh_x = social_track[raw_data_format["X"]]
            neigh_y = social_track[raw_data_format["Y"]]

            # Distance between agent and social
            instant_distance = np.sqrt((agent_x - neigh_x) ** 2 +
                                       (agent_y - neigh_y) ** 2)

            # If not a neighbor, continue
            if instant_distance > NEARBY_DISTANCE_THRESHOLD:
                continue

            # Check if the social track is in front or back
            is_front_or_back = get_is_front_or_back(
                agent_track[:2, :] if i == 0 else agent_track[:i + 1, :],
                neigh_x,
                neigh_y,
                raw_data_format,
            )
            if is_front_or_back == "front":
                min_distance_front_and_back[i, 0] = min(
                    min_distance_front_and_back[i, 0], instant_distance)

            elif is_front_or_back == "back":
                min_distance_front_and_back[i, 1] = min(
                    min_distance_front_and_back[i, 1], instant_distance)


    return min_distance_front_and_back

def get_num_neighbors(
            agent_track: np.ndarray,
            social_tracks: np.ndarray,
            obs_len: int,
            raw_data_format: Dict[str, int],
    ) -> np.ndarray:
        """Get minimum distance of the tracks in front and back.

        Args:
            agent_track (numpy array): Data for the agent track
            social_tracks (numpy array): Array of relevant tracks
            obs_len (int): Length of observed trajectory
            raw_data_format (Dict): Format of the sequence
        Returns:
            num_neighbors (numpy array): Number of neighbors at each timestep

        """
        num_neighbors = np.full((obs_len, 1), 0)

        for i in range(obs_len):

            agent_x, agent_y = (
                agent_track[i, raw_data_format["X"]],
                agent_track[i, raw_data_format["Y"]],
            )

            for social_track in social_tracks[:, i, :]:

                neigh_x = social_track[raw_data_format["X"]]
                neigh_y = social_track[raw_data_format["Y"]]

                instant_distance = np.sqrt((agent_x - neigh_x)**2 +
                                           (agent_y - neigh_y)**2)

                if instant_distance < NEARBY_DISTANCE_THRESHOLD:
                    num_neighbors[i, 0] += 1

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
        min_distance_front_and_back_obs = get_min_distance_front_and_back(
            agent_track_obs,
            social_tracks_obs,
            obs_len,
            raw_data_format)
        c = time.time()

        num_neighbors_obs = get_num_neighbors(agent_track_obs,
                                                   social_tracks_obs, obs_len,
                                                   raw_data_format)
        d = time.time()

        # Agent track with social features
        social_features_obs = np.concatenate(
            (min_distance_front_and_back_obs, num_neighbors_obs), axis=1)
        social_features = np.full((obs_len, social_features_obs.shape[1]), #####TODO, this is different from the original code, I use obs_len instead of seq_len
                                  None)
        social_features[:obs_len] = social_features_obs
        e = time.time()
        print(f"Time for social features: ea: {e-a} ba {b-a} cb {c-b} dc {d-c} ed {e-d}")

        return social_features

if __name__ == "__main__":
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