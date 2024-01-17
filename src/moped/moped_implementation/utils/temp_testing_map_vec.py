import random
import time
import numpy as np
import sys
sys.path.append('../')
sys.path.append('../summit_map/argoverse_api/map_utils')

from summit_map.summit_api.map_utils.summit_map_api import SummitMap
from summit_map.argoverse_api.map_utils.centerline_utils import get_nt_distance, remove_overlapping_lane_seq
from typing import List, Sequence
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import  unary_union

from utils.baseline_config import (
    _MANHATTAN_THRESHOLD,
    _DFS_THRESHOLD_FRONT_SCALE,
    _DFS_THRESHOLD_BACK_SCALE,
    _MAX_SEARCH_RADIUS_CENTERLINES,
    _MAX_CENTERLINE_CANDIDATES_TEST,
)

avm = SummitMap()


def get_point_in_polygon_score(lane_seq: List[int],
                               xy_seq: np.ndarray, city_name: str) -> int:
    """Get the number of coordinates that lie insde the lane seq polygon.

    Args:
        lane_seq: Sequence of lane ids
        xy_seq: Trajectory coordinates
        city_name: City name (PITT/MIA)
    Returns:
        point_in_polygon_score: Number of coordinates in the trajectory that lie within the lane sequence

    """
    lane_seq_polygon = unary_union([
        Polygon(avm.get_lane_segment_polygon(lane, city_name)).buffer(0)
        for lane in lane_seq
    ])
    point_in_polygon_score = 0
    for xy in xy_seq:
        point_in_polygon_score += lane_seq_polygon.contains(Point(xy))
    return point_in_polygon_score

def sort_lanes_based_on_point_in_polygon_score(
        lane_seqs: List[List[int]],
        xy_seq: np.ndarray,
        city_name: str,
) -> List[List[int]]:
    """Filter lane_seqs based on the number of coordinates inside the bounding polygon of lanes.

    Args:
        lane_seqs: Sequence of lane sequences
        xy_seq: Trajectory coordinates
        city_name: City name (PITT/MIA)
        avm: Argoverse map_api instance
    Returns:
        sorted_lane_seqs: Sequences of lane sequences sorted based on the point_in_polygon score

    """
    point_in_polygon_scores = []
    for lane_seq in lane_seqs:
        point_in_polygon_scores.append(
            get_point_in_polygon_score(lane_seq, xy_seq, city_name))
    randomized_tiebreaker = np.random.random(len(point_in_polygon_scores))
    sorted_point_in_polygon_scores_idx = np.lexsort(
        (randomized_tiebreaker, np.array(point_in_polygon_scores)))[::-1]
    sorted_lane_seqs = [
        lane_seqs[i] for i in sorted_point_in_polygon_scores_idx
    ]
    sorted_scores = [
        point_in_polygon_scores[i]
        for i in sorted_point_in_polygon_scores_idx
    ]
    return sorted_lane_seqs, sorted_scores

def get_heuristic_centerlines_for_test_set(
        lane_seqs: List[List[int]],
        xy_seq: np.ndarray,
        city_name: str,
        max_candidates: int,
        scores: List[int],
) -> List[np.ndarray]:
    """Sort based on distance along centerline and return the centerlines.

    Args:
        lane_seqs: Sequence of lane sequences
        xy_seq: Trajectory coordinates
        city_name: City name (PITT/MIA)
        avm: Argoverse map_api instance
        max_candidates: Maximum number of centerlines to return
    Return:
        sorted_candidate_centerlines: Centerlines in the order of their score

    """
    aligned_centerlines = []
    diverse_centerlines = []
    diverse_scores = []
    num_candidates = 0

    # Get first half as aligned centerlines
    aligned_cl_count = 0
    for i in range(len(lane_seqs)):
        lane_seq = lane_seqs[i]
        score = scores[i]
        diverse = True
        centerline = avm.get_cl_from_lane_seq([lane_seq], city_name)[0]
        if aligned_cl_count < int(max_candidates / 2):
            start_dist = LineString(centerline).project(Point(xy_seq[0]))
            end_dist = LineString(centerline).project(Point(xy_seq[-1]))
            if end_dist > start_dist:
                aligned_cl_count += 1
                aligned_centerlines.append(centerline)
                diverse = False
        if diverse:
            diverse_centerlines.append(centerline)
            diverse_scores.append(score)

    num_diverse_centerlines = min(len(diverse_centerlines),
                                  max_candidates - aligned_cl_count)
    test_centerlines = aligned_centerlines
    if num_diverse_centerlines > 0:
        probabilities = ([
                             float(score + 1) / (sum(diverse_scores) + len(diverse_scores))
                             for score in diverse_scores
                         ] if sum(diverse_scores) > 0 else [1.0 / len(diverse_scores)] *
                                                           len(diverse_scores))
        diverse_centerlines_idx = np.random.choice(
            range(len(probabilities)),
            num_diverse_centerlines,
            replace=False,
            p=probabilities,
        )
        diverse_centerlines = [
            diverse_centerlines[i] for i in diverse_centerlines_idx
        ]
        test_centerlines += diverse_centerlines

    return test_centerlines


def get_candidate_centerlines_for_trajectory(
        xy: np.ndarray,
        city_name: str,
        max_search_radius: float = 50.0,
        seq_len: int = 50,
        max_candidates: int = 10,
        mode: str = "test",
) -> List[np.ndarray]:
    """Get centerline candidates upto a threshold.

    Algorithm:
    1. Take the lanes in the bubble of last observed coordinate
    2. Extend before and after considering all possible candidates
    3. Get centerlines based on point in polygon score.

    Args:
        xy: Trajectory coordinates,
        city_name: City name,
        avm: Argoverse map_api instance,
        viz: Visualize candidate centerlines,
        max_search_radius: Max search radius for finding nearby lanes in meters,
        seq_len: Sequence length,
        max_candidates: Maximum number of centerlines to return,
        mode: train/val/test mode

    Returns:
        candidate_centerlines: List of candidate centerlines

    """
    # Get all lane candidates within a bubble
    a = time.time()
    manhattan_thresnold = _MANHATTAN_THRESHOLD

    curr_lane_candidates = avm.get_lane_ids_in_xy_bbox(
        xy[-1, 0], xy[-1, 1], city_name, manhattan_thresnold)
    b = time.time()

    # Keep expanding the bubble until at least 1 lane is found
    while (len(curr_lane_candidates) < 1
           and manhattan_thresnold < max_search_radius):
        manhattan_thresnold *= 2
        curr_lane_candidates = avm.get_lane_ids_in_xy_bbox(
            xy[-1, 0], xy[-1, 1], city_name, manhattan_thresnold)

    assert len(curr_lane_candidates) > 0, "No nearby lanes found!!"

    c = time.time()

    # Set dfs threshold
    traj_len = xy.shape[0]

    # Assuming a speed of 50 mps, set threshold for traversing in the front and back
    dfs_threshold_front = (_DFS_THRESHOLD_FRONT_SCALE *
                           (seq_len + 1 - traj_len) / 10)
    dfs_threshold_back = _DFS_THRESHOLD_BACK_SCALE * (traj_len +
                                                           1) / 10

    # DFS to get all successor and predecessor candidates
    obs_pred_lanes: List[List[int]] = []
    for lane in curr_lane_candidates:
        candidates_future = avm.dfs(lane, city_name, 0,
                                    dfs_threshold_front)
        candidates_past = avm.dfs(lane, city_name, 0, dfs_threshold_back,
                                  True)

        # Merge past and future
        for past_lane_seq in candidates_past:
            for future_lane_seq in candidates_future:
                assert (
                        past_lane_seq[-1] == future_lane_seq[0]
                ), "Incorrect DFS for candidate lanes past and future"
                obs_pred_lanes.append(past_lane_seq + future_lane_seq[1:])

    d = time.time()

    # Removing overlapping lanes
    obs_pred_lanes = remove_overlapping_lane_seq(obs_pred_lanes)
    e = time.time()

    # Sort lanes based on point in polygon score
    obs_pred_lanes, scores = sort_lanes_based_on_point_in_polygon_score(
        obs_pred_lanes, xy, city_name)
    f = time.time()
    # If the best centerline is not along the direction of travel, re-sort
    if mode == "test":
        candidate_centerlines = get_heuristic_centerlines_for_test_set(
            obs_pred_lanes, xy, city_name, max_candidates, scores)
    else:
        candidate_centerlines = avm.get_cl_from_lane_seq(
            [obs_pred_lanes[0]], city_name)

    g = time.time()

    print(f"Time taken for get_candidate_centerlines_for_trajectory: {e-a}, "
          f"each component: {b-a}, {c-b}, {d-c}, {e-d}, {f-e}, {g-f}")

    return candidate_centerlines


def compute_map_features_speedup(agent_track: np.ndarray, obs_len: int,
                                 seq_len: int,
                                 city_name: str):
    """
    Compute map features for the given sequence.
    agent_track : Data for the agent track, shape [obs_len, 2]
    obs_len: 20
    seq_len: obs_len + pred_len = 50

    """
    import time
    a = time.time()
    agent_xy = agent_track
    agent_track_obs = agent_track[:obs_len]
    agent_xy_obs = agent_track_obs


    b = time.time()

    # Get candidate centerlines using observed trajectory
    oracle_centerline = np.full((seq_len, 2), None)
    oracle_nt_dist = np.full((seq_len, 2), None)
    candidate_centerlines = get_candidate_centerlines_for_trajectory(
        agent_xy_obs,
        city_name,
        max_search_radius=_MAX_SEARCH_RADIUS_CENTERLINES,
        seq_len=seq_len,
        max_candidates=_MAX_CENTERLINE_CANDIDATES_TEST,
    )

    c = time.time()

    # Get nt distance for the entire trajectory using candidate centerlines
    candidate_nt_distances = []
    for candidate_centerline in candidate_centerlines:
        candidate_nt_distance = np.full((obs_len, 2), None)
        candidate_nt_distance[:obs_len] = get_nt_distance(
            agent_xy_obs, candidate_centerline)
        candidate_nt_distances.append(candidate_nt_distance)

    map_feature_helpers = {
        "ORACLE_CENTERLINE": oracle_centerline,
        "CANDIDATE_CENTERLINES": candidate_centerlines,
        "CANDIDATE_NT_DISTANCES": candidate_nt_distances,
    }

    d = time.time()
    print("time for map features: ", b - a, c - b, d - c)

    return oracle_nt_dist, map_feature_helpers


random.seed(42)
np.random.seed(42)

trajectories = np.random.normal(loc=[150,200], scale=1, size=(18,20,2))
#print(trajectories)
a = time.time()

agregated_list = []

for agent_index in range(trajectories.shape[0]):
    agent_track = trajectories[agent_index]
    social_features = compute_map_features_speedup(
        agent_track, 20, 50, "magic"
    )

    agregated_list.append(social_features)

b = time.time()
print(f"Time taken {b-a}")
print(f"agreegated_llist shape for each {[x.shape for x in agregated_list]}")
