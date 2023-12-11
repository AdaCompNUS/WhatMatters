
import datetime
import math
from typing import Iterable, List, Sequence, Set, Tuple, Mapping
import json
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LinearRing, LineString, Point
import os

from summit_map.summit_api.map_utils.summit_map_loader import LaneSegment
from summit_map.argoverse_api.map_utils.manhattan_search import compute_polygon_bboxes
from typing import Dict, Union

from summit_map.argoverse_api.map_utils.interpolate import interp_arc

_PathLike = Union[str, "os.PathLike[str]"]

def build_summit_bbox_and_tableidx(lane_objects: Mapping[str, LaneSegment],
                                   halluc_bbox_path: _PathLike, tableidx_to_laneid_path: _PathLike):
    '''
    Reading path_to_map, which is a sumo network file and return the halluc_bbox_table and tableidx_to_laneid_map
    '''
    tableidx = {}
    lane_polygons = []

    count = 0
    for lane_id, lane_segment in lane_objects.items():
        centerline = lane_segment.get_centerline()
        if np.unique(centerline, axis=0).shape[0] == 1:
            # only 1 centerline is unique, thus cannot find polygon
            #continue # if use this, have to remove successors/predecessors
            centerline[0,:] = centerline[0, :] + 1e-1 # 1e-1 is a small number enough. Used to be 1e-8 but not necessarily because the centerline is in meters
        polygon = centerline_to_polygon(centerline=centerline, width_scaling_factor=1.05)
        lane_polygons.append(polygon)

        tableidx[f"{count}"] = lane_id
        count += 1

    lane_bboxes = compute_polygon_bboxes(np.array(lane_polygons, dtype=object))

    np.save(halluc_bbox_path, lane_bboxes)

    with open(tableidx_to_laneid_path, "w") as f:
        json.dump(tableidx, f)

def swap_left_and_right(
    condition: np.ndarray, left_centerline: np.ndarray, right_centerline: np.ndarray
) -> Iterable[np.ndarray]:
    """
    Swap points in left and right centerline according to condition.

    Args:
       condition: Numpy array of shape (N,) of type boolean. Where true, swap the values in the left and
                   right centerlines.
       left_centerline: The left centerline, whose points should be swapped with the right centerline.
       right_centerline: The right centerline.

    Returns:
       left_centerline
       right_centerline
    """

    right_swap_indices = right_centerline[condition]
    left_swap_indices = left_centerline[condition]

    left_centerline[condition] = right_swap_indices
    right_centerline[condition] = left_swap_indices
    return left_centerline, right_centerline


def centerline_to_polygon(
    centerline: np.ndarray, width_scaling_factor: float = 1.0, visualize: bool = False
) -> np.ndarray:
    """
    Convert a lane centerline polyline into a rough polygon of the lane's area.

    On average, a lane is 3.8 meters in width. Thus, we allow 1.9 m on each side.
    We use this as the length of the hypotenuse of a right triangle, and compute the
    other two legs to find the scaled x and y displacement.

    Args:
       centerline: Numpy array of shape (N,2).
       width_scaling_factor: Multiplier that scales 3.8 meters to get the lane width.
       visualize: Save a figure showing the the output polygon.

    Returns:
       polygon: Numpy array of shape (2N+1,2), with duplicate first and last vertices.
    """
    # eliminate duplicates
    _, inds = np.unique(centerline, axis=0, return_index=True)
    #print(f'centerline {centerline}')
    # does not return indices in sorted order
    inds = np.sort(inds)
    centerline = centerline[inds]

    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])

    #print(f'dx {dx}')
    #print(f'dy {dy}')

    # To prevent dividing by 0, masking dx and dy with 1e-1
    dx = np.where(np.abs(dx) < 1e-8, 1e-8, dx)
    dy = np.where(np.abs(dy) < 1e-8, 1e-8, dy)

    # compute the normal at each point
    slopes = dy / dx
    inv_slopes = -1.0 / slopes

    thetas = np.arctan(inv_slopes)
    x_disp = 3.8 * width_scaling_factor / 2.0 * np.cos(thetas)
    y_disp = 3.8 * width_scaling_factor / 2.0 * np.sin(thetas)

    displacement = np.hstack([x_disp[:, np.newaxis], y_disp[:, np.newaxis]])
    right_centerline = centerline + displacement
    left_centerline = centerline - displacement

    # right centerline position depends on sign of dx and dy
    subtract_cond1 = np.logical_and(dx > 0, dy < 0)
    subtract_cond2 = np.logical_and(dx > 0, dy > 0)
    subtract_cond = np.logical_or(subtract_cond1, subtract_cond2)
    left_centerline, right_centerline = swap_left_and_right(subtract_cond, left_centerline, right_centerline)

    # right centerline also depended on if we added or subtracted y
    neg_disp_cond = displacement[:, 1] > 0
    left_centerline, right_centerline = swap_left_and_right(neg_disp_cond, left_centerline, right_centerline)

    if visualize:
        plt.scatter(centerline[:, 0], centerline[:, 1], 20, marker=".", color="b")
        plt.scatter(right_centerline[:, 0], right_centerline[:, 1], 20, marker=".", color="r")
        plt.scatter(left_centerline[:, 0], left_centerline[:, 1], 20, marker=".", color="g")
        fname = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")
        plt.savefig(f"polygon_unit_tests/{fname}.png")
        plt.close("all")

    # return the polygon
    return convert_lane_boundaries_to_polygon(right_centerline, left_centerline)


def convert_lane_boundaries_to_polygon(right_lane_bounds: np.ndarray, left_lane_bounds: np.ndarray) -> np.ndarray:
    """
    Take a left and right lane boundary and make a polygon of the lane segment, closing both ends of the segment.

    These polygons have the last vertex repeated (that is, first vertex == last vertex).

    Args:
       right_lane_bounds: Right lane boundary points. Shape is (N, 2).
       left_lane_bounds: Left lane boundary points.

    Returns:
       polygon: Numpy array of shape (2N+1,2)
    """
    assert right_lane_bounds.shape[0] == left_lane_bounds.shape[0]
    polygon = np.vstack([right_lane_bounds, left_lane_bounds[::-1]])
    polygon = np.vstack([polygon, right_lane_bounds[0]])
    return polygon

def lane_waypt_to_query_dist(
    query_xy_city_coords: np.ndarray, nearby_lane_objs: List[LaneSegment]
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Compute the distance from a query to the closest waypoint in nearby lanes.

    Args:
       query_xy_city_coords: Numpy array of shape (2,)
       nearby_lane_objs: list of LaneSegment objects

    Returns:
       per_lane_dists: array with distance to closest waypoint for each centerline
       min_dist_nn_indices: array with ranked indices of centerlines, closest first
       dense_centerlines: list of arrays, each representing (N,2) centerline
    """
    per_lane_dists: List[float] = []
    dense_centerlines: List[np.ndarray] = []
    for nn_idx, lane_obj in enumerate(nearby_lane_objs):
        centerline = lane_obj.centerline
        # densely sample more points
        sample_num = 50
        centerline = interp_arc(sample_num, centerline[:, 0], centerline[:, 1])
        dense_centerlines += [centerline]
        # compute norms to waypoints
        waypoint_dist = np.linalg.norm(centerline - query_xy_city_coords, axis=1).min()
        per_lane_dists += [waypoint_dist]
    per_lane_dists = np.array(per_lane_dists)
    min_dist_nn_indices = np.argsort(per_lane_dists)
    return per_lane_dists, min_dist_nn_indices, dense_centerlines
