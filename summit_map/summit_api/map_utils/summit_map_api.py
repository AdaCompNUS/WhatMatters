
import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union
import numpy as np
from summit_map.summit_api.map_utils.json_utils import read_json_file
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from scipy import ndimage

MAX_LABEL_DIST_TO_LANE = 20  # meters
OUT_OF_RANGE_LANE_DIST_THRESHOLD = 5.0  # 5 meters


from summit_map.summit_api.map_utils.summit_map_loader import LaneSegment, load_lane_segments_from_xml
from summit_map.summit_api.map_utils.summit_centerline_utils import build_summit_bbox_and_tableidx, lane_waypt_to_query_dist, centerline_to_polygon

# Add the path to import from argoverse library
import sys
MOPED_PATH = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(os.path.join(MOPED_PATH, "summit_map", "argoverse_api", "map_utils"))

from summit_map.argoverse_api.map_utils.manhattan_search import (
    compute_polygon_bboxes,
    find_all_polygon_bboxes_overlapping_query_bbox,
    find_local_polygons,
)
from summit_map.argoverse_api.map_utils.geometry import point_inside_polygon
from summit_map.argoverse_api.map_utils.dilation_utils import dilate_by_l2
from summit_map.argoverse_api.map_utils.se2 import SE2
from summit_map.argoverse_api.map_utils.cv2_plotting_utils import get_img_contours

# known City IDs from newest to oldest
BEIJING_ID = 1
CHANDNI_CHOWK_ID = 2
HIGHWAY_ID = 3
MAGIC_ID = 4
MESKEL_SQUARE_ID = 5
SHI_MEN_ER_LU_ID = 6

ROI_ISOCONTOUR = 5.0

ROOT = Path(__file__).resolve().parent.parent.parent / "map_files"


# Any numeric type
Number = Union[int, float]
_PathLike = Union[str, "os.PathLike[str]"]

class SummitMap():
    def __init__(self, root: _PathLike = ROOT) -> None:
        """Initialize the Argoverse Map."""
        self.root = root

        self.city_name_to_city_id_dict = {"beijing": BEIJING_ID, "chandni_chowk": CHANDNI_CHOWK_ID,
                                          "highway": HIGHWAY_ID, "magic": MAGIC_ID,
                                          "meskel_square": MESKEL_SQUARE_ID, "shi_men_er_lu": SHI_MEN_ER_LU_ID}

        self.city_lane_centerlines_dict = self.build_centerline_index()

        self.city_halluc_bbox_table, self.city_halluc_tableidx_to_laneid_map = self.build_hallucinated_lane_bbox_index()
        self.city_rasterized_da_roi_dict = self.build_city_driveable_area_roi_index()

        # get hallucinated lane extends and driveable area from binary img
        self.city_to_lane_polygons_dict: Mapping[str, np.ndarray] = {}
        self.city_to_driveable_areas_dict: Mapping[str, np.ndarray] = {}
        self.city_to_lane_bboxes_dict: Mapping[str, np.ndarray] = {}
        self.city_to_da_bboxes_dict: Mapping[str, np.ndarray] = {}

        for city_name in self.city_name_to_city_id_dict.keys():
            lane_polygons = np.array(self.get_vector_map_lane_polygons(city_name), dtype=object)
            driveable_areas = np.array(self.get_vector_map_driveable_areas(city_name), dtype=object)
            lane_bboxes = compute_polygon_bboxes(lane_polygons)
            da_bboxes = compute_polygon_bboxes(driveable_areas)

            self.city_to_lane_polygons_dict[city_name] = lane_polygons
            self.city_to_driveable_areas_dict[city_name] = driveable_areas
            self.city_to_lane_bboxes_dict[city_name] = lane_bboxes
            self.city_to_da_bboxes_dict[city_name] = da_bboxes
    
    def build_centerline_index(self) -> Mapping[str, Mapping[str, LaneSegment]]:
        """
        Build dictionary of centerline for each city, with lane_id as key

        Returns:
            city_lane_centerlines_dict:  Keys are city names, values are dictionaries
                                        (k=lane_id, v=lane info)
        """
        #print("Building centerline index...")
        city_lane_centerlines_dict = {}
        for city_name, city_id in self.city_name_to_city_id_dict.items():
            xml_fpath = self.map_files_root / f"summit_map/{city_name}.net.xml"
            city_lane_centerlines_dict[city_name] = load_lane_segments_from_xml(xml_fpath)

            # Extra to build .json map and .npy bbox
            json_fpath = self.map_files_root / f"summit_map/{city_name}_tableidx_to_laneid_map.json"
            npy_fpath = self.map_files_root / f"summit_map/{city_name}_halluc_bbox_table.npy"
            if not os.path.exists(json_fpath):

                build_summit_bbox_and_tableidx(lane_objects=city_lane_centerlines_dict[city_name],
                                        halluc_bbox_path=npy_fpath, tableidx_to_laneid_path=json_fpath)

        return city_lane_centerlines_dict

    @property
    def map_files_root(self) -> Path:
        if self.root is None:
            raise ValueError("Map root directory cannot be None!")
        return Path(self.root).resolve()

    def build_hallucinated_lane_bbox_index(
        self,
    ) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        """
        Populate the pre-computed hallucinated extent of each lane polygon, to allow for fast
        queries.

        Returns:
            city_halluc_bbox_table
            city_id_to_halluc_tableidx_map
        """

        city_halluc_bbox_table = {}
        city_halluc_tableidx_to_laneid_map = {}
        #print("Building hallucinated lane bbox index...")
        for city_name, city_id in self.city_name_to_city_id_dict.items():
            json_fpath = self.map_files_root / f"summit_map/{city_name}_tableidx_to_laneid_map.json"
            city_halluc_tableidx_to_laneid_map[city_name] = read_json_file(json_fpath)

            npy_fpath = self.map_files_root / f"summit_map/{city_name}_halluc_bbox_table.npy"
            city_halluc_bbox_table[city_name] = np.load(npy_fpath)

        return city_halluc_bbox_table, city_halluc_tableidx_to_laneid_map
    

    def build_city_driveable_area_roi_index(
        self,
    ) -> Mapping[str, Mapping[str, np.ndarray]]:
        """
        Load driveable area files from disk. Dilate driveable area to get ROI (takes about 1/2 second).
        Returns:
            city_rasterized_da_dict: a dictionary of dictionaries. Key is city_name, and
                    value is a dictionary with driveable area info. For example, includes da_matrix: Numpy array of
                    shape (M,N) representing binary values for driveable area
                    city_to_pkl_image_se2: SE(2) that produces takes point in pkl image to city coordinates, e.g.
                    p_city = city_Transformation_pklimage * p_pklimage
        """
        city_rasterized_da_roi_dict: Dict[str, Dict[str, np.ndarray]] = {}
        for city_name, city_id in self.city_name_to_city_id_dict.items():
            city_id = self.city_name_to_city_id_dict[city_name]
            city_rasterized_da_roi_dict[city_name] = {}
            npy_fpath = self.map_files_root / f"summit_map/{city_name}_driveable_area_mat.npy"
            load_da = np.load(npy_fpath).astype(np.uint8)
            load_da = ndimage.binary_fill_holes(load_da).astype(np.uint8)  #### adding for HOME --- Really SB
            city_rasterized_da_roi_dict[city_name]["da_mat"] = load_da

            se2_npy_fpath = self.map_files_root / f"summit_map/{city_name}_npyimage_to_city_se2.npy"
            city_rasterized_da_roi_dict[city_name]["npyimage_to_city_se2"] = np.load(se2_npy_fpath)
            da_mat = copy.deepcopy(city_rasterized_da_roi_dict[city_name]["da_mat"])
            
            city_rasterized_da_roi_dict[city_name]["roi_mat"] = dilate_by_l2(da_mat, dilation_thresh=ROI_ISOCONTOUR)

        return city_rasterized_da_roi_dict
    
    def get_vector_map_lane_polygons(self, city_name: str) -> List[np.ndarray]:
        """
        Get list of lane polygons for a specified city
        Args:
           city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
        Returns:
           Numpy array of polygons
        """
        lane_polygons = []
        lane_segments = self.city_lane_centerlines_dict[city_name]
        for lane_id, lane_segment in lane_segments.items():
            lane_polygon_xyz = self.get_lane_segment_polygon(lane_segment.id, city_name)
            lane_polygons.append(lane_polygon_xyz)

        return lane_polygons
    def get_vector_map_driveable_areas(self, city_name: str) -> List[np.ndarray]:
        """
        Get driveable area for a specified city

        Args:
           city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
           das: driveable areas as n-d array of NumPy objects of shape (n,3)

        Note:
         'z_min', 'z_max' were removed
        """
        return self.get_da_contours(city_name)

    def get_da_contours(self, city_name: str) -> List[np.ndarray]:
        """
        We threshold the binary driveable area or ROI image and obtain contour lines. These
        contour lines represent the boundary.

        Args:
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            Drivable area contours
        """
        da_imgray = self.city_rasterized_da_roi_dict[city_name]["da_mat"]
        contours = get_img_contours(da_imgray)

        # pull out 3x3 matrix parameterizing the SE(2) transformation from city coords -> npy image
        npyimage_T_city = self.city_rasterized_da_roi_dict[city_name]["npyimage_to_city_se2"]
        R = npyimage_T_city[:2, :2]
        t = npyimage_T_city[:2, 2]
        npyimage_SE2_city = SE2(rotation=R, translation=t)
        city_SE2_npyimage = npyimage_SE2_city.inverse()

        city_contours: List[np.ndarray] = []
        for i, contour_im_coords in enumerate(contours):
            assert contour_im_coords.shape[1] == 1
            contour_im_coords = contour_im_coords.squeeze(axis=1)
            contour_im_coords = contour_im_coords.astype(np.float64)

            contour_city_coords = city_SE2_npyimage.transform_point_cloud(contour_im_coords)
            city_contours.append(self.append_height_to_2d_city_pt_cloud(contour_city_coords, city_name))

        return city_contours
    
    def get_rasterized_driveable_area(self, city_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the driveable area.
        Args:
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
        Returns:
            da_mat: Numpy array of shape (M,N) representing binary values for driveable area
            city_to_pkl_image_se2: SE(2) that produces takes point in pkl image to city coordinates, e.g.
                    p_city = city_Transformation_pklimage * p_pklimage
        """
        da_mat = self.city_rasterized_da_roi_dict[city_name]["da_mat"]
        return (
            da_mat,
            self.city_rasterized_da_roi_dict[city_name]["npyimage_to_city_se2"],
        )

    def get_rasterized_roi(self, city_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the region of interest (5 meter dilation of driveable area).
        Args:
            city_name: string, either 'MIA' for Miami or 'PIT' for Pittsburgh
        Returns:
            roi_mat: Numpy array of shape (M,N) representing binary values for the region of interest.
            city_to_pkl_image_se2: SE(2) that produces takes point in pkl image to city coordinates, e.g.
                    p_city = city_Transformation_pklimage * p_pklimage
        """
        roi_mat = self.city_rasterized_da_roi_dict[city_name]["roi_mat"]
        return (
            roi_mat,
            self.city_rasterized_da_roi_dict[city_name]["npyimage_to_city_se2"],
        )

    def get_nearest_centerline(
        self, query_xy_city_coords: np.ndarray, city_name: str, visualize: bool = False
    ) -> Tuple[LaneSegment, float, np.ndarray]:
        """
        KD Tree with k-closest neighbors or a fixed radius search on the lane centroids
        is unreliable since (1) there is highly variable density throughout the map and (2)
        lane lengths differ enormously, meaning the centroid is not indicative of nearby points.
        If no lanes are found with MAX_LABEL_DIST_TO_LANE, we increase the search radius.

        A correct approach is to compare centerline-to-query point distances, e.g. as done
        in Shapely. Instead of looping over all points, we precompute the bounding boxes of
        each lane.

        We use the closest_waypoint as our criterion. Using the smallest sum to waypoints
        does not work in many cases with disproportionately shaped lane segments.

        and then choose the lane centerline with the smallest sum of 3-5
        closest waypoints.

        Args:
            query_xy_city_coords: Numpy array of shape (2,) representing xy position of query in city coordinates
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
            visualize:

        Returns:
            lane_object: Python dictionary with fields describing a lane.
                Keys include: 'centerline', 'predecessor', 'successor', 'turn_direction',
                             'is_intersection', 'has_traffic_control', 'is_autonomous', 'is_routable'
            conf: real-valued confidence. less than 0.85 is almost always unreliable
            dense_centerline: numpy array
        """
        #print("Getting nearest centerline...")
        query_x = query_xy_city_coords[0]
        query_y = query_xy_city_coords[1]

        lane_centerlines_dict = self.city_lane_centerlines_dict[city_name]

        search_radius = MAX_LABEL_DIST_TO_LANE
        while True:
            nearby_lane_ids = self.get_lane_ids_in_xy_bbox(
                query_x, query_y, city_name, query_search_range_manhattan=search_radius
            )
            if not nearby_lane_ids:
                search_radius *= 2  # double search radius
            else:
                break

        nearby_lane_objs = [lane_centerlines_dict[lane_id] for lane_id in nearby_lane_ids]

        cache = lane_waypt_to_query_dist(query_xy_city_coords, nearby_lane_objs)
        per_lane_dists, min_dist_nn_indices, dense_centerlines = cache

        closest_lane_obj = nearby_lane_objs[min_dist_nn_indices[0]]
        dense_centerline = dense_centerlines[min_dist_nn_indices[0]]

        # estimate confidence
        conf = 1.0 - (per_lane_dists.min() / OUT_OF_RANGE_LANE_DIST_THRESHOLD)
        conf = max(0.0, conf)  # clip to ensure positive value

        if visualize:
            # visualize dists to nearby centerlines
            fig = plt.figure(figsize=(22.5, 8))
            ax = fig.add_subplot(111)

            (query_x, query_y) = query_xy_city_coords.squeeze()
            ax.scatter([query_x], [query_y], 100, color="k", marker=".")
            # make another plot now!

            self.plot_nearby_halluc_lanes(ax, city_name, query_x, query_y)

            for i, line in enumerate(dense_centerlines):
                ax.plot(line[:, 0], line[:, 1], color="y")
                ax.text(line[:, 0].mean(), line[:, 1].mean(), str(per_lane_dists[i]))

            ax.axis("equal")
            plt.show()
            plt.close("all")
        return closest_lane_obj, conf, dense_centerline

    def get_lane_ids_in_xy_bbox(
        self,
        query_x: float,
        query_y: float,
        city_name: str,
        query_search_range_manhattan: float = 5.0,
    ) -> List[int]:
        """
        Prune away all lane segments based on Manhattan distance. We vectorize this instead
        of using a for-loop. Get all lane IDs within a bounding box in the xy plane.
        This is a approximation of a bubble search for point-to-polygon distance.

        The bounding boxes of small point clouds (lane centerline waypoints) are precomputed in the map.
        We then can perform an efficient search based on manhattan distance search radius from a
        given 2D query point.

        We pre-assign lane segment IDs to indices inside a big lookup array, with precomputed
        hallucinated lane polygon extents.

        Args:
            query_x: representing x coordinate of xy query location
            query_y: representing y coordinate of xy query location
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
            query_search_range_manhattan: search radius along axes

        Returns:
            lane_ids: lane segment IDs that live within a bubble
        """
        #print("Getting lane ids in xy bbox...")
        query_min_x = query_x - query_search_range_manhattan
        query_max_x = query_x + query_search_range_manhattan
        query_min_y = query_y - query_search_range_manhattan
        query_max_y = query_y + query_search_range_manhattan

        overlap_indxs = find_all_polygon_bboxes_overlapping_query_bbox(
            self.city_halluc_bbox_table[city_name],
            np.array([query_min_x, query_min_y, query_max_x, query_max_y]),
        )

        if len(overlap_indxs) == 0:
            return []

        neighborhood_lane_ids: List[int] = []
        for overlap_idx in overlap_indxs:
            lane_segment_id = self.city_halluc_tableidx_to_laneid_map[city_name][str(overlap_idx)]
            neighborhood_lane_ids.append(lane_segment_id)

        return neighborhood_lane_ids

    def get_lane_segment_predecessor_ids(self, lane_segment_id: str, city_name: str) -> List[str]:
        """
        Get land id for the lane predecessor of the specified lane_segment_id

        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            predecessor_ids: list of integers, representing lane segment IDs of predecessors
        """
        predecessor_ids = self.city_lane_centerlines_dict[city_name][lane_segment_id].predecessors
        return predecessor_ids

    def get_lane_segment_successor_ids(self, lane_segment_id: str, city_name: str) -> Optional[List[str]]:
        """
        Get land id for the lane sucessor of the specified lane_segment_id

        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh

        Returns:
            successor_ids: list of integers, representing lane segment IDs of successors
        """
        successor_ids = self.city_lane_centerlines_dict[city_name][lane_segment_id].successors
        return successor_ids
    
    def get_lane_segment_adjacent_ids(self, lane_segment_id: str, city_name: str) -> List[Optional[str]]:
        """
        Get land id for the lane adjacent left/right neighbor of the specified lane_segment_id
        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh
        Returns:
            adjacent_ids: list of integers, representing lane segment IDs of adjacent
                            left/right neighbor lane segments
        """
        r_neighbor = self.city_lane_centerlines_dict[city_name][lane_segment_id].r_neighbor_id
        l_neighbor = self.city_lane_centerlines_dict[city_name][lane_segment_id].l_neighbor_id
        adjacent_ids = [r_neighbor, l_neighbor]
        return adjacent_ids
    
    def get_ground_height_at_xy(self, point_cloud: np.ndarray, city_name: str) -> np.ndarray:
        """Get ground height for each of the xy locations in a point cloud.

        For SUMMIT, no height is given, thus return nans

        Args:
            point_cloud: Numpy array of shape (k,2) or (k,3)
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            ground_height_values: Numpy array of shape (k,)
        """
        city_coords = np.round(point_cloud[:, :2]).astype(np.int64)
        ground_height_values = np.full((city_coords.shape[0]), np.nan)

        return ground_height_values

    def append_height_to_2d_city_pt_cloud(self, pt_cloud_xy: np.ndarray, city_name: str) -> np.ndarray:
        """Accept 2d point cloud in xy plane and return 3d point cloud (xyz).

        Args:
            pt_cloud_xy: Numpy array of shape (N,2)
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            pt_cloud_xyz: Numpy array of shape (N,3)
        """
        pts_z = self.get_ground_height_at_xy(pt_cloud_xy, city_name)
        return np.hstack([pt_cloud_xy, pts_z[:, np.newaxis]])

    def get_lane_segment_centerline(self, lane_segment_id: str, city_name: str) -> np.ndarray:
        """
        We return a 3D centerline for any particular lane segment.

        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh

        Returns:
            lane_centerline: Numpy array of shape (N,3)
        """
        lane_centerline = self.city_lane_centerlines_dict[city_name][lane_segment_id].centerline
        if len(lane_centerline[0]) == 2:
            lane_centerline = self.append_height_to_2d_city_pt_cloud(lane_centerline, city_name)

        return lane_centerline

    def get_lane_segment_polygon(self, lane_segment_id: str, city_name: str) -> np.ndarray:
        """
        Hallucinate a 3d lane polygon based around the centerline. We rely on the average
        lane width within our cities to hallucinate the boundaries. We rely upon the
        rasterized maps to provide heights to points in the xy plane.

        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh

        Returns:
            lane_polygon: Array of polygon boundary (K,3), with identical and last boundary points
        """
        lane_centerline = self.city_lane_centerlines_dict[city_name][lane_segment_id].centerline
        lane_polygon = centerline_to_polygon(lane_centerline[:, :2])
        return self.append_height_to_2d_city_pt_cloud(lane_polygon, city_name)

    def dfs(
        self,
        lane_id: str,
        city_name: str,
        dist: float = 0,
        threshold: float = 30,
        extend_along_predecessor: bool = False,
    ) -> List[List[str]]:
        """
        Perform depth first search over lane graph up to the threshold.

        Args:
            lane_id: Starting lane_id (Eg. 12345)
            city_name
            dist: Distance of the current path
            threshold: Threshold after which to stop the search
            extend_along_predecessor: if true, dfs over predecessors, else successors

        Returns:
            lanes_to_return (list of list of integers): List of sequence of lane ids
                Eg. [[12345, 12346, 12347], [12345, 12348]]

        """
        if dist > threshold:
            return [[lane_id]]
        else:
            traversed_lanes = []
            child_lanes = (
                self.get_lane_segment_predecessor_ids(lane_id, city_name)
                if extend_along_predecessor
                else self.get_lane_segment_successor_ids(lane_id, city_name)
            )
            if child_lanes is not None:
                for child in child_lanes:
                    centerline = self.get_lane_segment_centerline(child, city_name)
                    cl_length = LineString(centerline).length
                    curr_lane_ids = self.dfs(
                        child,
                        city_name,
                        dist + cl_length,
                        threshold,
                        extend_along_predecessor,
                    )
                    traversed_lanes.extend(curr_lane_ids)
            if len(traversed_lanes) == 0:
                return [[lane_id]]
            lanes_to_return = []
            for lane_seq in traversed_lanes:
                lanes_to_return.append(lane_seq + [lane_id] if extend_along_predecessor else [lane_id] + lane_seq)
            return lanes_to_return
    
    def get_raster_layer_points_boolean(self, point_cloud: np.ndarray, city_name: str, layer_name: str) -> np.ndarray:
        """
        driveable area is "da"

        Args:
            point_cloud: Numpy array of shape (N,3)
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
            layer_name: indicating layer name, either "roi" or "driveable area"

        Returns:
            is_ground_boolean_arr: Numpy array of shape (N,) where ith entry is True if the LiDAR return
                is likely a hit from the ground surface.
        """
        if layer_name == "roi":
            layer_raster_mat, npyimage_to_city_se2_mat = self.get_rasterized_roi(city_name)
        elif layer_name == "driveable_area":
            (
                layer_raster_mat,
                npyimage_to_city_se2_mat,
            ) = self.get_rasterized_driveable_area(city_name)
        else:
            raise ValueError("layer_name should be wither roi or driveable_area.")

        city_coords = np.round(point_cloud[:, :2]).astype(np.int64)

        se2_rotation = npyimage_to_city_se2_mat[:2, :2]
        se2_trans = npyimage_to_city_se2_mat[:2, 2]

        npyimage_to_city_se2 = SE2(rotation=se2_rotation, translation=se2_trans)
        npyimage_coords = npyimage_to_city_se2.transform_point_cloud(city_coords)
        npyimage_coords = npyimage_coords.astype(np.int64)

        # index in at (x,y) locations, which are (y,x) in the image
        layer_values = np.full((npyimage_coords.shape[0]), 0.0)
        ind_valid_pts = (
            (npyimage_coords[:, 1] > 0)
            * (npyimage_coords[:, 1] < layer_raster_mat.shape[0])
            * (npyimage_coords[:, 0] > 0)
            * (npyimage_coords[:, 0] < layer_raster_mat.shape[1])
        )
        layer_values[ind_valid_pts] = layer_raster_mat[
            npyimage_coords[ind_valid_pts, 1], npyimage_coords[ind_valid_pts, 0]
        ]
        is_layer_boolean_arr = layer_values == 1.0
        return is_layer_boolean_arr
    
    def get_cl_from_lane_seq(self, lane_seqs: Iterable[List[str]], city_name: str) -> List[np.ndarray]:
        """Get centerlines corresponding to each lane sequence in lane_sequences

        Args:
            lane_seqs: Iterable of sequence of lane ids (Eg. [[12345, 12346, 12347], [12345, 12348]])
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            candidate_cl: list of numpy arrays for centerline corresponding to each lane sequence
        """
        candidate_cl = []
        for lanes in lane_seqs:
            curr_candidate_cl = np.empty((0, 2))
            for curr_lane in lanes:
                curr_candidate = self.get_lane_segment_centerline(curr_lane, city_name)[:, :2]
                curr_candidate_cl = np.vstack((curr_candidate_cl, curr_candidate))
            candidate_cl.append(curr_candidate_cl)
        return candidate_cl
    
    def lane_is_in_intersection(self, lane_segment_id: str, city_name: str) -> bool:
        """
        Check if the specified lane_segment_id falls within an intersection

        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh

        Returns:
            is_intersection: indicating if lane segment falls within an
                intersection
        """
        return self.city_lane_centerlines_dict[city_name][lane_segment_id].is_intersection
    
    def get_lane_turn_direction(self, lane_segment_id: str, city_name: str) -> str:
        """
        Get left/right/none direction of the specified lane_segment_id

        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh

        Returns:
            turn_direction: string, can be 'RIGHT', 'LEFT', or 'NONE'
        """
        return self.city_lane_centerlines_dict[city_name][lane_segment_id].turn_direction

    def lane_has_traffic_control_measure(self, lane_segment_id: str, city_name: str) -> bool:
        """
        You can have an intersection without a control measure.

        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh

        Returns:
            has_traffic_control: indicating if lane segment has a
                traffic control measure
        """
        return self.city_lane_centerlines_dict[city_name][lane_segment_id].has_traffic_control


    def get_lane_segments_containing_xy(self, query_x: float, query_y: float, city_name: str) -> List[int]:
        """
        Get the occupied lane ids, i.e. given (x,y), list those lane IDs whose hallucinated
        lane polygon contains this (x,y) query point.
        This function performs a "point-in-polygon" test.
        Args:
            query_x: representing x coordinate of xy query location
            query_y: representing y coordinate of xy query location
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
        Returns:
            occupied_lane_ids: list of integers, representing lane segment IDs containing (x,y)
        """
        neighborhood_lane_ids = self.get_lane_ids_in_xy_bbox(query_x, query_y, city_name)

        occupied_lane_ids: List[int] = []
        if neighborhood_lane_ids is not None:
            for lane_id in neighborhood_lane_ids:
                lane_polygon = self.get_lane_segment_polygon(lane_id, city_name)
                inside = point_inside_polygon(
                    lane_polygon.shape[0],
                    lane_polygon[:, 0],
                    lane_polygon[:, 1],
                    query_x,
                    query_y,
                )
                if inside:
                    occupied_lane_ids += [lane_id]
        return occupied_lane_ids


if __name__ == "__main__":
    summitmap = SummitMap()
