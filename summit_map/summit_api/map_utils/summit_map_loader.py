#!/usr/bin/env python3
# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""
Utility to load the Argoverse vector map from disk, where it is stored in an XML format.

We release our Argoverse vector map in a modified OpenStreetMap (OSM) form. We also provide
the map data loader. OpenStreetMap (OSM) provides XML data and relies upon "Nodes" and "Ways" as
its fundamental element.

A "Node" is a point of interest, or a constituent point of a line feature such as a road.
In OpenStreetMap, a `Node` has tags, which might be
        -natural: If it's a natural feature, indicates the type (hill summit, etc)
        -man_made: If it's a man made feature, indicates the type (water tower, mast etc)
        -amenity: If it's an amenity (e.g. a pub, restaurant, recycling
            centre etc) indicates the type

In OSM, a "Way" is most often a road centerline, composed of an ordered list of "Nodes".
An OSM way often represents a line or polygon feature, e.g. a road, a stream, a wood, a lake.
Ways consist of two or more nodes. Tags for a Way might be:
        -highway: the class of road (motorway, primary,secondary etc)
        -maxspeed: maximum speed in km/h
        -ref: the road reference number
        -oneway: is it a one way road? (boolean)

However, in Argoverse, a "Way" corresponds to a LANE segment centerline. An Argoverse Way has the
following 9 attributes:
        -   id: integer, unique lane ID that serves as identifier for this "Way"
        -   has_traffic_control: boolean
        -   turn_direction: string, 'RIGHT', 'LEFT', or 'NONE'
        -   is_intersection: boolean
        -   l_neighbor_id: integer, unique ID for left neighbor
        -   r_neighbor_id: integer, unique ID for right neighbor
        -   predecessors: list of integers or None
        -   successors: list of integers or None
        -   centerline_node_ids: list

In Argoverse, a `LaneSegment` object is derived from a combination of a single `Way` and two or more
`Node` objects.
"""

import logging
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple, Union, cast

import numpy as np

logger = logging.getLogger(__name__)


_PathLike = Union[str, "os.PathLike[str]"]

class LaneSegment:
    def __init__(
        self,
        id: str,
        has_traffic_control: bool,
        turn_direction: str,
        is_intersection: bool,
        l_neighbor_id: Optional[str],
        r_neighbor_id: Optional[str],
        predecessors: List[str],
        successors: Optional[List[str]],
        index: str,
        centerline: np.ndarray,
    ) -> None:
        """Initialize the lane segment.

        Args:
            id: Unique lane ID that serves as identifier for this "Way"
            has_traffic_control:
            turn_direction: 'RIGHT', 'LEFT', or 'NONE'
            is_intersection: Whether or not this lane segment is an intersection
            l_neighbor_id: Unique ID for left neighbor
            r_neighbor_id: Unique ID for right neighbor
            predecessors: The IDs of the lane segments that come after this one
            successors: The IDs of the lane segments that come before this one.
            index: lane index (0 means lane_0, 1 means lane_1)
            centerline: The coordinates of the lane segment's center line.
        """
        self.id = id
        self.has_traffic_control = has_traffic_control
        self.turn_direction = turn_direction
        self.is_intersection = is_intersection
        self.l_neighbor_id = l_neighbor_id
        self.r_neighbor_id = r_neighbor_id
        self.predecessors = predecessors
        self.successors = successors
        self.index = index
        self.centerline = centerline

    def get_lane_id(self):
        return self.id

    def get_centerline(self):
        return self.centerline

    def set_successors(self, lane_id):
        # when reading <connection> tag
        if self.successors == None:
            self.successors = []
        self.successors.append(lane_id)
        self.successors = list(set(self.successors)) # this is to prevent double connection

    def set_predecessors(self, lane_id):
        # when reading <connection> tag
        if self.predecessors == None:
            self.predecessors = []
        self.predecessors.append(lane_id)
        self.predecessors = list(set(self.predecessors)) # this is to prevent double connection

    def set_turn_direction(self, direction):
        '''
        # when reading <connection> tag
        direction: string, 'RIGHT', 'LEFT', or 'NONE'
        '''
        if direction == "l" or direction == "L":
            self.turn_direction = "LEFT"
        elif direction == "r" or direction == "R":
            self.turn_direction = "RIGHT"
        else:
            self.turn_direction = "NONE"

    def set_traffic_control(self):
        self.has_traffic_control = True

    def set_intersection(self):
        self.is_intersection = True

class Junction:
    """
    e.g. a point of interest, or a constituent point of a
    line feature such as a road
    """

    def __init__(self, id: str, x: float, y: float, height: Optional[float],
                 lanes_within_junction = Optional[List[str]]):
        """
        Args:
            id: representing unique node ID
            x: x-coordinate in city reference system
            y: y-coordinate in city reference system

        Returns:
            None
        """
        self.id = id
        self.x = x
        self.y = y
        self.height = height
        self.lanes_within_junction = lanes_within_junction

    def get_lanes_within_junction(self) -> List[str]:
        return self.lanes_within_junction


def str_to_bool(s: str) -> bool:
    """
    Args:
       s: string representation of boolean, either 'True' or 'False'

    Returns:
       boolean
    """
    if s == "True":
        return True
    assert s == "False"
    return False


def convert_dictionary_to_lane_segment_obj(lane_id: str, lane_dictionary: Mapping[str, Any]) -> LaneSegment:
    """
    Not all lanes have predecessors and successors.

    Args:
       lane_id: representing unique lane ID
       lane_dictionary: dictionary with LaneSegment attributes, not yet in object instance form

    Returns:
       ls: LaneSegment object
    """
    predecessors = lane_dictionary.get("predecessor", None)
    successors = lane_dictionary.get("successor", None)
    has_traffic_control = str_to_bool(lane_dictionary["has_traffic_control"])
    is_intersection = str_to_bool(lane_dictionary["is_intersection"])
    lnid = lane_dictionary["l_neighbor_id"]
    rnid = lane_dictionary["r_neighbor_id"]
    l_neighbor_id = None if lnid == "None" else lnid
    r_neighbor_id = None if rnid == "None" else rnid
    index = lane_dictionary["index"]
    ls = LaneSegment(
        lane_id,
        has_traffic_control,
        lane_dictionary["turn_direction"],
        is_intersection,
        l_neighbor_id,
        r_neighbor_id,
        predecessors,
        successors,
        index,
        lane_dictionary["centerline"],
    )
    return ls


def extract_junction_from_ET_element(child: ET.Element) -> Junction:
    """
    Given a line of XML, build a node object. The "node_fields" dictionary will hold "id", "x", "y".
    The XML will resemble:

        <node id="0" x="3168.066310258233" y="1674.663991981186" />

    Args:
        child: xml.etree.ElementTree element

    Returns:
        Node object
    """
    node_fields = child.attrib
    node_id = node_fields["id"]
    # // Swap for SUMO -> CARLA.
    x = float(node_fields["y"])
    y = float(node_fields["x"])

    lanes_within_junction = node_fields.get("intLanes", "")
    lanes_within_junction = lanes_within_junction.split()

    return Junction(id=node_id, x=y, y=x, height=None, lanes_within_junction=lanes_within_junction)


def extract_lane_segment_from_ET_element(
    child: ET.Element
) -> Union[List[Tuple[LaneSegment, str]], None]:
    """
    We build a lane segment from an XML element. A lane segment is equivalent
    to an "Edge" in our XML file. Each Lane Segment has a polyline representing its centerline.
    The relevant XML data might resemble::

        <edge id="123">
            <lane id="123_0" index="0" shape="90.1, 12.2 11.2, 11.2">
            <lane id="123_1" index="1" shape="12.12,22.2 32.12, 43.23">
        </edge>

    Args:
        child: xml.etree.ElementTree element

    Returns:
        None to indicate this is internal edge and no need to read or
        Tuple (lane_segment: LaneSegment object, lane_id string)
    """
    output = []
    # Step 1. Disregard internal edge.
    # if child.attrib.get("function", None) == "internal":
    #     return None

    # Step 2. Get all the lane_id to have left and right neighbor lanes
    all_lanes = []
    for element in child:
        all_lanes.append(element.attrib["id"])

    for index, element in enumerate(child):

        lane_id = element.attrib["id"]
        lane_index = element.attrib["index"]

        lane_shape = element.attrib["shape"].split()
        centerline = []
        for shape in lane_shape:
            shape = shape.split(",")
            # // Swap for SUMO -> CARLA.
            x = float(shape[1])
            y = float(shape[0])
            centerline.append([x, y])
        centerline = np.array(centerline)

        # To prevent error in polygon calculation, we pad a little zeros if there is 1 unique centerline
        if np.unique(centerline, axis=0).shape[0] == 1:
            # only 1 centerline is unique, thus cannot find polygon
            centerline[0, :] = centerline[0, :] + 1e-8

        # Finding left-right neighbor
        left_neighbor = "None" if index-1 < 0 else all_lanes[index-1]
        right_neighbor = "None" if index+1 >= len(all_lanes) else all_lanes[index+1]

        lane_obj = {
            "predecessor": None, # found by <connection>
            "successor": None, # found by <connection>
            "has_traffic_control": "False", # found by <connection>
            "is_intersection": "False",  #TODO
            "turn_direction": "NONE", # found by <connection>
            "centerline": centerline,
            "index": lane_index,
            "l_neighbor_id": left_neighbor,
            "r_neighbor_id": right_neighbor
        }

        lane_segment = convert_dictionary_to_lane_segment_obj(lane_id, lane_obj)
        output.append((lane_segment, lane_id))

    return output

def extract_connection_from_xml(child: ET.Element, lane_objs: Dict[str, LaneSegment]) -> None:
    from_lane = f"{child.attrib['from']}_{child.attrib['fromLane']}"
    to_lane =  f"{child.attrib['to']}_{child.attrib['toLane']}"

    if from_lane in lane_objs.keys() and to_lane in lane_objs.keys():
        from_lane_segment = lane_objs[from_lane]
        to_lane_segment = lane_objs[to_lane]
        # Add successors (to_lane_segment) to from_lane_segment
        from_lane_segment.set_successors(to_lane_segment.get_lane_id())
        # set direction to from_lane_segment
        from_lane_segment.set_turn_direction(child.attrib["dir"])
        # Add predecessors (from_lane_segment) to to_lane_segment
        to_lane_segment.set_predecessors(from_lane_segment.get_lane_id())
        # set traffic control
        if child.attrib.get('tl', None) != None:
            from_lane_segment.set_traffic_control()
            to_lane_segment.set_traffic_control()

def set_intersection_for_lanes(lane_objs: Dict[str, LaneSegment], junction_objs: Dict[str, Junction]):
    """
    Set is_intersection for lane segment
    """
    lane_intersections = [lane_id for junction in junction_objs.values()
                          for lane_id in junction.get_lanes_within_junction()]

    for lane_id, lane_segment in lane_objs.items():
        if lane_id in lane_intersections:
            lane_segment.set_intersection()

def load_lane_segments_from_xml(map_fpath: _PathLike) -> Mapping[str, LaneSegment]:
    """
    Load lane segment object from xml file

    Args:
       map_fpath: path to xml file

    Returns:
       lane_objs: List of LaneSegment objects
    """
    tree = ET.parse(os.fspath(map_fpath))
    root = tree.getroot()

    #logger.info(f"Loaded root: {root.tag}")

    all_graph_nodes = {}
    lane_objs = {}
    # all children are either Nodes or Ways
    for child in root:
        if child.tag == "junction":
            node_obj = extract_junction_from_ET_element(child)
            all_graph_nodes[node_obj.id] = node_obj
        elif child.tag == "edge":
            edge_output = extract_lane_segment_from_ET_element(child)
            if edge_output:
                for lane_obj, lane_id in edge_output:
                    lane_objs[lane_id] = lane_obj
        elif child.tag == "connection":
            assert len(lane_objs) > 0 # connection tag must at the end
            extract_connection_from_xml(child, lane_objs)
        else:
            #print(f"Unknown XML item encountered: {child.tag}.")
            pass

    # Once having both lanes and junctions, we can set is_intersection of lane
    set_intersection_for_lanes(lane_objs=lane_objs, junction_objs=all_graph_nodes)

    return lane_objs
