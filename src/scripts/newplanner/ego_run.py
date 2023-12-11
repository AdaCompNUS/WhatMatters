#!/usr/bin/env python2

import os, sys, glob
import numpy as np
import math

summit_root = os.path.expanduser("~/summit/")
api_root = os.path.expanduser("~/summit/PythonAPI")
summit_connector_path = os.path.expanduser('~/catkin_ws/src/summit_connector/src/')

try:
    sys.path.append(glob.glob(api_root + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print(os.path.basename(__file__) + ": Cannot locate the CARLA egg file!!!")
    sys.exit()

sys.path.append(summit_connector_path)
from pathlib2 import Path

import carla
import summit
import random
from basic_agent import BasicAgent

DATA_PATH = Path(summit_root)/'Data'

def draw_waypoints(waypoints, world, color=carla.Color(255, 0, 0), life_time=50.0):

    for i in range(len(waypoints) - 1):
        world.debug.draw_line(
            carla.Location(waypoints[i].x, waypoints[i].y, 0.0),
            carla.Location(waypoints[i + 1].x, waypoints[i + 1].y, 0.0),
            2,
            color,
            life_time)


client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()

map_location = 'meskel_square'

with (DATA_PATH/'{}.sim_bounds'.format(map_location)).open('r') as f:
    bounds_min = carla.Vector2D(*[float(v) for v in f.readline().split(',')])
    bounds_max = carla.Vector2D(*[float(v) for v in f.readline().split(',')])

sumo_network = carla.SumoNetwork.load(str(DATA_PATH/'{}.net.xml'.format(map_location)))
sumo_network_segments = sumo_network.create_segment_map()
sumo_network_spawn_segments = sumo_network_segments.intersection(carla.OccupancyMap(bounds_min, bounds_max))
sumo_network_spawn_segments.seed_rand(42)
sumo_network_occupancy = carla.OccupancyMap.load(str(DATA_PATH/'{}.network.wkt'.format(map_location)))

def rand_path(sumo_network, min_points, interval, segment_map, min_safe_points=None, rng=random):
    if min_safe_points is None:
        min_safe_points = min_points

    spawn_point = None
    route_paths = None
    while not spawn_point or len(route_paths) < 1:
        spawn_point = segment_map.rand_point()
        spawn_point = sumo_network.get_nearest_route_point(spawn_point)
        route_paths = sumo_network.get_next_route_paths(spawn_point, min_safe_points - 1, interval)

    return rng.choice(route_paths)[0:min_points]

def get_position(path_point):
    return sumo_network.get_route_point_position(path_point)

def get_yaw(path_point, path_point_next):
    pos = sumo_network.get_route_point_position(path_point)
    next_pos = sumo_network.get_route_point_position(path_point_next)
    return np.rad2deg(math.atan2(next_pos.y - pos.y, next_pos.x - pos.x))

path = rand_path(
    sumo_network, 200, 1.0, sumo_network_spawn_segments,
    min_safe_points=100, rng=random.Random(42))

print("Path[0]: ", path[0], " position: ", get_position(path[0]))
# Creating carla waypoints

values = [get_position(path[i]) for i in range(len(path) - 1)]
print(values[0]) # A vector2D
draw_waypoints(values, world, color=carla.Color(r=0, g=255, b=0), life_time=20)

# We first reset the world
actor_list = world.get_actors()

# Find the actor with name "model3"
for actor in actor_list:
    if actor.type_id == "vehicle.tesla.model3":
        actor.destroy()
        break

# query for the cars blueprint.
vehicle_blueprint = client.get_world().get_blueprint_library().filter('model3')[0]

# We now need to obtain a spawn location.
spawn_point = values[10] # spawn at 10th position
spawn_trans = carla.Transform()
spawn_trans.location.x = spawn_point.x
spawn_trans.location.y = spawn_point.y
spawn_trans.location.z = 0.5
spawn_trans.rotation.yaw = get_yaw(path[10], path[11])
vehicle = client.get_world().try_spawn_actor(vehicle_blueprint, spawn_trans)

print('inspection')
print(vehicle.get_location())
print(vehicle.get_transform())
print(spawn_trans)
### Now is the control part
# To start a basic agent
agent = BasicAgent(vehicle)
agent.set_target_speed(6)

for i in range(50):
    vehicle.apply_control(agent.run_step())
