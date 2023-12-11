import numpy as np
from typing import List
from shapely.geometry import Polygon
import math

def get_corners(agent):
    width, length = agent['bb']
    x, y = agent['pos']
    heading = agent['heading']

    dx = length / 2
    dy = width / 2

    corners = [
        [x - dx, y - dy],
        [x + dx, y - dy],
        [x + dx, y + dy],
        [x - dx, y + dy],
    ]

    # Rotate corners based on heading
    rotated_corners = []
    for corner in corners:
        x_diff = corner[0] - x
        y_diff = corner[1] - y
        new_x = x + (x_diff * np.cos(heading) - y_diff * np.sin(heading))
        new_y = y + (x_diff * np.sin(heading) + y_diff * np.cos(heading))
        rotated_corners.append([new_x, new_y])

    return rotated_corners

### ------------------ Safety 1 - Collision rate --------------------------
def check_collision(ego, exo):
    # Calculate the distance between ego and exo
    dist = np.sqrt((ego['pos'][0] - exo['pos'][0])**2 + (ego['pos'][1] - exo['pos'][1])**2)

    # Calculate the sum of the ego and exo bounding box extents along the x and y axes
    box_extents_sum = (ego['bb'][0] + exo['bb'][0]) / 2 + (ego['bb'][1] + exo['bb'][1]) / 2

    # Check for collision
    return dist < box_extents_sum

def check_collision_by_distance(ego, exo):
    x, y = ego['pos']
    x_exo, y_exo = exo['pos']
    return np.sqrt((x-x_exo)**2 + (y-y_exo)**2)

def check_collision_by_considering_headings(ego, exo, buffer):
    ego_corners = get_corners(ego)
    exo_corners = get_corners(exo)
    
    ego_polygon = Polygon(ego_corners)
    exo_polygon = Polygon(exo_corners)

     # Calculate the buffered ego polygon with the near_miss_threshold
    buffered_ego_polygon = ego_polygon.buffer(buffer)

    return buffered_ego_polygon.intersects(exo_polygon)

def find_collision_rate(ego_dict, exos_dict):
    collision_count = 0
    total_time_steps = len(ego_dict)
    collided_exo_ids = set()

    # Iterate through each time step in the episode
    for time_step in list(ego_dict.keys()):
        # We are not sure time_step is in exos_dict so we check
        if time_step in exos_dict.keys():
            ego_agent = ego_dict[time_step]
            exo_agents = exos_dict[time_step]

            # Chck for collisions with exo cars at the current time step
            for exo_agent in exo_agents:
                if check_collision_by_considering_headings(ego_agent, exo_agent) and exo_agent['id'] not in collided_exo_ids:
                    collision_count += 1
                    #print(f"Collision t {time_step} ego: {ego_agent} and exo {exo_agent} collides")
                    #assert False
                    collided_exo_ids.add(exo_agent['id'])
                    #print(f"Collision at time step {time_step} with exo car {exo_agent} and ego car {ego_agent}")
                    break

    # Calculate the collision rate
    collision_rate = collision_count / total_time_steps
    return collision_rate


# ---------------- Safety 2 - Near-collision rate ------------------------

def better_is_near_miss(ego, exo, buffer=0.5):
    ego_x, ego_y = ego['pos'][0], ego['pos'][1]
    exo_x, exo_y = exo['pos'][0], exo['pos'][1]

    ego_width, ego_length = ego['bb'][0], ego['bb'][1]
    exo_width, exo_length = exo['bb'][0], exo['bb'][1]

    ego_yaw, exo_yaw = ego['heading'], exo['heading']

    # Compute the distance between the agents
    distance = math.sqrt((ego_x - exo_x)**2 + (ego_y - exo_y)**2)

    # Calculate the diagonal length of the bounding boxes with added buffer
    ego_diag = math.sqrt((ego_length + buffer)**2 + (ego_width + buffer)**2)
    exo_diag = math.sqrt((exo_length + buffer)**2 + (exo_width + buffer)**2)

    # Check if the agents are close enough to be considered a near miss
    if distance > (ego_diag + exo_diag) / 2:
        return False

    # Calculate the relative heading angle between the agents (in radians)
    relative_yaw = abs(ego_yaw - exo_yaw) % (2 * math.pi)
    if relative_yaw > math.pi:
        relative_yaw = 2 * math.pi - relative_yaw

    # If the agents are heading in roughly the same direction, they are not considered a near miss
    # Using 30 degrees (pi/6 radians) as the threshold
    if relative_yaw < math.pi / 6:
        return False

    return True

def check_near_miss(ego, exo, near_miss_threshold):
    ego_corners = get_corners(ego)
    exo_corners = get_corners(exo)
    
    ego_polygon = Polygon(ego_corners)
    exo_polygon = Polygon(exo_corners)

    # Calculate the buffered ego polygon with the near_miss_threshold
    buffered_ego_polygon = ego_polygon.buffer(near_miss_threshold)

    return buffered_ego_polygon.intersects(exo_polygon)

def find_near_miss_rate(ego_dict, exos_dict):
    near_miss_count = 0
    total_time_steps = len(ego_dict)
    collided_exo_ids = set()
    near_miss_threshold = 0.5

    # Iterate through each time step in the episode
    for time_step in list(ego_dict.keys()):
        if time_step in exos_dict.keys():
            ego_agent = ego_dict[time_step]
            exo_agents = exos_dict[time_step]

            # Chck for collisions with exo cars at the current time step
            for exo_agent in exo_agents:
                if better_is_near_miss(ego_agent, exo_agent, near_miss_threshold) and exo_agent['id'] not in collided_exo_ids:
                    near_miss_count += 1
                    #print(f"t {time_step} ego: {ego_agent} and exo {exo_agent} collides")
                    #assert False
                    collided_exo_ids.add(exo_agent['id'])
                    #print(f"Miss-collide at time step {time_step} with exo car {exo_agent} and ego car {ego_agent}")
                    break

    # Calculate the collision rate
    near_miss_rate = near_miss_count / total_time_steps
    return near_miss_rate


# ---------------- Safety 3 - Time-to-collision rate ------------------------

def time_to_collision(ego, exo, reaction_time = 1.0):
    ego_vel = np.array(ego['vel'])
    exo_vel = np.array(exo['vel'])
    relative_vel = exo_vel - ego_vel
    rv_norm = np.linalg.norm(relative_vel)

    if rv_norm == 0:
        return np.inf

    ego_corners = get_corners(ego)
    exo_corners = get_corners(exo)

    min_ttc = np.inf

    for t in np.arange(0, 3, 0.5):  # Sample times from 0 to 10 seconds with 0.1-second increments
        future_ego_corners = [np.array(corner) + (t + reaction_time) * ego_vel for corner in ego_corners]
        future_exo_corners = [np.array(corner) + t * exo_vel for corner in exo_corners]

        ego_polygon = Polygon(future_ego_corners)
        exo_polygon = Polygon(future_exo_corners)

        if ego_polygon.intersects(exo_polygon):
            min_ttc = min(min_ttc, t)
            break

    return min_ttc


def find_ttc(ego_dict, exos_dict, reaction_time=1.0):
    '''
        If reaction_time = 0.0, this is ttc
        Otherwise, it is ttr(time to react)
    '''
    total_time_steps = 0
    true_min_ttc = np.inf
    mean_min_ttc = 0

    # Iterate through each time step in the episode
    for time_step in list(ego_dict.keys()):
        if time_step in exos_dict.keys():
            ego_agent = ego_dict[time_step]
            exo_agents = exos_dict[time_step]

            min_ttc = np.inf
            # Chck for collisions with exo cars at the current time step
            for exo_agent in exo_agents:
                # We only calculate the TTC if the two agents are not colliding
                if not check_collision_by_considering_headings(ego_agent, exo_agent):                    
                    ttc = time_to_collision(ego_agent, exo_agent)
                    min_ttc = min(min_ttc, ttc)

            true_min_ttc = min(true_min_ttc, min_ttc)
            if min_ttc != np.inf:
                mean_min_ttc += min_ttc
                total_time_steps += 1
    
    mean_min_ttc /= total_time_steps

    # Calculate the collision rate
    return true_min_ttc, mean_min_ttc


#### ----------- Final Safety, Combining mine with Haoran ----------

def find_safety(ego_dict, exos_dict):
    
    near_threshold = 1.0
    near_distance = 2.0
    
    collision_count = 0 # For collision
    near_collision_count = 0 # For collision
    near_distance_count = 0 # For collision

    true_min_ttc = np.inf # For ttc
    mean_min_ttc = 0 # For ttc
    true_min_ttr = np.inf # For ttr
    mean_min_ttr = 0 # For ttr

    total_time_steps = 0

    # Iterate through each time step in the episode
    for time_step in list(ego_dict.keys()):
        # We are not sure time_step is in exos_dict so we check
        if time_step in exos_dict.keys():
            ego_agent = ego_dict[time_step]
            exo_agents = exos_dict[time_step]

            min_ttc = np.inf
            min_ttr = np.inf

            # Check for collisions with exo cars at the current time step
            isCollision = False
            for exo_agent in exo_agents:
                if check_collision_by_considering_headings(ego_agent, exo_agent, buffer=-0.5):
                    collision_count += 1
                    isCollision = True
                if check_collision_by_considering_headings(ego_agent, exo_agent, buffer=near_threshold):
                    near_collision_count += 1
                if check_collision_by_distance(ego_agent, exo_agent):
                    near_distance_count += 1

                if isCollision:
                    break

                ttc = time_to_collision(ego_agent, exo_agent)
                min_ttc = min(min_ttc, ttc)
                ttr = time_to_collision(ego_agent, exo_agent, 1.0)
                min_ttr = min(min_ttr, ttr)

            true_min_ttc = min(true_min_ttc, min_ttc)
            true_min_ttr = min(true_min_ttr, min_ttr)

            if min_ttc != np.inf:
                mean_min_ttc += min_ttc

            if min_ttr != np.inf:
                mean_min_ttr += min_ttr
            
            total_time_steps += 1

            if isCollision:
                break

    # Calculate the collision rate
    collision_rate = collision_count / total_time_steps
    near_collision_rate = near_collision_count / total_time_steps
    near_distance_rate = near_distance_count / total_time_steps
    mean_min_ttc /= total_time_steps
    mean_min_ttr /= total_time_steps

    return collision_rate, near_collision_rate, near_distance_rate, \
        mean_min_ttc, mean_min_ttr