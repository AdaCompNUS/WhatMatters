import numpy as np

#Okay gpt, given ego_list # dictionary[timestep] = {pos: [x,y], heading: 
# float, speed: float, vel: [vek_x, vel_y], bb: [bb_x, bby]}, 
# which is a list of x,y, heading, speed, vel and bb at each timestep, and path, 
# which is dictionary[timestep] = [(x,y),...], which is the path of desired trajectory of each time step, 
#please help me to write the code to calculate the efficiency that you can get from these info


### ----------- Efficiency 1  distance travelled and efficient time --------------- ###

import numpy as np

def distance_traveled(ego_list):
    total_distance = 0

    for timestep in list(ego_list.keys()):
        if (timestep+1) in ego_list.keys():
            ego_position = np.array(ego_list[timestep]['pos'])
            next_ego_position = np.array(ego_list[timestep + 1]['pos'])
            total_distance += np.linalg.norm(next_ego_position - ego_position)

    return total_distance

def actual_travel_time(ego_list):
    total_time = 0

    for timestep in list(ego_list.keys()):
        if (timestep+1) in ego_list.keys():
            pos1 = np.array(ego_list[timestep]['pos'])
            pos2 = np.array(ego_list[timestep + 1]['pos'])
            speed = ego_list[timestep]['speed']
            if speed > 1e-3:
                distance = np.linalg.norm(pos2 - pos1)
                time = distance / speed
                if time > 100:
                    print(f"Something is wrong with the time calculation d {distance} s {speed}")
                    continue # Do not add this time to the total
                total_time += time

    return total_time

# Below 2 functions are complicated one for calculating expected travel time
def closest_path_index(ego_pos, path):
    min_distance = float('inf')
    closest_index = -1

    for i, pos in enumerate(path):
        distance = np.linalg.norm(np.array(ego_pos) - np.array(pos))
        if distance < min_distance:
            min_distance = distance
            closest_index = i

    return closest_index

def expected_travel_time(ego_list, path, desired_speed):
    total_time = 0

    for timestep in list(ego_list.keys()):
        if (timestep+1) in ego_list.keys():
            ego_pos = ego_list[timestep]['pos']
            next_ego_pos = ego_list[timestep + 1]['pos']

            if timestep in path:
                path_at_timestep = path[timestep]

                # Find the closest path index for the current and next ego positions
                closest_index = closest_path_index(ego_pos, path_at_timestep)
                next_closest_index = closest_path_index(next_ego_pos, path_at_timestep)

                # Calculate the distance between the closest points in the path
                if next_closest_index > closest_index:
                    distance = 0
                    for i in range(closest_index, next_closest_index):
                        pos1 = np.array(path_at_timestep[i])
                        pos2 = np.array(path_at_timestep[i + 1])
                        distance += np.linalg.norm(pos2 - pos1)

                    # Calculate the time based on the distance and desired_speed
                    time = distance / desired_speed
                    total_time += time

    return total_time

# The lower the better because it means the car is driving faster
def efficiency_time_traveled(ego_list, path, desired_speed=6.0):
    
    #actual_time = actual_travel_time(ego_list)
    dist = distance_traveled(ego_list)
    avg_speed = average_speed(ego_list)
    avg_time = dist / avg_speed
    expected_time = dist / desired_speed
    # A simple expected time by assuming each time step is 0.3 seconds
    #expected_time = len(ego_list) * 0.3
    # A more complicated expected time
    #expected_time2 = expected_travel_time(ego_list, path, desired_speed)
    #print(f"Actual time: {actual_time}, Expected time: {expected_time}")

    # Old formular
    #time_efficiency = actual_time / expected_time
    # New formula:
    time_efficiency = avg_time / expected_time

    return time_efficiency

### ----------- Efficiency 2  Speed --------------- ###

# The higher the better
def average_speed(ego_list):
    speeds = [ego_data['speed'] for timestep, ego_data in ego_list.items()]
    return np.mean(np.abs(speeds))


### ----------- Efficiency 3 Tracking error --------------- ###


# The lower the better
def path_tracking_error(ego_list, path):
    error = []

    for timestep in list(ego_list.keys()):
        ego_position = np.array(ego_list[timestep]['pos'])

        if timestep in path:
            desired_positions = np.array(path[timestep])
            # We should use index 0 or 1 but actually I do not know
            # 0 or 1 is better so just use min but bigger than 0
            # I think the next 5 more steps are important so we filter next 10 steps
            if len(desired_positions) > 5:
                desired_positions = desired_positions[:5]
            distance_to_desired = np.linalg.norm(ego_position - desired_positions, axis=1)
            non_zero_distances = distance_to_desired[distance_to_desired > 0]

            if len(non_zero_distances) > 0:
                min_distance = np.min(non_zero_distances)
                error.append(min_distance)

    return np.mean(error)


