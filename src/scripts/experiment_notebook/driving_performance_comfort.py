import numpy as np

def calculate_acceleration(velocity_data, delta_time):
    return np.diff(velocity_data, axis=0) / delta_time

def calculate_jerk(accel_data, delta_time):
    jerk = np.diff(accel_data, axis=0) / delta_time
    jerk = np.linalg.norm(jerk, axis=1)
    return jerk

def calculate_lateral_acceleration(speed_data, heading_data, delta_time):
    delta_speed = np.diff(speed_data)
    delta_heading = np.diff(heading_data)
    lateral_acceleration = delta_speed * np.sin(delta_heading) / delta_time
    return lateral_acceleration

### ----------- Comfort 1 - Acceleration and jerk -------------------------- ###

def find_acceleration_and_jerk(ego_dict):

    # Extract velocity, speed, and heading data from ego_list
    velocity_data = np.array([ego_dict[t]["vel"] for t in sorted(ego_dict.keys())]) # [time_step, 2]
    speed_data = np.array([ego_dict[t]["speed"] for t in sorted(ego_dict.keys())]) # [time_step, 1]
    heading_data = np.array([ego_dict[t]["heading"] for t in sorted(ego_dict.keys())]) # [time_step, 1]

    delta_time = 0.3
    # Calculate acceleration, jerk, and lateral acceleration
    accel_data = calculate_acceleration(velocity_data, delta_time)
    jerk_data = calculate_jerk(accel_data, delta_time)
    lateral_accel_data = calculate_lateral_acceleration(speed_data, heading_data, delta_time)

    mean_lat_accel = np.mean(np.sqrt(lateral_accel_data**2)) # (n,1) for lateral_acce
    mean_accel = np.mean(np.sqrt(np.sum(accel_data**2, axis=1))) # (n,2) for accel_data
    #assert np.isclose(np.mean(np.sqrt(np.sum(accel_data**2, axis=1))), np.mean(np.linalg.norm(accel_data, axis=1)))
    mean_jerk = np.mean(jerk_data)

    print(f"{np.mean(np.sqrt(np.sum(accel_data**2, axis=1)))} and {np.mean(np.sqrt(accel_data**2))}")

    return mean_jerk, mean_lat_accel, mean_accel

if __name__ == "__main__":
    ego_dict = {0: {'vel': [1,2], 'speed': 2.0, 'heading': 1.45},
                1: {'vel': [3,4], 'speed': 3.0, 'heading': 2.43},
                2: {'vel': [2,3], 'speed': 1.2, 'heading': 3.33}}
    find_acceleration_and_jerk(ego_dict)