import matplotlib.pyplot as plt
import os
import numpy as np
import fnmatch
import sys
import math
import matplotlib as mpl
import matplotlib.lines as mlines

from driving_performance_safety import find_collision_rate, find_near_miss_rate, find_ttc, find_safety
from driving_performance_comfort import find_acceleration_and_jerk
from driving_performance_efficiency import efficiency_time_traveled, average_speed, path_tracking_error, distance_traveled
from temporal_consistency_calculation import calculate_consistency
from prediction_performance import ade_fde

# This function parse logging of DESPOT and return a dictionary of data
def parse_data(txt_file):
    action_list = {}
    ego_list = {}
    exos_list = {}
    coll_bool_list = {}
    ego_path_list = {}
    pred_car_list = {}
    pred_exo_list = {}
    trial_list = {}
    depth_list = {}
    expanded_nodes = {}
    total_nodes = {}
    despot_prediction_time = []
    gamma_prediction_time = []

    number_of_predictions = {}
    time_per_call = []

    exo_count = 0
    start_recording = False
    start_recording_prediction = False

    with open(txt_file, 'r') as f:
        for line in f:
            if 'Round 0 Step' in line:
                line_split = line.split('Round 0 Step ', 1)[1]
                cur_step = int(line_split.split('-', 1)[0])
                start_recording = True
                start_recording_prediction = True
                no_of_predictions_per_timestep = 0
            if not start_recording:
                continue
            try:
                if "car pos / heading / vel" in line:  # ego_car info
                    speed = float(line.split(' ')[12])
                    heading = float(line.split(' ')[10])
                    pos_x = float(line.split(' ')[7].replace('(', '').replace(',', ''))
                    pos_y = float(line.split(' ')[8].replace(')', '').replace(',', ''))
                    bb_x = float(line.split(' ')[15])
                    bb_y = float(line.split(' ')[16])

                    pos = [pos_x, pos_y]

                    agent_dict = {'pos': [pos_x, pos_y],
                                  'heading': heading,
                                  'speed': speed,
                                  'vel': (speed * math.cos(heading), speed * math.sin(heading)),
                                  'bb': (bb_x, bb_y)
                                  }
                    ego_list[cur_step] = agent_dict
                elif " pedestrians" in line:  # exo_car info start
                    exo_count = int(line.split(' ')[0])
                    exos_list[cur_step] = []
                elif "id / pos / speed / vel / intention / dist2car / infront" in line:  # exo line, info start from index 16
                    # agent 0: id / pos / speed / vel / intention / dist2car / infront =  54288 / (99.732, 462.65) / 1 / (-1.8831, 3.3379) / -1 / 9.4447 / 0 (mode) 1 (type) 0 (bb) 0.90993 2.1039 (cross) 1 (heading) 2.0874
                    line_split = line.split(' ')
                    agent_id = int(line_split[16 + 1])

                    pos_x = float(line_split[18 + 1].replace('(', '').replace(',', ''))
                    pos_y = float(line_split[19 + 1].replace(')', '').replace(',', ''))
                    pos = [pos_x, pos_y]
                    vel_x = float(line_split[23 + 1].replace('(', '').replace(',', ''))
                    vel_y = float(line_split[24 + 1].replace(')', '').replace(',', ''))
                    vel = [vel_x, vel_y]
                    bb_x = float(line_split[36 + 1])
                    bb_y = float(line_split[37 + 1])
                    heading = float(line_split[41 + 1])
                    agent_dict = {'id': agent_id,
                                  'pos': [pos_x, pos_y],
                                  'heading': heading,
                                  'vel': [vel_x, vel_y],
                                  'bb': (bb_x , bb_y)
                                  }

                    exos_list[cur_step].append(agent_dict)
                    assert (len(exos_list[cur_step]) <= exo_count)
                elif "Path: " in line:  # path info
                    # Path: 95.166 470.81 95.141 470.86 ...
                    line_split = line.split(' ')
                    path = []
                    for i in range(1, len(line_split) - 1, 2):
                        x = float(line_split[i])
                        y = float(line_split[i + 1])
                        path.append([x, y])
                    ego_path_list[cur_step] = path
                elif 'predicted_car_' in line and start_recording_prediction:
                    # predicted_car_0 378.632 470.888 5.541
                    # (x, y, heading in rad)
                    line_split = line.split(' ')
                    pred_step = int(line_split[0][14:])
                    x = float(line_split[1])
                    y = float(line_split[2])
                    heading = float(line_split[3])
                    agent_dict = {'pos': [x, y],
                                  'heading': heading,
                                  'bb': (10.0, 10.0)
                                  }
                    if pred_step == 0:
                        pred_car_list[cur_step] = []
                    pred_car_list[cur_step].append(agent_dict)

                elif 'predicted_agents_' in line and start_recording_prediction:
                    # predicted_agents_0 380.443 474.335 5.5686 0.383117 1.1751
                    # [(x, y, heading, bb_x, bb_y)]
                    line_split = line.split(' ')
                    if line_split[-1] == "" or line_split[-1] == "\n":
                        line_split = line_split[:-1]
                    pred_step = int(line_split[0][17:])
                    if pred_step == 0:
                        pred_exo_list[cur_step] = []
                    num_agents = (len(line_split) - 1) / 5
                    agent_list = []
                    for i in range(int(num_agents)):
                        start = 1 + i * 5
                        x = float(line_split[start])
                        y = float(line_split[start + 1])
                        heading = float(line_split[start + 2])
                        bb_x = float(line_split[start + 3])
                        bb_y = float(line_split[start + 4])
                        agent_dict = {'pos': [x, y],
                                      'heading': heading,
                                      'bb': (bb_x, bb_y)
                                      }
                        agent_list.append(agent_dict)
                    pred_exo_list[cur_step].append(agent_list)
                elif 'INFO: Executing action' in line:
                    line_split = line.split(' ')
                    steer = float(line_split[5].split('/')[0])
                    acc = float(line_split[5].split('/')[1])
                    speed = float(line_split[5].split('/')[2])
                    action_list[cur_step] = (steer, acc, speed)
                    # INFO: Executing action:22 steer/acc = 0/3
                elif "Trials: no. / max length" in line:
                    line_split = line.split(' ')
                    trial = int(line_split[6])
                    depth = int(line_split[8])
                    trial_list[cur_step] = trial
                    depth_list[cur_step] = depth
                if 'collision = 1' in line or 'INININ' in line or 'in real collision' in line:
                    coll_bool_list[cur_step] = 1

                if "# nodes: expanded" in line:
                    expanded, total, policy = line.split("=")[-1].split("/")
                    expanded_nodes[cur_step] = int(expanded)
                    total_nodes[cur_step] = int(total)

                if "[PredictAgents] Reach terminal state" in line: # if this is despot
                    pred_car_list.pop(cur_step, None)
                    pred_exo_list.pop(cur_step, None)
                    start_recording_prediction = False

                if "Time for step: actual" in line:
                    despot_prediction_time.append(float(line.split('=')[-1].strip().split('/')[0]))

                if "Prediction status:" in line: # If this is gamma
                    if "Success" not in line:
                        assert False, "This file is GAMMA prediction and it fail in prediction"
                if "Time taken" in line and "GAMMA" in line:
                    gamma_prediction_time.append(float(line.split(':')[-1]))

                if "Error in prediction" in line: # if this is gamma
                    assert False, "This file is in gamma prediction and it fail in prediction"

                if "ContextPomdp::Step 123" in line:
                    no_of_predictions_per_timestep += 1
                    number_of_predictions[cur_step] = no_of_predictions_per_timestep
                if "All MopedPred Time" in line:
                    time_per_call.append(float(line.split("Time:")[-1].strip().split("agents_length")[0]))
            
            except Exception as e:
                error_handler(e)
                assert False
                #pdb.set_trace()

    return action_list, ego_list, ego_path_list, exos_list, coll_bool_list, \
        pred_car_list, pred_exo_list, trial_list, depth_list, expanded_nodes, total_nodes, gamma_prediction_time, \
        despot_prediction_time, number_of_predictions, time_per_call


def error_handler(e):
    print(
        'Error on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename, sys.exc_info()[-1].tb_lineno),
        type(e).__name__, e)


def get_dynamic_ade(ABSOLUTE_DIR):
    #ABSOLUTE_DIR = '/home/phong/driving_data/official/same_computation/lstmdefault_1Hz/'
    '''
        Recorded data map, key is method, value is another dict, whose key is file, and values is recorded data
    '''

    prediction_performance = {
        'ade': [],
        'ade_predlen3': [],
        'ade_predlen10': [],
        'ade_predlen30': [],
        'ade_obs20': [],
        'ade_obs20_predlen3': [],
        'ade_obs20_predlen10': [],
        'ade_obs20_predlen30': [],
        'ade_closest': [],
        'ade_closest_predlen3': [],
        'ade_closest_predlen10': [],
        'ade_closest_predlen30': [],
        'ade_20meters_closest': [],
        'ade_20meters_closest_predlen3': [],
        'ade_20meters_closest_predlen10': [],
        'ade_20meters_closest_predlen30': [],
        'ade_obs20_closest_predlen30': [],
        'ade_obs20_20meters_closest_predlen30': [],

        'fde': [],
        'fde_predlen3': [],
        'fde_predlen10': [],
        'fde_predlen30': [],
        'fde_obs20': [],
        'fde_obs20_predlen3': [],
        'fde_obs20_predlen10': [],
        'fde_obs20_predlen30': [],
        'fde_closest': [],
        'fde_closest_predlen3': [],
        'fde_closest_predlen10': [],
        'fde_closest_predlen30': [],
        'fde_20meters_closest': [],
        'fde_20meters_closest_predlen3': [],
        'fde_20meters_closest_predlen10': [],
        'fde_20meters_closest_predlen30': [],
        'fde_obs20_closest_predlen30': [],
        'fde_obs20_20meters_closest_predlen30': [],

        'std': [],
        'std_predlen30': [],
        'std_closest': [],
        'std_closest_predlen30': [],
        'std_20meters_closest': [],
        'std_20meters_closest_predlen30': [],
        
        'temp_consistency3': [],
        'temp_consistency_closest3': [],
        'temp_consistency5': [],
        'temp_consistency_closest5': [],
        'temp_consistency15': [],
        'temp_consistency_closest15': [],
        'temp_consistency30': [],
        'temp_consistency_closest30': [],
    }
    driving_performance = {
        'safety': {
            'collision_rate': [],
            'near_miss_rate': [],
            'near_distance_rate': [],
            'mean_min_ttc': [],
            'mean_min_ttr': [],
        },
        'comfort': {
            'jerk': [],
            'lateral_acceleration': [],
            'acceleration': []
        },
        'efficiency': {
            'avg_speed': [],
            'tracking_error': [],
            'efficiency_time': [],
            'distance_traveled': [],
        },
    }
    tree_performance = {
        'expanded_nodes': [],
        'total_nodes': [],
        'trial': [],
        'depth': [],
        'gamma_time': [], # available only for gamma
        'despot_time': [], # available only for despot
        'number_valid_files': 0,
        'number_timesteps': [],
        'number_mopeds': [],# available only for despot
        "time_per_mopeds": []# available only for despot
    }

    for root, subdirs, files in os.walk(ABSOLUTE_DIR):
        if len(files) > 0:
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)

                    print(f"Processing {file_path}")
                    try:
                        action_list, ego_list, ego_path_list, exos_list, coll_bool_list, \
                        pred_car_list, pred_exo_list, trial_list, depth_list, expanded_nodes,\
                         total_nodes, gamma_time, despot_time, despot_no_mopeds, time_per_moped = parse_data(file_path)
                        print("Done no error for above file")
                    except Exception as e:
                        print(f"Exception {e}")
                        continue
                    
                    # The number of steps are too small which can affect ADE/FDE, thus we ignore these files
                    if len(ego_list) <= 10:
                        print(f'Length of step is less than 10. Skip this record {file_path}')
                        continue

                    # Tree performance
                    tree_performance['expanded_nodes'].append(list(expanded_nodes.values()))
                    tree_performance['total_nodes'].append(list(total_nodes.values()))
                    tree_performance['trial'].append(list(trial_list.values()))
                    tree_performance['depth'].append(list(depth_list.values()))
                    tree_performance['number_mopeds'].append(list(despot_no_mopeds.values()))
                    tree_performance['gamma_time'].append(gamma_time)
                    tree_performance['despot_time'].append(despot_time)
                    tree_performance['number_valid_files'] += 1
                    tree_performance['number_timesteps'].append(len(ego_list))
                    tree_performance['time_per_mopeds'].append(np.average(time_per_moped))

                    if 'despot_planner' in file_path:
                        flattened_list = [value for sublist in tree_performance['expanded_nodes'] for value in sublist]
                        print(f"Expanded nodes: {np.average(list(expanded_nodes.values())):.2f}/{np.average(flattened_list):.2f}", end=" ")
                        flattened_list = [value for sublist in tree_performance['total_nodes'] for value in sublist]
                        print(f"Total nodes: {np.average(list(total_nodes.values())):.2f}/{np.average(flattened_list):.2f}", end=" ")
                        flattened_list = [value for sublist in tree_performance['trial'] for value in sublist]
                        print(f"Trials: {np.average(list(trial_list.values())):.2f}/{np.average(flattened_list):.2f}", end=" ")
                        flattened_list = [value for sublist in tree_performance['depth'] for value in sublist]
                        print(f"Depth: {np.average(list(depth_list.values())):.2f}/{np.average(flattened_list):.2f}", end=" ")
                        flattened_list = [value for sublist in tree_performance['number_mopeds'] for value in sublist]
                        print(f"NoMopeds: {np.average(list(despot_no_mopeds.values())):.2f}/{np.average(flattened_list):.2f}", end=" ")
                        #flattened_list = [value for sublist in tree_performance['time_per_mopeds'] for value in sublist]
                        print(f"TimPerMoped: {np.average(time_per_moped):.5f}/{np.average(tree_performance['time_per_mopeds']):.5f}", end=" ")
                        flattened_list = [value for sublist in tree_performance['despot_time'] for value in sublist]
                        print(f"TimePerAction: {np.average(despot_time):.2f}/{np.average(flattened_list):.2f}")

                    elif 'gamma_planner' in file_path:
                        flattened_list = [value for sublist in tree_performance['gamma_time'] for value in sublist]
                        print(f"Gamma time: {np.average(gamma_time):.2f} / {np.average(flattened_list):.2f}")

                    #continue

                    # Prediction performance
                    try:
                         exo_ade_mean,\
                            exo_ade_predlen3_mean ,\
                            exo_ade_predlen10_mean ,\
                            exo_ade_predlen30_mean,\
                            exo_ade_obs20_mean,\
                            exo_ade_obs20_predlen3_mean,\
                            exo_ade_obs20_predlen10_mean ,\
                            exo_ade_obs20_predlen30_mean ,\
                            exo_ade_closest_mean ,\
                            exo_ade_closest_predlen3_mean,\
                            exo_ade_closest_predlen10_mean ,\
                            exo_ade_closest_predlen30_mean ,\
                            exo_ade_20meters_closest_mean ,\
                            exo_ade_20meters_closest_predlen3_mean ,\
                            exo_ade_20meters_closest_predlen10_mean,\
                            exo_ade_20meters_closest_predlen30_mean ,\
                            exo_ade_obs20_closest_predlen30_mean, \
                            exo_ade_obs20_20meters_closest_predlen30_mean, \
                            exo_fde_mean,\
                            exo_fde_predlen3_mean ,\
                            exo_fde_predlen10_mean,\
                            exo_fde_predlen30_mean,\
                            exo_fde_obs20_mean ,\
                            exo_fde_obs20_predlen3_mean ,\
                            exo_fde_obs20_predlen10_mean ,\
                            exo_fde_obs20_predlen30_mean ,\
                            exo_fde_closest_mean ,\
                            exo_fde_closest_predlen3_mean ,\
                            exo_fde_closest_predlen10_mean ,\
                            exo_fde_closest_predlen30_mean ,\
                            exo_fde_20meters_closest_mean ,\
                            exo_fde_20meters_closest_predlen3_mean ,\
                            exo_fde_20meters_closest_predlen10_mean ,\
                            exo_fde_20meters_closest_predlen30_mean ,\
                            exo_fde_obs20_closest_predlen30_mean, \
                            exo_fde_obs20_20meters_closest_predlen30_mean, \
                            exo_ade,\
                            exo_ade_predlen30, \
                            exo_ade_closest,\
                            exo_ade_closest_predlen30,\
                            exo_ade_20meters_closest,\
                            exo_ade_20meters_closest_predlen30 = ade_fde(pred_car_list, pred_exo_list, ego_list, exos_list)
                        #if exo_ade is None:
                        #    continue
                    except Exception as e:
                        print(e)
                        #continue
                        assert False

                    tmp_consistency3, temp_consistency_closest3 = calculate_consistency(exos_list, pred_exo_list, 3)
                    tmp_consistency5, temp_consistency_closest5 = calculate_consistency(exos_list, pred_exo_list, 5)
                    tmp_consistency15, temp_consistency_closest15 = calculate_consistency(exos_list, pred_exo_list, 15)
                    tmp_consistency30, temp_consistency_closest30 = calculate_consistency(exos_list, pred_exo_list, 30)

                    prediction_performance['ade'].append(exo_ade_mean)
                    prediction_performance['ade_predlen3'].append(exo_ade_predlen3_mean)
                    prediction_performance['ade_predlen10'].append(exo_ade_predlen10_mean)
                    prediction_performance['ade_predlen30'].append(exo_ade_predlen30_mean)
                    prediction_performance['ade_obs20'].append(exo_ade_obs20_mean)
                    prediction_performance['ade_obs20_predlen3'].append(exo_ade_obs20_predlen3_mean)
                    prediction_performance['ade_obs20_predlen10'].append(exo_ade_obs20_predlen10_mean)
                    prediction_performance['ade_obs20_predlen30'].append(exo_ade_obs20_predlen30_mean)
                    prediction_performance['ade_closest'].append(exo_ade_closest_mean)
                    prediction_performance['ade_closest_predlen3'].append(exo_ade_closest_predlen3_mean)
                    prediction_performance['ade_closest_predlen10'].append(exo_ade_closest_predlen10_mean)
                    prediction_performance['ade_closest_predlen30'].append(exo_ade_closest_predlen30_mean)
                    prediction_performance['ade_20meters_closest'].append(exo_ade_20meters_closest_mean)
                    prediction_performance['ade_20meters_closest_predlen3'].append(exo_ade_20meters_closest_predlen3_mean)
                    prediction_performance['ade_20meters_closest_predlen10'].append(exo_ade_20meters_closest_predlen10_mean)
                    prediction_performance['ade_20meters_closest_predlen30'].append(exo_ade_20meters_closest_predlen30_mean)
                    prediction_performance['ade_obs20_closest_predlen30'].append(exo_ade_obs20_closest_predlen30_mean)
                    prediction_performance['ade_obs20_20meters_closest_predlen30'].append(exo_ade_obs20_20meters_closest_predlen30_mean)

                    prediction_performance['fde'].append(exo_fde_mean)
                    prediction_performance['fde_predlen3'].append(exo_fde_predlen3_mean)
                    prediction_performance['fde_predlen10'].append(exo_fde_predlen10_mean)
                    prediction_performance['fde_predlen30'].append(exo_fde_predlen30_mean)
                    prediction_performance['fde_obs20'].append(exo_fde_obs20_mean)
                    prediction_performance['fde_obs20_predlen3'].append(exo_fde_obs20_predlen3_mean)
                    prediction_performance['fde_obs20_predlen10'].append(exo_fde_obs20_predlen10_mean)
                    prediction_performance['fde_obs20_predlen30'].append(exo_fde_obs20_predlen30_mean)
                    prediction_performance['fde_closest'].append(exo_fde_closest_mean)
                    prediction_performance['fde_closest_predlen3'].append(exo_fde_closest_predlen3_mean)
                    prediction_performance['fde_closest_predlen10'].append(exo_fde_closest_predlen10_mean)
                    prediction_performance['fde_closest_predlen30'].append(exo_fde_closest_predlen30_mean)
                    prediction_performance['fde_20meters_closest'].append(exo_fde_20meters_closest_mean)
                    prediction_performance['fde_20meters_closest_predlen3'].append(exo_fde_20meters_closest_predlen3_mean)
                    prediction_performance['fde_20meters_closest_predlen10'].append(exo_fde_20meters_closest_predlen10_mean)
                    prediction_performance['fde_20meters_closest_predlen30'].append(exo_fde_20meters_closest_predlen30_mean)
                    prediction_performance['fde_obs20_closest_predlen30'].append(exo_fde_obs20_closest_predlen30_mean)
                    prediction_performance['fde_obs20_20meters_closest_predlen30'].append(exo_fde_obs20_20meters_closest_predlen30_mean)

                    prediction_performance['std'].append(np.std(exo_ade))
                    prediction_performance['std_predlen30'].append(np.std(exo_ade_predlen30))
                    prediction_performance['std_closest'].append(np.std(exo_ade_closest))
                    prediction_performance['std_closest_predlen30'].append(np.std(exo_ade_closest_predlen30))
                    prediction_performance['std_20meters_closest'].append(np.std(exo_ade_20meters_closest))
                    prediction_performance['std_20meters_closest_predlen30'].append(np.std(exo_ade_20meters_closest_predlen30))

                    prediction_performance['temp_consistency3'].append(tmp_consistency3)
                    prediction_performance['temp_consistency_closest3'].append(temp_consistency_closest3)
                    prediction_performance['temp_consistency5'].append(tmp_consistency5)
                    prediction_performance['temp_consistency_closest5'].append(temp_consistency_closest5)
                    prediction_performance['temp_consistency15'].append(tmp_consistency15)
                    prediction_performance['temp_consistency_closest15'].append(temp_consistency_closest15)
                    prediction_performance['temp_consistency30'].append(tmp_consistency30)
                    prediction_performance['temp_consistency_closest30'].append(temp_consistency_closest30)
                
                    print(f"Dynamic ADE {exo_ade_mean:.2f}, Dynamic ADE OBS20  {exo_ade_obs20_mean:.2f}, Dynamic ADE CLOSEST {exo_ade_closest_mean:.2f}", end=" ")
                    
                    # Driving performance - safety
                    collision_rate, near_miss_rate, near_distance_rate, mean_min_ttc, mean_min_ttr = find_safety(ego_list, exos_list)
                    driving_performance['safety']['collision_rate'].append(collision_rate)
                    driving_performance['safety']['near_miss_rate'].append(near_miss_rate)
                    driving_performance['safety']['near_distance_rate'].append(near_distance_rate)
                    driving_performance['safety']['mean_min_ttc'].append(mean_min_ttc)
                    driving_performance['safety']['mean_min_ttr'].append(mean_min_ttr)

                    print(f"coll-rate {collision_rate:.2f}, miss-rate {near_miss_rate:.2f} ttc {mean_min_ttr:.2f}", end=" ")

                    # Driving performance - comfort
                    jerk, lateral_acceleration, acceleration = find_acceleration_and_jerk(ego_list)
                    driving_performance['comfort']['jerk'].append(jerk)
                    driving_performance['comfort']['lateral_acceleration'].append(lateral_acceleration)
                    driving_performance['comfort']['acceleration'].append(acceleration)

                    print(f"jerk {jerk:.2f}, lat-acc {lateral_acceleration:.2f}", end=" ")

                    # Driving performance - efficiency
                    avg_speed = average_speed(ego_list)
                    tracking_error = path_tracking_error(ego_list, ego_path_list)
                    efficiency_time = efficiency_time_traveled(ego_list, ego_path_list)
                    distance_travel = distance_traveled(ego_list)
                    driving_performance['efficiency']['avg_speed'].append(avg_speed)
                    driving_performance['efficiency']['tracking_error'].append(tracking_error)
                    driving_performance['efficiency']['efficiency_time'].append(efficiency_time)
                    driving_performance['efficiency']['distance_traveled'].append(distance_travel)

                    print(f"avg_speed {avg_speed:.2f}, track-err {tracking_error:.2f} eff-time {efficiency_time:.2f}")
    

    return prediction_performance, driving_performance, tree_performance
