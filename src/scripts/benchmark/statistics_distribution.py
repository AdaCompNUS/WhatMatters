import matplotlib.pyplot as plt

import argparse
import pdb
import matplotlib.path as mpath
import random
import math
import matplotlib
import numpy as np
import sys



def error_handler(e):
    print(
        'Error on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename, sys.exc_info()[-1].tb_lineno),
        type(e).__name__, e)


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

    exo_count = 0
    start_recording = False

    with open(txt_file, 'r') as f:
        for line in f:
            if 'Round 0 Step' in line:
                line_split = line.split('Round 0 Step ', 1)[1]
                cur_step = int(line_split.split('-', 1)[0])
                start_recording = True
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
                                  'bb': (bb_x * 2, bb_y * 2)
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
                elif 'predicted_car_' in line:
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

                elif 'predicted_agents_' in line:
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
                                      'bb': (bb_x * 2, bb_y * 2)
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

            except Exception as e:
                error_handler(e)
                pdb.set_trace()

    print('done')

    return action_list, ego_list, ego_path_list, exos_list, coll_bool_list, pred_car_list, pred_exo_list, trial_list, depth_list


if __name__ == "__main__":
    from average_statistics_of_1_data import filter_txt_files, collect_txt_files


    FOLDER_PATHS = [

        '/home/cunjun/driving_data/DEL/hivt1hz_t0_03_slowdownHz30times/',
        '/home/cunjun/driving_data/DEL/lanegcn1hz_t0_03_slowdownHz30times/',
        '/home/cunjun/driving_data/DEL/knndefault_3Hz_ts0_1_allHzscale/',
        '/home/cunjun/driving_data/DEL/knnsocial1hz_t0_03_slowdownHz15times_2/',
        '/home/cunjun/driving_data/DEL/lstmdefault1hz_t0_03_slowdownHz15times_2/',
        '/home/cunjun/driving_data/DEL/lstmsocial1hz_t0_03_slowdownHz30times/',
        '/home/cunjun/driving_data/DEL/cv10hz_t0_33_slowdownHz3times/', # a bit less, slow down not too much, using time.sleep(0.00008)
        '/home/cunjun/driving_data/DEL/ca10hz_t0_33_slowdownHz3times/',# still too much
    ]

    distributions = {}

    for folder_path in FOLDER_PATHS:
        file_path_list = filter_txt_files(folder_path,
                                      collect_txt_files(folder_path, flag="", ignore_flag="nothing"))

        experiment_name = folder_path.split('/')[5]

        # Get only 1 file for now
        # pred_exo_list: keys is time step, value is 30 time steps of 20 agents -> list of list of dict
        # pred_car_list: keys is time step, value is 30 time steps of 1 agent -> list of dict
        distributions[experiment_name] = []

        for file_path in file_path_list:
            _, ego_list, _, exos_list, _, pred_car_list, pred_exo_list, _, _ = parse_data(file_path)

            # loop through time and find distribution
            if len(ego_list) >= 80:
                for t in range(len(ego_list)-30):
                    exo_pos_list = exos_list[t]
                    pred_exo_pos_list = pred_exo_list[t]

                    # find pred and ground-truth of each agent

                    for i in range(len(exo_pos_list)):
                        agent_id = exo_pos_list[i]['id']
                        # Find next 30 ground-truth of this agent
                        gt_pos_list = []
                        for next_t in range(t, t+30):
                            all_agent_ids_of_next_t = [agent['id'] for agent in exos_list[next_t]]
                            if agent_id in all_agent_ids_of_next_t:
                                gt_pos_list.append(exos_list[next_t][all_agent_ids_of_next_t.index(agent_id)]['pos'])
                        # if enough 30, then find the distance
                        if len(gt_pos_list) == 30:
                            # need to find prediction of this agent id
                            pred_pos_list = []
                            all_agent_ids_of_t = [agent['id'] for agent in exos_list[t]]
                            for next_t in range(30):
                                pred_pos_list.append(pred_exo_pos_list[next_t][all_agent_ids_of_t.index(agent_id)]['pos'])

                            # find distance
                            dist = np.mean(np.array(gt_pos_list) - np.array(pred_pos_list))
                            distributions[experiment_name].append(dist)
                #break
            else:
                continue

    fig, axs = plt.subplots(2, 4, figsize=(10,20)) # we plot 4 values so we need 2 times 2. Can be adjusted if draw more than that
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    # plot the distribution by histogram in a 4x4 subplot
    for i, key in enumerate(distributions.keys()):
        print(f"key: {key} i//4: {i//4} i%4: {i%4}")
        ax = axs[i//4, i%4]
        ax.hist(distributions[key], bins=100, color=colors[i])
        ax.set_xlim(-20, 20)

        ax.set_title(key)
        ax.set_xlabel('distance')
        ax.set_ylabel('frequency')
    # set axes limit x


    plt.show()
