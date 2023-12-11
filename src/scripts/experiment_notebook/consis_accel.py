import os, sys
import fnmatch
import argparse
import numpy as np
import math, pdb

import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import animation
import random
import copy
import torch
import torch.nn as nn
from scipy.stats import pearsonr




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
    time_list = {}

    max_min_coords = {'max_x': -math.inf, 'min_x': math.inf, 'max_y': -math.inf, 'min_y': math.inf}

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

                    max_min_coords['max_x'] = max(max_min_coords['max_x'], pos_x)
                    max_min_coords['min_x'] = min(max_min_coords['min_x'], pos_x)
                    max_min_coords['max_y'] = max(max_min_coords['max_y'], pos_y)
                    max_min_coords['min_y'] = min(max_min_coords['min_y'], pos_y)

                    agent_dict = {'pos': [pos_x, pos_y],
                                    'heading': heading,
                                    'speed': speed,
                                    'vel': (speed * math.cos(heading), speed * math.sin(heading)),
                                    'bb': (bb_x, bb_y)
                                    }
                    ego_list[cur_step] = agent_dict

                elif " pedestrians" in line: # exo_car info start
                    exo_count = int(line.split(' ')[0])
                    exos_list[cur_step] = []
                elif "id / pos / speed / vel / intention / dist2car / infront" in line: # exo line, info start from index 16
                    # agent 0: id / pos / speed / vel / intention / dist2car / infront =  54288 / (99.732, 462.65) / 1 / (-1.8831, 3.3379) / -1 / 9.4447 / 0 (mode) 1 (type) 0 (bb) 0.90993 2.1039 (cross) 1 (heading) 2.0874
                    line_split = line.split(' ')
                    agent_id = int(line_split[16+1])

                    pos_x = float(line_split[18+1].replace('(', '').replace(',', ''))
                    pos_y = float(line_split[19+1].replace(')', '').replace(',', ''))
                    pos = [pos_x, pos_y]

                    max_min_coords['max_x'] = max(max_min_coords['max_x'], pos_x)
                    max_min_coords['min_x'] = min(max_min_coords['min_x'], pos_x)
                    max_min_coords['max_y'] = max(max_min_coords['max_y'], pos_y)
                    max_min_coords['min_y'] = min(max_min_coords['min_y'], pos_y)

                    vel_x = float(line_split[23+1].replace('(', '').replace(',', ''))
                    vel_y = float(line_split[24+1].replace(')', '').replace(',', ''))
                    vel = [vel_x, vel_y]

                    bb_x = float(line_split[36+1])
                    bb_y = float(line_split[37+1])

                    heading = float(line_split[41+1])

                    agent_dict = {  'id': agent_id,
                                    'pos': [pos_x, pos_y],
                                    'heading': heading,
                                    'vel': [vel_x, vel_y],
                                    'bb': (bb_x*2, bb_y*2)
                    }

                    exos_list[cur_step].append(agent_dict)
                    assert(len(exos_list[cur_step]) <= exo_count)
                elif "Path: " in line: # path info
                    # Path: 95.166 470.81 95.141 470.86 ...
                    line_split = line.split(' ')
                    path = []
                    for i in range(1, len(line_split)-1, 2):
                        x = float(line_split[i])
                        y = float(line_split[i+1])
                        path.append([x,y])
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

                    max_min_coords['max_x'] = max(max_min_coords['max_x'], x)
                    max_min_coords['min_x'] = min(max_min_coords['min_x'], x)
                    max_min_coords['max_y'] = max(max_min_coords['max_y'], y)
                    max_min_coords['min_y'] = min(max_min_coords['min_y'], y)

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
                                    'bb': (bb_x*2, bb_y*2) 
                                    }
                        agent_list.append(agent_dict)

                        max_min_coords['max_x'] = max(max_min_coords['max_x'], x)
                        max_min_coords['min_x'] = min(max_min_coords['min_x'], x)
                        max_min_coords['max_y'] = max(max_min_coords['max_y'], y)
                        max_min_coords['min_y'] = min(max_min_coords['min_y'], y)

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
                elif "INFO: The current time is" in line:
                    line_split = line.split(' ')
                    time = float((line_split[6].split(':')[-1]).split('[')[0])
                    time_list[cur_step] = time
                if 'collision = 1' in line or 'INININ' in line or 'in real collision' in line:
                    coll_bool_list[cur_step] = 1

            except Exception as e:
                error_handler(e)
                #pdb.set_trace()
                assert False

    return action_list, ego_list, ego_path_list, exos_list, coll_bool_list, pred_car_list, pred_exo_list, trial_list, depth_list, max_min_coords, time_list


def agent_rect(agent_dict, origin, color, fill=True):
    try:
        pos = agent_dict['pos']
        heading = agent_dict['heading']
        bb_x, bb_y = agent_dict['bb']
        x_shift = [bb_y/2.0 * math.cos(heading), bb_y/2.0 * math.sin(heading)]
        y_shift = [-bb_x/2.0 * math.sin(heading), bb_x/2.0 * math.cos(heading)]
        
        coord = [pos[0] - 0 - x_shift[0] - y_shift[0], pos[1] - 0 - x_shift[1] - y_shift[1]]
        rect = mpatches.Rectangle(
            xy=coord, 
            width=bb_y , height=bb_x, angle=np.rad2deg(heading), fill=fill, color=color)
        return rect

    except Exception as e:
        error_handler(e)
        pdb.set_trace()


def vel_arrow(agent_dict, origin, color):
    try:
        vel = agent_dict['vel']
        pos = agent_dict['pos']
        arrow = mpatches.Arrow(
            x=pos[0] - 0, y=pos[1] - 0, dx=vel[0], dy=vel[1], color=color)
        return arrow

    except Exception as e:
        error_handler(e)
        pdb.set_trace()


def acc_arrow(action, ego_dict, mode):
    try:
        heading = ego_dict['heading']
        steer = action[0]
        acc = action[1]
        speed = action[2]
        # print('heading {}, steer {}, acc {}'.format(heading, steer, acc))
        if mode == 'acc':
            arrow = mpatches.Arrow(
                x=ego_dict['pos'][0], y=ego_dict['pos'][1], dx=math.cos(heading) * acc, dy=math.sin(heading) * acc, color='red', width=4)
        else:
            arrow = mpatches.Arrow(
                x=ego_dict['pos'][0], y=ego_dict['pos'][1], dx=math.cos(heading) * speed, dy=math.sin(heading) * speed, color='lightgreen', width=2)
        return arrow

    except Exception as e:
        error_handler(e)
        pdb.set_trace()


def init():
    # initialize an empty list of cirlces
    return [trial_text, depth_text]

def animate(time_step):
    patches = [trial_text, depth_text]

    time_step =  time_step + config.frame


    ego_pos = ego_list[time_step]['pos']
    print(f"Drawing time step {time_step} ego pos {ego_pos}")

    # draw ego car
    if time_step in coll_bool_list.keys():
        ego_color = 'red'
    else:
        ego_color = 'green'

    if time_step in trial_list.keys():
        trial_text.set_text("trial #: " + str(trial_list[time_step]))
        depth_text.set_text("depth: " + str(depth_list[time_step]))
    # print('ego_heading: {}'.format(ego_list[time_step]['heading']))

    # draw exo agents
    for agent_dict in exos_list[time_step]:
        patches.append(ax.add_patch(
            agent_rect(agent_dict, ego_pos, 'black')))
        patches.append(ax.add_patch(
            vel_arrow(agent_dict, ego_pos, 'grey')))

    if time_step in pred_car_list.keys():
        for car_dict in pred_car_list[time_step]:
            car_dict['bb'] = ego_list[time_step]['bb']
            patches.append(ax.add_patch(
                agent_rect(car_dict, ego_pos, 'lightgreen', False)))

    if time_step in pred_exo_list.keys():
        for agent_list in pred_exo_list[time_step]:
            for agent_dict in agent_list:
                patches.append(ax.add_patch(
                    agent_rect(agent_dict, ego_pos, 'grey', False)))

    # draw path
    # path = ego_path_list[time_step]
    # for i in range(0, len(path), 2):
    #     point = path[i]
    #     patches.append(ax.add_patch(
    #         mpatches.Circle([point[0]-ego_pos[0], point[1]-ego_pos[1]],
    #                              0.1, color='orange')))

    if time_step in ego_list.keys():
        patches.append(ax.add_patch(
            agent_rect(ego_list[time_step], ego_pos, ego_color)))

    if time_step in action_list.keys():
        patches.append(ax.add_patch(
            acc_arrow(action_list[time_step], ego_list[time_step], mode='acc')))
        patches.append(ax.add_patch(
            acc_arrow(action_list[time_step], ego_list[time_step], mode='speed')))

    if time_step in ego_list.keys():
        patches.append(ax.add_patch(
            vel_arrow(ego_list[time_step], ego_pos, 'brown')))    

    return patches

def onClick(event):
    global anim_running
    if anim_running:
        anim.event_source.stop()
        anim_running = False
    else:
        anim.event_source.start()
        anim_running = True

def progress(ego_list):
    time =[]
    pre,suc = 0,0
    for key in ego_list.keys():
        if np.abs(ego_list[key]['speed'])>2:
            pre = 1
        elif np.abs(ego_list[key]['speed'])<0.2:
            suc = 1
            if pre == 1:
                time.append(int(key))
                pre=suc=0
        else:
            pre=suc=0
    return time

def getCircles(pre,pos):
    radius = 0.7*pos['bb'][0] # radius of circles, need to change
    center1 = [pre['pos'][0]-(pos['bb'][1]-pos['bb'][0])*np.cos(pre['heading'])/2, 
               pre['pos'][1]-(pos['bb'][1]-pos['bb'][0])*np.sin(pre['heading'])/2]
    center2 = [pre['pos'][0]+(pos['bb'][1]-pos['bb'][0])*np.cos(pre['heading'])/2, 
               pre['pos'][1]+(pos['bb'][1]-pos['bb'][0])*np.sin(pre['heading'])/2]
    return np.array([center1,center2]), radius

def predTime(ego_pres, others_pres, ego_pos, others_pos):
    pred_steps = 10
    for i in range(pred_steps):
        # print("Step", i+1)
        ego_pre, others_pre = ego_pres[i],others_pres[i]
        ego_core,ego_range = getCircles(ego_pre,ego_pos)
        # print("Ego core of circle is", ego_core, 'and the radius is', ego_range)
        for other_pre,other_pos in zip(others_pre,others_pos):
            other_core,other_range = getCircles(other_pre,other_pos)
            dis = (np.expand_dims(ego_core, 0) - np.expand_dims(other_core, 1))
            dis = np.sqrt((dis**2).sum(2))
            # print("Core of circle for this car is", other_core, 'and the radius is', other_range)
            # print("The minimum distance is ", dis.min(), "less than", ego_range + other_range)
            if dis.min() < ego_range + other_range:
                return i+1
    return 'NAN'

def consistency(pred, pred_1forward, time_shifting):
    
    num = pred.shape[0]
    loss = nn.SmoothL1Loss(beta=1.0)
    future = torch.tensor(pred_1forward[:,:(-time_shifting)], dtype = torch.float32)
    target = torch.tensor(pred[:,time_shifting:], dtype = torch.float32)
    output = loss(future, target)
    return output.item()

def list_all_files(rootdir):
    _files = []
    listdir = os.listdir(rootdir)
    for i in range(len(listdir)):
        path = os.path.join(rootdir, listdir[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            _files.append(path)
    return _files


                
if __name__ == "__main__":
    
    root = '/home/phong/driving_data/official/gamma_planner/'
    all_files = list_all_files(root)
    all_files = [txtfile for txtfile in all_files if txtfile.split(".")[-1] == "txt"]
    random.shuffle(all_files)
    all_files = all_files[0:100]
    plot_x = []
    plot_y = []
    for myfile in all_files:
        print(myfile.split("/")[1].split("_")[0], " ", myfile.split("/")[-1].split("_")[4])
        try:
            action_list, ego_list, ego_path_list, exos_list, coll_bool_list, pred_car_list, pred_exo_list, trial_list, depth_list, max_min_coords, time_list = \
            parse_data(myfile)
        except:
            continue
        progress_time = progress(ego_list)
        print("Conflict happends at timestep", progress_time)

        #### CALCULATE THE TIME CONSISTENCY OF PREDICTION ####
        try:
            pred_len = 5
            pred = dict()
            pred_1forward = dict()
            compare, compare_1forward = [], []
            for key in exos_list.keys():
                if key == 0:
                    for i in range(len(exos_list[key])):
                        id = exos_list[key][i]['id']
                        pre = np.array([pred_exo_list[key][j][i]['pos'] for j in range(pred_len)])
                        pred[id] = pre
                else:
                    pred = copy.deepcopy(pred_1forward)
                
                pred_1forward = dict()
                if exos_list.get(key+2) is not None and pred_exo_list.get(key+2) is not None:
                    key_1forward = key+1
                    for i in range(len(exos_list[key_1forward])):
                        id_1forward = exos_list[key_1forward][i]['id']
                        pre_1forward = np.array([pred_exo_list[key_1forward][j][i]['pos'] for j in range(pred_len)])
                        pred_1forward[id_1forward] = pre_1forward
                for key in pred.keys():
                    if key in pred_1forward:
                        compare.append(pred[key])
                        compare_1forward.append(pred_1forward[key])
            compare = np.array(compare)
            compare_1forward = np.array(compare_1forward)
            tem_cos = consistency(compare,compare_1forward,1)
            print("Num of calculas:", compare.shape[0])
            print("Temporal Consistency for the closest Prediction:", tem_cos)
            
            #### CALCULATE THE TIME CONSISTENCY OF PLANNING ####
            T = 0.4
            a = 0
            comfort = []
            for key in ego_list.keys():
                if (key+1 in ego_list) and (key+2 in ego_list):
                    pos_xy = np.array(ego_list[key+1]['pos']) - np.array(ego_list[key]['pos'])
                    pos_xy_1forward = np.array(ego_list[key+2]['pos']) - np.array(ego_list[key+1]['pos'])
                    a_xy = (pos_xy_1forward - pos_xy)/(T*T)
                    a = np.sqrt(np.sum(a_xy**2))
                comfort.append(a)
            comfort = np.array(comfort)
            tem_cos_plan = np.sum(comfort)/len(comfort)
            print("Average Acceleration for Planning:", tem_cos_plan)
            plot_x.append(tem_cos)
            plot_y.append(tem_cos_plan)
        except:
            #print(myfile.split("\\")[1].split("_")[0], " ", myfile.split("\\")[-1].split("_")[4])
            #print("This file is not good")
            continue
    fig = plt.figure()
    pc = pearsonr(plot_x,plot_y)
    print("相关系数：",pc[0])
    print("显著性水平：",pc[1])
    plt.scatter(plot_x,plot_y,color = "red")
    plt.xlim(0,1.5)
    plt.ylim(0,7)
    plt.xlabel("Temporal Consistency")
    plt.ylabel("Acceleration")
    plt.show()


