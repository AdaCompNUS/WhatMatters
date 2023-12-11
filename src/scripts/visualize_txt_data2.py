import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.lines as mlines
import argparse
import pdb
import matplotlib.path as mpath
import random
import math
import matplotlib
import numpy as np
import sys
import seaborn as sns

matplotlib.use('Agg')
sns.set()


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


def agent_rect(agent_dict, origin, color, fill=True):
    try:
        pos = agent_dict['pos']
        heading = agent_dict['heading']
        bb_x, bb_y = agent_dict['bb']
        x_shift = [bb_y / 2.0 * math.cos(heading), bb_y / 2.0 * math.sin(heading)]
        y_shift = [-bb_x / 2.0 * math.sin(heading), bb_x / 2.0 * math.cos(heading)]
        coord = [pos[0] - origin[0] - x_shift[0] - y_shift[0], pos[1] - origin[1] - x_shift[1] - y_shift[1]]
        rect = mpatches.Rectangle(xy=coord, width=bb_y, height=bb_x, angle=np.rad2deg(heading), fill=fill, color=color)
        return rect
    except Exception as e:
        error_handler(e)
        pdb.set_trace()



def agent_rect_line(agent_dict, origin, color, fill=True, noise_min=None, noise_max=None, width=15, height=0.35, alpha=1):
    try:
        pos = agent_dict['pos']
        heading = agent_dict['heading']
        coord = [pos[0] - origin[0], pos[1] - origin[1]]
        if noise_min is None or noise_max is None:
            rect = mpatches.Rectangle(xy=coord, width=width, height=height, angle=np.rad2deg(heading), fill=fill, color=color, alpha=alpha)
        else:
            rect = mpatches.Rectangle(xy=coord, width=width, height=height, angle=np.rad2deg(heading * np.random.uniform(noise_min,noise_max)), fill=fill, color=color)
        return rect
    except Exception as e:
        error_handler(e)
        pdb.set_trace()


def agent_rect_curve(agent_dict, origin, color, fill=True, noise_min=None, noise_max=None, width=15, height=0.35, alpha=1):
    try:
        Path = mpath.Path
        pos = agent_dict['pos']
        heading = agent_dict['heading']
        coord = [pos[0] - origin[0], pos[1] - origin[1]]
        if noise_min is None or noise_max is None:
            # rect = mpatches.Rectangle(xy=coord, width=width, height=height, angle=np.rad2deg(heading), fill=fill, color=color, alpha=alpha)
            noise = random.randint(1,4)
            # curve = mpatches.PathPatch(Path([(coord[0], coord[1]), (coord[0]-noise/2, coord[1]), (coord[0]-noise, coord[1]+noise)],[Path.MOVETO, Path.CURVE3, Path.CURVE3]), fc="none", color=color)
            # curve = mpatches.PathPatch(Path([(coord[0], coord[1]), (coord[0], coord[1]-noise/2), (coord[0]-noise, coord[1]+noise)],[Path.MOVETO, Path.CURVE3, Path.CURVE3]), fc="none", color=color)
            curve = mpatches.PathPatch(Path([(coord[0], coord[1]), (coord[0], coord[1]+agent_dict['vel'][1]*2), (coord[0]+agent_dict['vel'][0]*2, coord[1]+agent_dict['vel'][1]*2)],[Path.MOVETO, Path.CURVE3, Path.CURVE3]), fc="none", color=color)
        else:
            # rect = mpatches.Rectangle(xy=coord, width=width, height=height, angle=np.rad2deg(heading * np.random.uniform(noise_min,noise_max)), fill=fill, color=color)
            noise = np.random.uniform(noise_min,noise_max) * 4
            # curve = mpatches.PathPatch(Path([(coord[0], coord[1]), (coord[0]-noise/2, coord[1]), (coord[0]-noise, coord[1]+noise)],[Path.MOVETO, Path.CURVE3, Path.CURVE3]), fc="none", color=color)
            curve = mpatches.PathPatch(Path([(coord[0], coord[1]), (coord[0], coord[1]-noise/2), (coord[0]-noise, coord[1]+noise)],[Path.MOVETO, Path.CURVE3, Path.CURVE3]), fc="none", color=color)

        return curve
    except Exception as e:
        error_handler(e)
        pdb.set_trace()


def vel_arrow(agent_dict, origin, color):
    try:
        vel = agent_dict['vel']
        pos = agent_dict['pos']
        arrow = mpatches.Arrow(x=pos[0] - origin[0], y=pos[1] - origin[1], dx=vel[0], dy=vel[1], color=color)
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
        if mode == 'acc':
            arrow = mpatches.Arrow(x=0.0, y=0.0, dx=math.cos(heading) * acc, dy=math.sin(heading) * acc, color='red', width=4)
        else:
            arrow = mpatches.Arrow(x=0.0, y=0.0, dx=math.cos(heading) * speed, dy=math.sin(heading) * speed, color='hotpink', width=2)
        return arrow
    except Exception as e:
        error_handler(e)
        pdb.set_trace()


def init():
    # initialize an empty list of circles
    return [trial_text, depth_text]


def animate(time_step):
    ax.clear()
    plt.axis([-sx, sx, -sy, sy])
    ax.set_aspect(sy / sx)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title("Time step: " + str(time_step), fontsize=8)
    patches = [trial_text, depth_text]

    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('2')

    time_step = time_step + args.frame

    print("Drawing time step {}...".format(time_step))


    ego_pos = ego_list[time_step]['pos']
    # draw ego agent
    if time_step in coll_bool_list.keys():
        ego_color = 'crimson'
    else:
        ego_color = 'royalblue'

    # draw path
    if time_step < len(ego_path_list):
        path = ego_path_list[time_step]
        for i in range(0, int(len(path)/2), 2):
            point = path[i]
            patches.append(ax.add_patch(mpatches.Circle([point[0] - ego_pos[0], point[1] - ego_pos[1]], 0.1, color='cyan')))

    if time_step in pred_car_list.keys():
        for car_dict in pred_car_list[time_step]:
            car_dict['bb'] = ego_list[time_step]['bb']
            # patches.append(ax.add_patch(agent_rect_line(car_dict, ego_pos, 'lightskyblue', True)))

    closest_exo_agent_id = []
    l2_dists = []
    # find closest exo agent
    closest_exo_agent_dist = float('inf')
    for agent_dict in exos_list[time_step]:
        l2_distance = ((np.array(car_dict['pos']) - np.array(agent_dict['pos'])) ** 2).sum()
        l2_dists.append((l2_distance, agent_dict['id']))
        # if l2_distance < closest_exo_agent_dist and random.random() > 0.01:
        #     closest_exo_agent_dist = l2_distance
        #     closest_exo_agent_id = agent_dict['id']
    closest_exo_agent_id = [ids for _, ids in sorted(l2_dists)[:1]]

    # draw exo agents
    for agent_dict in exos_list[time_step]:
        if agent_dict['id'] in closest_exo_agent_id:
            patches.append(ax.add_patch(agent_rect_curve(agent_dict, ego_pos, 'red', noise_min=0.7, noise_max=0.9, width=7, height=0.3)))
            patches.append(ax.add_patch(agent_rect_curve(agent_dict, ego_pos, 'gold', noise_min=0.9, noise_max=1, width=7, height=0.3)))
            patches.append(ax.add_patch(agent_rect_curve(agent_dict, ego_pos, 'green', width=7, height=0.3)))
            patches.append(ax.add_patch(agent_rect(agent_dict, ego_pos, 'darkorange')))
        else:
            patches.append(ax.add_patch(agent_rect_curve(agent_dict, ego_pos, 'purple', width=agent_dict['bb'][1]*1.5, height=0.25, alpha=0.5)))
            patches.append(ax.add_patch(agent_rect(agent_dict, ego_pos, 'darkgray')))
            # patches.append(ax.add_patch(vel_arrow(agent_dict, ego_pos, 'purple')))

    # draw predicted exo agents
    # if time_step in pred_exo_list.keys():
    #     for agent_list in pred_exo_list[time_step]:
    #         for agent_dict in agent_list:
    #             patches.append(ax.add_patch(
    #                 agent_rect(agent_dict, ego_pos, 'purple', False)))

    if time_step in ego_list.keys():
        patches.append(ax.add_patch(agent_rect(ego_list[time_step], ego_pos, ego_color)))
    if time_step in action_list.keys():
        patches.append(ax.add_patch(acc_arrow(action_list[time_step], ego_list[time_step], mode='acc')))
        # patches.append(ax.add_patch(acc_arrow(action_list[time_step], ego_list[time_step], mode='speed')))
    if time_step in ego_list.keys():
        patches.append(ax.add_patch(vel_arrow(ego_list[time_step], ego_pos, 'purple')))

    red_arrow = mlines.Line2D([], [], color='red', label="Accel / Decel", marker='$\u2192$', linestyle='None')
    pink_arrow = mlines.Line2D([], [], color='hotpink', label="Desired Velocity", marker='$\u2192$', linestyle='None')
    purple_arrow = mlines.Line2D([], [], color='purple', label="Actual Velocity", marker='$\u2192$', linestyle='None')
    black_square = mlines.Line2D([], [], color='darkgray', label="Exo-agents", marker='s', linestyle='None')
    darkorange_square = mlines.Line2D([], [], color='darkorange', label="Highlighted Exo-agent", marker='s', linestyle='None')
    royalblue_square = mlines.Line2D([], [], color='royalblue', label="Ego-agent Loc", marker='s', linestyle='None')
    path_line = mlines.Line2D([], [], color='cyan', label="Path To Follow", marker='_', linestyle='None')
    lightskyblue_square = mlines.Line2D([], [], color='lightskyblue', label="Predicted Ego-agent Loc", marker='$\u25A1$', linestyle='None')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              handles=[royalblue_square, black_square, lightskyblue_square, darkorange_square, red_arrow, purple_arrow, pink_arrow, path_line],
              fontsize=6, fancybox=True, ncol=4)

    return patches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True, help='Text file to animate')
    parser.add_argument('--frame', type=int, default=0, help='Start frame')
    #parser.add_argument('--output-file', type=str, required=True, help='File name to save the output animation')
    args = parser.parse_args()
    action_list, ego_list, ego_path_list, exos_list, coll_bool_list, pred_car_list, pred_exo_list, trial_list, depth_list = \
        parse_data(args.input_file)

    sx = 40.0
    sy = 40.0
    fig = plt.figure()
    plt.axis([-sx, sx, -sy, sy])
    ax = plt.gca()
    ax.set_aspect(sy / sx)
    anim_running = True

    trial_text = plt.text(-sx + 5.0, -sy + 2.0, '', fontsize=10)
    depth_text = plt.text(-sx + 5.0, -sy + 6.0, '', fontsize=10)

    print(ego_list)

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(ego_list.keys()) - args.frame, interval=300,
                         blit=True)
    #anim.save(args.output_file, dpi=300, writer=PillowWriter(fps=25))