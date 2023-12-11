import os
import fnmatch
import argparse
import numpy as np
import math
import random
import pandas as pd

# This script is different from statistics_for_moped.py in which  the starting point of outputing agents does not
# need to have that starting point

OUTPUT_FOLDER = "/home/cunjun/driving_data_sparse3/result/summit_process/"#"/home/cunjun/moped_data/summit/"

cap = 10

def collect_txt_files(rootpath, flag):
    txt_files = list([])
    for root, dirnames, filenames in os.walk(rootpath):

        if flag in root and ignore_flag not in root and 'debug' not in root:
            # print("subfolder %s found" % root)
            for filename in fnmatch.filter(filenames, '*.txt'):
                # append the absolute path for the file
                txt_files.append(os.path.join(root, filename))
    print("%d files found in %s" % (len(txt_files), rootpath))
    return txt_files


def filter_txt_files(root_path, txt_files):
    # container for files to be converted to h5 data
    filtered_files = list([])

    no_aa_count = 0
    # Filter trajectories that don't reach goal or collide before reaching goal
    for txtfile in txt_files:
        ok_flag = False
        no_aa = False
        with open(txtfile, 'r') as f:
            for line in reversed(list(f)):
                if 'Step {}'.format(cap + 1) in line or 'step {}'.format(cap + 1) in line:
                    ok_flag = True
                if 'No agent array messages received after' in line:
                    no_aa_count += 1
                    no_aa = True
        if ok_flag == True:
            filtered_files.append(txtfile)
            # print("good file: ", txtfile)
        else:
            if no_aa:
                pass # print("no aa file: ", txtfile)
            else:
                pass # print("unused file: ", txtfile)
    print("NO agent array in {} files".format(no_aa_count))

    filtered_files.sort()
    print("%d filtered files found in %s" % (len(filtered_files), root_path))
    # print (filtered_files, start_file, end_file)
    #
    return filtered_files


def get_statistics(root_path, filtered_files):
    total_count = len(filtered_files)
    col_count = 0

    # 80% for train and 20% for test
    random.shuffle(filtered_files)
    train_filtered_files, test_filtered_files = filtered_files[0: int(0.8*len(filtered_files))], filtered_files[int(0.8*len(filtered_files)):]

    print("%d filtered files found in %s" % (len(filtered_files), root_path))

    for files in [train_filtered_files, test_filtered_files]:

        if files == train_filtered_files:
            train_test = "train"
        else:
            train_test = "test"

        for txtfile in files:
            # if "29688" not in txtfile:
            #     continue
            #
            reach_goal_flag = False
            collision_flag = False
            cur_step = 0
            dec_count = 0
            acc_count = 0
            mat_count = 0
            speed = 0.0
            last_speed = 0.0
            ave_speed = 0.0
            dist = 0.0
            last_pos = None

            print(f"Processing file {txtfile}")

            with open(txtfile, 'r') as f:
                data_pos = {}
                cur_step = 0
                city_name = ""

                try:
                    for line in f:
                        if '-map_location' in line:
                            city_name = line.split()[1]
                        if 'executing step' in line:
                            line_1 = line.split('executing step ', 1)[1]
                            cur_step = int(line_1.split('=', 1)[0])
                        elif 'Round 0 Step' in line:
                            line_1 = line.split('Round 0 Step ', 1)[1]
                            cur_step = int(line_1.split('-', 1)[0])
                        elif 'goal reached at step' in line:
                            line_1 = line.split('goal reached at step ', 1)[1]
                            cur_step = int(line_1.split(' ', 1)[0])
                        elif ("pomdp" in folder or "gamma" in folder or "rollout" in folder) and "car pos / heading / vel" in line:
                            # = (149.52, 171.55) / 1.3881 / 0.50245
                            speed = float(line.split(' ')[12])
                            pos_x = float(line.split(' ')[7].replace('(', '').replace(',', ''))
                            pos_y = float(line.split(' ')[8].replace(')', '').replace(',', ''))
                            if cur_step >= cap:
                                ave_speed += speed
                            pos = [pos_x, pos_y]

                            if last_pos:
                                dist += math.sqrt((pos[0]-last_pos[0])**2 + (pos[1]-last_pos[1])**2)
                            last_pos = pos
                            if "gamma" in folder or 'pomdp' in folder or "rollout" in folder:
                                if speed< last_speed - 0.2:
                                    dec_count += 1
                                last_speed = speed

                            if "av" in data_pos:
                                val = data_pos["av"]
                                val[cur_step] = pos
                            else:
                                data_pos["av"] = {cur_step: pos}

                        elif ("pomdp" in folder or "gamma" in folder or "rollout" in folder) and "id / pos / speed / vel" in line:
                            agentid = int(line.split()[16].replace('(', '').replace(',', ''))
                            pos_x = float(line.split()[18].replace('(', '').replace(',', ''))
                            pos_y = float(line.split()[19].replace(')', '').replace(',', ''))
                            pos = [pos_x, pos_y]

                            if f"agent{agentid}" in data_pos:
                                val = data_pos[f"agent{agentid}"]
                                val[cur_step] = pos
                            else:
                                data_pos[f"agent{agentid}"] = {cur_step: pos}

                        if 'goal reached' in line:
                            reach_goal_flag = True
                            break
                        if ('collision = 1' in line or 'INININ' in line or 'in real collision' in line) and reach_goal_flag == False:
                            collision_flag = True
                            col_count += 1
                            break
                except Exception as e:
                    print(e)
                    print("Will not process this file")
                    continue

                if cur_step >= 60:
                    # 0. Get the starting index by considering the maximum number of agents having that index
                    start_index = 0
                    end_index = start_index + min(150, cur_step-50) # we want at max 50 but at least 150 for many random

                    index_count_dict = {} # This dict to maintain the counts so that we can sample probabilities
                    for index in random.sample(range(start_index, end_index), k = min(end_index, 50)):
                        count = 0
                        for v in data_pos.values():
                            if index in v.keys():
                                count += 1
                            if index in v.keys() and (index+49) in v.keys(): # prioritize to have at least 1 50-sequence agents
                                count += 100

                        index_count_dict[index] = count

                    # Normalize the index
                    index_count_dict_sum_of_vals = sum([v for v in index_count_dict.values()])
                    normalized_count_dict = {k: v / index_count_dict_sum_of_vals for k,v in index_count_dict.items()}
                    sampled_index = random.choices(range(len(normalized_count_dict)), weights = list(normalized_count_dict.values()), k=1)[0]
                    start_index = sampled_index

                    # 1. Get only agent's whose trajectory start at time start_index
                    considered_agents = {}
                    for k, v in data_pos.items():
                        if ("agent" in k) and (start_index in v.keys()):
                            v_from_start_index = {x:y for x, y in v.items() if x >= start_index}
                            considered_agents[k] = v_from_start_index

                    # 2. We find the interested agent by sampling a list of nearest agents to the car and trajectory of nearest agent >= 50
                    car_pos = np.array(list(data_pos["av"].values()))
                    nearest_values = {}
                    for k, v in considered_agents.items():
                        pos = np.array(list(v.values()))
                        if pos.shape[0] >= 50:
                            val = np.mean(np.sum((pos[0:50] - car_pos[start_index:start_index+50])**2, axis=-1))
                            nearest_values.setdefault(val, k)

                    # Get at most 3 interested neareast agents and sampling from them
                    if len(nearest_values) == 0:
                        print(f'ahaha start{start_index} end_index {end_index}')
                        print(txtfile)
                        continue

                    sorted_nearest = sorted(nearest_values.items(), key = lambda x : x[0])
                    random_interested_agent = random.choice(sorted_nearest[0:3])

                    # 3. Building file. Each file has TIMESTAMP, TRACk_ID OBJECT_TYPE, X, Y, CITY_NAME
                    # 3.1 Add "av" to considered_agents
                    #considered_agents["av"] = data_pos["av"]
                    data_frame = []
                    # 3.2 Build the dictionary to output pandas frame
                    for k, v in data_pos.items():
                        for timestamp, pos in v.items():
                            if ((timestamp - start_index) >= 0) and ((timestamp - start_index) < 50): # cater for 'av' type
                                temp = {
                                    "TIMESTAMP":timestamp-start_index,
                                    "TRACK_ID": 0 if "av" in k else k[5:],
                                    "OBJECT_TYPE": "AV" if k == "av" else ("AGENT" if k == random_interested_agent[1] else "OTHERS"),
                                    "X":pos[0],
                                    "Y":pos[1],
                                    "CITY_NAME": city_name
                                }
                                data_frame.append(temp)

                    file_name = os.path.split(txtfile)[-1].split('.')[0] + '.csv'
                    file_path = os.path.join(os.path.join(OUTPUT_FOLDER, train_test), file_name)

                    os.makedirs(os.path.dirname(file_path), exist_ok=True)

                    data_frame = pd.DataFrame(data_frame)
                    # Some agent has lacking frames between, thus we just delete this file instead considering it
                    if data_frame.loc[data_frame["OBJECT_TYPE"] == "AGENT"].shape[0] != 50:
                        continue

                    assert data_frame.loc[data_frame["OBJECT_TYPE"] == "AV"].shape[0] == 50

                    data_frame.to_csv(file_path, index=False)
                    print(f"Done exporting start_index {start_index} file {file_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--flag',
        type=str,
        default='',
        help='Folder name to track')
    parser.add_argument(
        '--ignore',
        type=str,
        default='nothing',
        help='folder flag to ignore')
    parser.add_argument(
        '--folder',
        type=str,
        default='/home/cunjun/driving_data_sparse3/result/joint_pomdp_drive_mode/',
        help='Subfolder to check')

    flag = parser.parse_args().flag
    folder = parser.parse_args().folder
    ignore_flag = parser.parse_args().ignore

    files = collect_txt_files(folder, flag)
    filtered = filter_txt_files(folder, files)
    get_statistics(folder, filtered)




