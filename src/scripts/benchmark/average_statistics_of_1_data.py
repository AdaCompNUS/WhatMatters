import matplotlib.pyplot as plt
import os
import numpy as np
import fnmatch
import sys
import math

FILE_PATH = f'/home/cunjun/driving_data/representative/original_gamma_nosync/'

cap = 10


def collect_txt_files(rootpath, flag, ignore_flag="nothing"):
    txt_files = list([])
    for root, dirnames, filenames in os.walk(rootpath):

        if flag in root and ignore_flag not in root and 'debug' not in root:
            # print("subfolder %s found" % root)
            for filename in fnmatch.filter(filenames, '*.txt'):
                # append the absolute path for the file
                txt_files.append(os.path.join(root, filename))
    print("%d files found in %s" % (len(txt_files), rootpath))
    return txt_files


def filter_txt_files(root_path, txt_files, cap=10):
    # container for files to be converted to h5 data
    filtered_files = list([])

    no_aa_count = 0
    # Filter trajectories that don't reach goal or collide before reaching goal
    for txtfile in txt_files:
        ok_flag = False
        no_aa = False
        with open(txtfile, 'r') as f:
            try:
                for line in reversed(list(f)):
                    if 'Step {}'.format(cap + 1) in line or 'step {}'.format(cap + 1) in line:
                        ok_flag = True
                    if 'No agent array messages received after' in line:
                        no_aa_count += 1
                        no_aa = True
            except:
                print(txtfile)
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


def merge_3_get_statistics(folder_path):
    file_path_list = filter_txt_files(folder_path, collect_txt_files(folder_path, flag="", ignore_flag="nothing"))

    total_moped_steps = [] # Number of motion prediction steps in each tree construction
    total_times = [] # Number of time used for search for each action (time for tree construction)
    total_trials = [] # Number of trials for each tree construction
    total_node_expansions = [] # Number of node expansions (equal to number of rollout) per tree construction
    total_node_in_total = [] # Number of nodes in tree per tree construction
    tree_depth = [] # Depth of tree per tree construction
    total_execution_steps = [] # Number of actions take per scenario (len of this variable = length of files)
    total_running_time_each_scenario = [] # Running time per scenario (len of this variable = length of files)

    prediction_time = []

    total_count = len(file_path_list)
    col_count = 0
    goal_count = 0
    # Filter trajectories that don't reach goal or collide before reaching goal
    eps_step = []
    goal_step = []
    ave_speeds = []
    dec_counts = []
    acc_counts = []
    mat_counts = []
    rew_counts = []
    trav_dists= []
    stuck_counts = []
    progress_dists = []


    connect_finish_at = []
    time_car_states = []
    time_agent_arrays = []
    agent_counts = []
    total_agents = []
    agent_steps_counts = []
    agent_speeds = []

    # Limit to 100 files only
    file_count = 0
    for file_path in file_path_list:
        with open(file_path, 'r') as f:
            start_reading = False

            sum_steps_per_tree_construction = 0
            execution_steps = 0 # The maximum excution steps
            running_time_each_scenario = 0 # Maximum running time, roughly equal execution_steps * time_each_search_tree
            data_pos = {}

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
            avg_reward = 0.0
            progress_dist = 0.0 # Increase using dist. Reset if speed < 0.2 meter / seconds

            time_car_state = []
            time_agent_array = []
            agent_count = []

            for line in f.readlines():
                if 'Construct tree' in line:
                    start_reading = True # to start reading from beginning

                if start_reading and "ContextPomdp::Step 123" in line:
                    sum_steps_per_tree_construction += 1

                if "deleting tree" in line:
                    total_moped_steps.append(sum_steps_per_tree_construction)
                    sum_steps_per_tree_construction = 0

                if "[RunStep] Time spent in N6despot6DESPOTE::Search()" in line:
                    total_times.append(float(line.split(' ')[-1]))

                if "Trials: no." in line:
                    total_trials.append(int(line.split(' ')[6]))
                    tree_depth.append(int(line.split(' ')[8]))

                if "# nodes: expanded" in line:
                    total_node_expansions.append(int(line.split(' ')[8]))
                    total_node_in_total.append(int(line.split(' ')[10]))

                if "All MopedPred Time" in line:
                    try:
                        prediction_time.append(float(line.split(' ')[6]))
                    except:
                        pass

                if "Round 0 Step" in line:
                    execution_steps = int(line.split()[-1].split('-')[0])

                if "ExecuteAction at the" in line:
                    running_time_each_scenario = float(line.split()[4][:-2])

                if 'executing step' in line:
                    line_1 = line.split('executing step ', 1)[1]
                    cur_step = int(line_1.split('=', 1)[0])
                elif 'Round 0 Step' in line:
                    line_1 = line.split('Round 0 Step ', 1)[1]
                    cur_step = int(line_1.split('-', 1)[0])
                elif 'goal reached at step' in line:
                    line_1 = line.split('goal reached at step ', 1)[1]
                    cur_step = int(line_1.split(' ', 1)[0])
                elif 'action **=' in line:
                    acc = int(line.split(' ')[2]) % 3
                    if acc == 1:
                        acc_count += 1
                    elif acc == 0:
                        mat_count += 1
                elif 'reward **=' in line:
                    if cur_step >= cap:
                        avg_reward += float(line.split(' ')[2])
                elif "car pos / heading / vel" in line:
                    # = (149.52, 171.55) / 1.3881 / 0.50245
                    speed = float(line.split(' ')[12])
                    pos_x = float(line.split(' ')[7].replace('(', '').replace(',', ''))
                    pos_y = float(line.split(' ')[8].replace(')', '').replace(',', ''))
                    if cur_step >= cap:
                        ave_speed += speed
                    pos = [pos_x, pos_y]

                    if last_pos:
                        dist += math.sqrt((pos[0]-last_pos[0])**2 + (pos[1]-last_pos[1])**2)
                        progress_dist += math.sqrt((pos[0]-last_pos[0])**2 + (pos[1]-last_pos[1])**2)
                    last_pos = pos

                    if cur_step >= cap:
                        if last_speed > 2.0 and speed < 0.2:
                            progress_dists.append(progress_dist)
                            progress_dist = 0.0

                    if speed< last_speed - 0.2:
                        dec_count += 1
                    last_speed = speed

                    if "av" in data_pos:
                        val = data_pos["av"]
                        val[cur_step] = pos
                    else:
                        data_pos["av"] = {cur_step: pos}
                elif "id / pos / speed / vel" in line:
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

                if "Connect finished at" in line:
                    connect_finish_at.append(float(line.split()[5][:-2]))

                if "get car state at" in line:
                    t = float(line.split()[5].split("=")[-1])
                    time_car_state.append(t)

                if "MSG: receive" in line and "agents at time" in line:
                    t = float(line.split()[6][:-1])
                    time_agent_array.append(t)
                    count = int(line.split()[2])
                    agent_count.append(count)

            total_execution_steps.append(execution_steps)
            total_running_time_each_scenario.append(running_time_each_scenario)

        eps_step.append(cur_step)
        if cur_step > cap:
            ave_speed = ave_speed / (cur_step - cap)
            ave_speeds.append(ave_speed)
            dec_count  = dec_count / float(cur_step)
            acc_count  = acc_count / float(cur_step)
            mat_count  = mat_count / float(cur_step)
            dec_counts.append(dec_count)
            acc_counts.append(acc_count)
            mat_counts.append(mat_count)
            trav_dists.append(dist)
            avg_reward = avg_reward / float(cur_step - cap)
            rew_counts.append(avg_reward)
            progress_dists.append(progress_dist)
            if dist <= 1:
                stuck_counts.append(1)
            else:
                stuck_counts.append(0)
            if reach_goal_flag == True:
                goal_count+=1
                assert(cur_step != 0)
                goal_step.append(cur_step)
            else:
                pass # print("fail file: ", txtfile)
            if collision_flag == True:
                pass #col_count += 1
                # print("col file: ", txtfile)

            time_agent_arrays.append(time_agent_array)
            time_car_states.append(time_car_state)
            agent_counts.append(np.mean(np.array(agent_count)))

            total_agents.append(len(data_pos) - 1)
            agent_steps_count = [] # count how long agent lives

            for k, v in data_pos.items():
                avg_speed = []
                agent_steps_count.append(len(v))
                prev_time = None
                for timestep in v.keys():
                    if prev_time is None:
                        prev_time = timestep
                        continue
                    if timestep == (prev_time + 1):
                        zzz = np.sqrt((v[timestep][0] - v[prev_time][0])**2 + (v[timestep][1] - v[prev_time][1])**2)
                        avg_speed.append(zzz)
                        prev_time = timestep

                agent_speeds.append(np.mean(np.array(avg_speed)))

            agent_steps_counts.append(np.mean(np.array(agent_steps_count)))

        file_count += 1
        if file_count >= 10:
            break

    # print("goal rate :", float(goal_count)/total_count)
    print("col rate :", float(col_count)/total_count)
    ave_speeds_np = np.asarray(ave_speeds)
    print("ave speed :", np.average(ave_speeds_np))
    freq = 3
    # if 'porca' in folder:
    #     freq = 10

    # print('time to goal :', float(sum(goal_step))/len(goal_step)/freq)
    dec_np = np.asarray(dec_counts)
    acc_np = np.asarray(acc_counts)
    mat_np = np.asarray(mat_counts)
    print('dec_count:', np.average(dec_np))
    print('acc_count:', np.average(acc_np))
    print('mat_count:', np.average(mat_np))
    trav_np = np.asarray(trav_dists)
    rew_np = np.asarray(rew_counts)
    progress_np = np.asarray(progress_dists)
    stuck_np = np.asarray(stuck_counts)
    print('travelled dist:', np.average(trav_np))
    print('travelled dist total:', np.sum(trav_np))
    print("col rate per meter:", float(col_count)/np.sum(trav_np))
    print("col rate per step:", float(col_count)/np.sum(eps_step))
    print('reward average: ', np.average(rew_np))
    print('progress distance average: ', np.average(progress_np))
    print('stuck average: ', np.average(stuck_np))

    connect_finish_at = np.array(connect_finish_at)


    return (total_moped_steps, total_times, total_trials, total_node_expansions, total_node_in_total, tree_depth, prediction_time,
        total_execution_steps, total_running_time_each_scenario), \
            (connect_finish_at, time_car_states, time_agent_arrays, agent_counts, total_agents, agent_steps_counts, agent_speeds), \
            (dec_np, acc_np, mat_np, float(col_count)/total_count, trav_np, rew_np, progress_np, ave_speeds_np)

def get_statistics_search_tree(folder_path):

    file_path_list = filter_txt_files(folder_path, collect_txt_files(folder_path, flag="", ignore_flag="nothing"))

    total_moped_steps = [] # Number of motion prediction steps in each tree construction
    total_times = [] # Number of time used for search for each action (time for tree construction)
    total_trials = [] # Number of trials for each tree construction
    total_node_expansions = [] # Number of node expansions (equal to number of rollout) per tree construction
    total_node_in_total = [] # Number of nodes in tree per tree construction
    tree_depth = [] # Depth of tree per tree construction
    total_execution_steps = [] # Number of actions take per scenario (len of this variable = length of files)
    total_running_time_each_scenario = [] # Running time per scenario (len of this variable = length of files)

    prediction_time = []

    # Limit to 100 files only
    file_count = 0
    for file_path in file_path_list:
        with open(file_path, 'r') as f:
            start_reading = False

            sum_steps_per_tree_construction = 0
            execution_steps = 0 # The maximum excution steps
            running_time_each_scenario = 0 # Maximum running time, roughly equal execution_steps * time_each_search_tree
            for line in f.readlines():
                if 'Construct tree' in line:
                    start_reading = True # to start reading from beginning

                if start_reading and "ContextPomdp::Step 123" in line:
                    sum_steps_per_tree_construction += 1

                if "deleting tree" in line:
                    total_moped_steps.append(sum_steps_per_tree_construction)
                    sum_steps_per_tree_construction = 0

                if "[RunStep] Time spent in N6despot6DESPOTE::Search()" in line:
                    total_times.append(float(line.split(' ')[-1]))

                if "Trials: no." in line:
                    total_trials.append(int(line.split(' ')[6]))
                    tree_depth.append(int(line.split(' ')[8]))

                if "# nodes: expanded" in line:
                    total_node_expansions.append(int(line.split(' ')[8]))
                    total_node_in_total.append(int(line.split(' ')[10]))

                if "All MopedPred Time" in line:
                    try:
                        prediction_time.append(float(line.split(' ')[6]))
                    except:
                        pass

                if "Round 0 Step" in line:
                    execution_steps = int(line.split()[-1].split('-')[0])

                if "ExecuteAction at the" in line:
                    running_time_each_scenario = float(line.split()[4][:-2])

            total_execution_steps.append(execution_steps)
            total_running_time_each_scenario.append(running_time_each_scenario)

        file_count += 1
        if file_count >= 10:
            break

    return total_moped_steps, total_times, total_trials, total_node_expansions, total_node_in_total, tree_depth, prediction_time,\
        total_execution_steps, total_running_time_each_scenario

def get_statistics_results(folder_path):

    file_path_list = filter_txt_files(folder_path, collect_txt_files(folder_path, flag="", ignore_flag="nothing"))


    total_count = len(file_path_list)
    col_count = 0
    goal_count = 0
    # Filter trajectories that don't reach goal or collide before reaching goal
    eps_step = []
    goal_step = []
    ave_speeds = []
    dec_counts = []
    acc_counts = []
    mat_counts = []
    rew_counts = []
    trav_dists= []
    stuck_counts = []
    progress_dists = []
    for txtfile in file_path_list:
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
        avg_reward = 0.0
        progress_dist = 0.0 # Increase using dist. Reset if speed < 0.2 meter / seconds
        with open(txtfile, 'r') as f:
            for line in f:
                if 'executing step' in line:
                    line_1 = line.split('executing step ', 1)[1]
                    cur_step = int(line_1.split('=', 1)[0])
                elif 'Round 0 Step' in line:
                    line_1 = line.split('Round 0 Step ', 1)[1]
                    cur_step = int(line_1.split('-', 1)[0])
                elif 'goal reached at step' in line:
                    line_1 = line.split('goal reached at step ', 1)[1]
                    cur_step = int(line_1.split(' ', 1)[0])
                # elif "porca" in folder and 'pos / yaw / speed /' in line:
                #     speed = line.split(' ')[13]
                #     if speed< last_speed:
                #         dec_count += 1
                #     last_speed = speed
                elif 'action **=' in line:
                    acc = int(line.split(' ')[2]) % 3
                    # if acc == 2:
                    # dec_count += 1
                    if acc == 1:
                        acc_count += 1
                    elif acc == 0:
                        mat_count += 1
                elif 'reward **=' in line:
                    if cur_step >= cap:
                        avg_reward += float(line.split(' ')[2])
                elif "car pos / heading / vel" in line:
                    # = (149.52, 171.55) / 1.3881 / 0.50245
                    speed = float(line.split(' ')[12])
                    pos_x = float(line.split(' ')[7].replace('(', '').replace(',', ''))
                    pos_y = float(line.split(' ')[8].replace(')', '').replace(',', ''))
                    if cur_step >= cap:
                        ave_speed += speed
                    pos = [pos_x, pos_y]

                    if last_pos:
                        dist += math.sqrt((pos[0]-last_pos[0])**2 + (pos[1]-last_pos[1])**2)
                        progress_dist += math.sqrt((pos[0]-last_pos[0])**2 + (pos[1]-last_pos[1])**2)
                    last_pos = pos

                    if cur_step >= cap:
                        if last_speed > 2.0 and speed < 0.2:
                            progress_dists.append(progress_dist)
                            progress_dist = 0.0

                    if speed< last_speed - 0.2:
                        dec_count += 1
                    last_speed = speed



                # elif "imitation" in folder and 'car pos / dist_trav / vel' in line:
                #     speed = line.split(' ')[12]
                #     if speed< last_speed:
                #         dec_count += 1
                #     last_speed = speed
                # elif "lets-drive" in folder and 'car pos / dist_trav / vel' in line:
                #     speed = line.split(' ')[12]
                #     if speed< last_speed:
                #         dec_count += 1
                #     last_speed = speed

                if 'goal reached' in line:
                    reach_goal_flag = True
                    break
                if ('collision = 1' in line or 'INININ' in line or 'in real collision' in line) and reach_goal_flag == False:
                    collision_flag = True
                    col_count += 1
                    break

        eps_step.append(cur_step)
        if cur_step > cap:
            ave_speed = ave_speed / (cur_step - cap)
            ave_speeds.append(ave_speed)
            dec_count  = dec_count / float(cur_step)
            acc_count  = acc_count / float(cur_step)
            mat_count  = mat_count / float(cur_step)
            dec_counts.append(dec_count)
            acc_counts.append(acc_count)
            mat_counts.append(mat_count)
            trav_dists.append(dist)
            avg_reward = avg_reward / float(cur_step - cap)
            rew_counts.append(avg_reward)
            progress_dists.append(progress_dist)
            if dist <= 1:
                stuck_counts.append(1)
            else:
                stuck_counts.append(0)
            if reach_goal_flag == True:
                goal_count+=1
                assert(cur_step != 0)
                goal_step.append(cur_step)
            else:
                pass # print("fail file: ", txtfile)
            if collision_flag == True:
                pass #col_count += 1
                # print("col file: ", txtfile)
    #print("%d filtered files found in %s" % (len(filtered_files), root_path))

    # print("goal rate :", float(goal_count)/total_count)
    print("col rate :", float(col_count)/total_count)
    ave_speeds_np = np.asarray(ave_speeds)
    print("ave speed :", np.average(ave_speeds_np))
    freq = 3
    # if 'porca' in folder:
    #     freq = 10

    # print('time to goal :', float(sum(goal_step))/len(goal_step)/freq)
    dec_np = np.asarray(dec_counts)
    acc_np = np.asarray(acc_counts)
    mat_np = np.asarray(mat_counts)
    print('dec_count:', np.average(dec_np))
    print('acc_count:', np.average(acc_np))
    print('mat_count:', np.average(mat_np))
    trav_np = np.asarray(trav_dists)
    rew_np = np.asarray(rew_counts)
    progress_np = np.asarray(progress_dists)
    stuck_np = np.asarray(stuck_counts)
    print('travelled dist:', np.average(trav_np))
    print('travelled dist total:', np.sum(trav_np))
    print("col rate per meter:", float(col_count)/np.sum(trav_np))
    print("col rate per step:", float(col_count)/np.sum(eps_step))
    print('reward average: ', np.average(rew_np))
    print('progress distance average: ', np.average(progress_np))
    print('stuck average: ', np.average(stuck_np))


def get_auxiliary_results(folder_path):
    file_path_list = filter_txt_files(folder_path, collect_txt_files(folder_path, flag="", ignore_flag="nothing"))

    connect_finish_at = 0.0
    time_car_state = []
    agent_array = []
    agent_counts = []

    # Limit to 100 files only
    for file_path in file_path_list:
        with open(file_path, 'r') as f:
            data_pos = {}
            cur_step = 0
            for line in f.readlines():
                if "Connect finished at" in line:
                    connect_finish_at = float(line.split()[5][:-2])

                if "get car state at" in line:
                    t = float(line.split()[5].split("=")[-1])
                    time_car_state.append(t)

                if "MSG: receive" in line and "agents at time" in line:
                    t = float(line.split()[6][:-1])
                    agent_array.append(t)
                    count = int(line.split()[2])
                    agent_counts.append(count)

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
                elif "car pos / heading / vel" in line:
                    # = (149.52, 171.55) / 1.3881 / 0.50245
                    pos_x = float(line.split(' ')[7].replace('(', '').replace(',', ''))
                    pos_y = float(line.split(' ')[8].replace(')', '').replace(',', ''))
                    pos = [pos_x, pos_y]

                    if "av" in data_pos:
                        val = data_pos["av"]
                        val[cur_step] = pos
                    else:
                        data_pos["av"] = {cur_step: pos}

                elif "id / pos / speed / vel" in line:
                    agentid = int(line.split()[16].replace('(', '').replace(',', ''))
                    pos_x = float(line.split()[18].replace('(', '').replace(',', ''))
                    pos_y = float(line.split()[19].replace(')', '').replace(',', ''))
                    pos = [pos_x, pos_y]

                    if f"agent{agentid}" in data_pos:
                        val = data_pos[f"agent{agentid}"]
                        val[cur_step] = pos
                    else:
                        data_pos[f"agent{agentid}"] = {cur_step: pos}

    total_agents = len(data_pos) - 1
    agent_steps_count = [] # count how long agent lives
    avg_speed = []
    for k, v in data_pos.items():
        agent_steps_count.append(len(v))
        prev_time = None
        for timestep in v.keys():
            if prev_time is None:
                prev_time = timestep
                continue
            if timestep == (prev_time + 1):
                zzz = np.sqrt((v[timestep][0] - v[prev_time][0])**2 + (v[timestep][1] - v[prev_time][1])**2)
                avg_speed.append(zzz)
                prev_time = timestep

    return connect_finish_at, time_car_state,  agent_array, agent_counts, total_agents, agent_steps_count, avg_speed

if __name__ == "__main__":
    folder_path = FILE_PATH

    total_steps, total_times, total_trials, total_node_expansions, total_node_in_total, tree_depth, prediction_time,\
        total_execution_steps, total_running_time_each_scenario = get_statistics_search_tree(folder_path)

    get_statistics_results(folder_path)

    get_auxiliary_results(folder_path)

    fig, axs = plt.subplots(2, 2, figsize=(20,10))

    # For total time steps
    x = np.arange(len(total_steps))
    # axs[0][0].hist(total_steps, bins=30)
    # axs[0][0].title.set_text("Histogram of total Motion Prediction Steps Per Tree Construction")
    #
    # axs[0][1].hist(tree_depth, bins=20)
    # axs[0][1].title.set_text("Histogram of max tree depth Per Tree Construction")
    #
    # axs[1][0].hist(total_trials, bins=30)
    # axs[1][0].title.set_text("Histogram of total trials Per Tree Construction")
    #
    # axs[1][1].hist(total_node_expansions, bins=30)
    # axs[1][1].title.set_text("Histogram of node expansions Per Tree Construction")
    #
    # plt.legend()
    # plt.show()

    print(f" Total Motion Prediction Steps: and avg: {np.mean(np.array(total_steps))}")
    print(f" Time for each Search (One Tree Construction):and avg: {np.mean(np.array(total_times))}")
    print(f" Max tree depth each Search (One Tree Construction):  and avg: {np.mean(np.array(tree_depth))}")
    print(f" Trials: {np.mean(np.array(total_trials))}")
    print(f" Node expansions: {np.mean(np.array(total_node_expansions))}")
    print(f" Total of Node: {np.mean(np.array(total_node_in_total))}")
    print(f" Pred time average: {np.mean(np.array(prediction_time))} ")
    print(f" Execution steps average: {np.mean(np.array(total_execution_steps))} ")
    print(f" Running time of scenario average: {np.mean(np.array(total_running_time_each_scenario))} ")

