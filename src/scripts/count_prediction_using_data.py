import matplotlib.pyplot as plt
import os
import numpy as np
import fnmatch
import sys

file_path = f'/home/cunjun/driving_data_knn/result/' \
            'joint_pomdp_drive_mode/magic/pomdp_search_log-0_0_pid-26829_r-5906418.txt'

folder_path = '/home/cunjun/driving_data_benchmark/ca1/result/' \
              'joint_pomdp_drive_mode/'

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


# This is to decide using 1 file or list of files:

# Use this line for 1 file
#file_path_list = [file_path]

# Use this line for multi files
file_path_list = filter_txt_files(folder_path, collect_txt_files(folder_path, flag="", ignore_flag="nothing"))

total_steps = []
total_times = []
total_trials = []
total_node_expansions = []
total_node_in_total = []
tree_depth = []

prediction_time = []

for file_path in file_path_list:
    with open(file_path, 'r') as f:
        start_reading = False

        sum_steps_per_tree_construction = 0
        for line in f.readlines():
            if 'Construct tree' in line:
                start_reading = True # to start reading from beginning

            if start_reading and "ContextPomdp::Step 123" in line:
                sum_steps_per_tree_construction += 1

            if "deleting tree" in line:
                total_steps.append(sum_steps_per_tree_construction)
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


fig, axs = plt.subplots(2, 2, figsize=(20,10))

# For total time steps
x = np.arange(len(total_steps))
axs[0][0].hist(total_steps, bins=30)
axs[0][0].title.set_text("Histogram of total Motion Prediction Steps Per Tree Construction")

axs[0][1].hist(tree_depth, bins=20)
axs[0][1].title.set_text("Histogram of max tree depth Per Tree Construction")

axs[1][0].hist(total_trials, bins=30)
axs[1][0].title.set_text("Histogram of total trials Per Tree Construction")

axs[1][1].hist(total_node_expansions, bins=30)
axs[1][1].title.set_text("Histogram of node expansions Per Tree Construction")

plt.legend()
plt.show()

print(f" Total Motion Prediction Steps: and avg: {np.mean(np.array(total_steps))}")
print(f" Time for each Search (One Tree Construction):and avg: {np.mean(np.array(total_times))}")
print(f" Max tree depth each Search (One Tree Construction):  and avg: {np.mean(np.array(tree_depth))}")
print(f" Trials: {np.mean(np.array(total_trials))}")
print(f" Node expansions: {np.mean(np.array(total_node_expansions))}")
print(f" Pred time average: {np.mean(np.array(prediction_time))} ")
#print(f" time per action steps (tree construction) average: {np.mean(np.array(total_times))}")
