import matplotlib.pyplot as plt
import os
import numpy as np
import fnmatch
import sys
import math
import matplotlib as mpl

# This class read each line and returns array
class GetTimeSyncAndAgentsInfo():
    def __init__(self):
        # just for drawing
        self.exp_names = {}

        self.connect_finish_at = []# One value per file
        self.time_car_states = [] # A list per file (time when controller get info of ego agent)
        self.time_agent_arrays = [] # A list per file (time when controller get info of exo agents)
        self.agent_counts = [] # A list per file (number of agents per time when controller get info)
        self.total_agents = [] # One value per file (how many agents total in the whole episode)
        self.agent_steps_counts = [] # A list per file (how long does each agent live)
        self.agent_speeds = [] # A list per file (average velocity of each agent)

        # these variables is for reading each file
        self.connect_finish = 0.0 # A value
        self.time_car_state = [] # a list of time. Length corresponds to how many times controller receive ego
        self.time_agent_array = [] # a list of time. Length corresponds to how many times controller receive exo
        self.agent_count = [] # a list of agent. Length corresponds to how many times controller receive exo
        self.total_agent = 0 # A value per file
        self.agent_step_count = [] # duration of each agents. Length corresponds to how many agents
        self.agent_speed = [] # speed of each agent. Length corresponds to how many agents

        # Temporary variables
        self.new_read = True
        self.data_pos = {}
        self.cur_step = 0

    def read(self, line):
        # Resetting for a new file
        if self.new_read:
            self.data_pos = {}
            self.cur_step = 0

            self.connect_finish = 0.0 # A value
            self.time_car_state = [] # a list of time. Length corresponds to how many times controller receive ego
            self.time_agent_array = [] # a list of time. Length corresponds to how many times controller receive exo
            self.agent_count = [] # a list of agent. Length corresponds to how many times controller receive exo
            self.total_agent = 0 # A value per file
            self.agent_step_count = [] # duration of each agents. Length corresponds to how many agents
            self.agent_speed = [] # speed of each agent. Length corresponds to how many agents

            self.new_read = False

        if 'executing step' in line:
            line_1 = line.split('executing step ', 1)[1]
            self.cur_step = int(line_1.split('=', 1)[0])
        elif 'Round 0 Step' in line:
            line_1 = line.split('Round 0 Step ', 1)[1]
            self.cur_step = int(line_1.split('-', 1)[0])
        elif 'goal reached at step' in line:
            line_1 = line.split('goal reached at step ', 1)[1]
            self.cur_step = int(line_1.split(' ', 1)[0])
        elif "car pos / heading / vel" in line:
            # = (149.52, 171.55) / 1.3881 / 0.50245
            speed = float(line.split(' ')[12])
            pos_x = float(line.split(' ')[7].replace('(', '').replace(',', ''))
            pos_y = float(line.split(' ')[8].replace(')', '').replace(',', ''))
            pos = [pos_x, pos_y]


            if "av" in self.data_pos:
                val = self.data_pos["av"]
                val[self.cur_step] = pos
            else:
                self.data_pos["av"] = {self.cur_step: pos}
        elif "id / pos / speed / vel" in line:
            agentid = int(line.split()[16].replace('(', '').replace(',', ''))
            pos_x = float(line.split()[18].replace('(', '').replace(',', ''))
            pos_y = float(line.split()[19].replace(')', '').replace(',', ''))
            pos = [pos_x, pos_y]

            if f"agent{agentid}" in self.data_pos:
                val = self.data_pos[f"agent{agentid}"]
                val[self.cur_step] = pos
            else:
                self.data_pos[f"agent{agentid}"] = {self.cur_step: pos}

        if "Connect finished at" in line:
            self.connect_finish = float(line.split()[5][:-2])

        if "get car state at" in line:
            try:
                t = float(line.split()[5].split("=")[-1])
                self.time_car_state.append(t)
            except:
                pass



        if "MSG: receive" in line and "agents at time" in line:
            t = float(line.split()[6][:-1])
            self.time_agent_array.append(t)
            count = int(line.split()[2])
            self.agent_count.append(count)

    def finish(self, exp_name):
        self.new_read = True

        # Resetting for a new experiment folder
        if exp_name not in self.exp_names.keys():
            self.connect_finish_at = []# One value per file
            self.time_car_states = [] # A list per file (time when controller get info of ego agent)
            self.time_agent_arrays = [] # A list per file (time when controller get info of exo agents)
            self.agent_counts = [] # A list per file (number of agents per time when controller get info)
            self.total_agents = [] # One value per file (how many agents total in the whole episode)
            self.agent_steps_counts = [] # A list per file (how long does each agent live)
            self.agent_speeds = [] # A list per file (average velocity of each agent)


        self.exp_names.setdefault(exp_name, None)

        self.time_agent_arrays.append(self.time_agent_array)
        self.time_car_states.append(self.time_car_state)
        self.agent_counts.append(self.agent_count)
        self.connect_finish_at.append(self.connect_finish)

        self.total_agents.append(len(self.data_pos) - 1)

        agent_steps_count = [] # count how long agent lives
        agent_speed = []
        for k, v in self.data_pos.items():
            agent_steps_count.append(len(v))
            prev_time = None
            avg_speed = []

            for timestep in v.keys():
                if prev_time is None:
                    prev_time = timestep
                    continue

                if timestep == (prev_time + 1):
                    zzz = np.sqrt((v[timestep][0] - v[prev_time][0])**2 + (v[timestep][1] - v[prev_time][1])**2)

                    avg_speed.append(zzz)
                prev_time = timestep

            if not math.isnan(np.mean(np.array(avg_speed))):
                agent_speed.append(np.mean(np.array(avg_speed)))
            self.agent_speeds.append(agent_speed)

        self.agent_steps_counts.append(np.array(agent_steps_count))


        self.exp_names[exp_name] = {
            'connect_finish_at': self.connect_finish_at,
            'time_car_states': self.time_car_states ,
            'time_agent_arrays': self.time_agent_arrays,
            'agent_counts': self.agent_counts,
            'total_agents': self.total_agents,
            'agent_steps_counts': self.agent_steps_counts,
            'agent_speeds': self.agent_speeds
        }


    def draw(self):

        number_of_rows = len(self.exp_names)

        fig, axs = plt.subplots(3, 3, figsize=(10,20)) # we plot 4 values so we need 2 times 2. Can be adjusted if draw more than that

        for i in range(number_of_rows):
            key = list(self.exp_names)[i]

            #axs[0][0].boxplot(self.exp_names[key]['connect_finish_at'], positions=[i])
            axs[0][0].violinplot(self.exp_names[key]['connect_finish_at'], positions=[i],
                                 showmeans=False, showmedians=True, showextrema=True)

            axs[0][1].violinplot(self.exp_names[key]['total_agents'], positions=[i],
                                 showmeans=False, showmedians=True, showextrema=True)

            arrays = []
            for z in range(len(self.exp_names[key]['time_car_states'])):
                vals = np.array(self.exp_names[key]['time_car_states'][z][1:-1]) - \
                               np.array(self.exp_names[key]['time_car_states'][z][0:-2])
                arrays.extend(list(vals))
            axs[0][2].violinplot(arrays, positions=[i],
                                 showmeans=False, showmedians=True, showextrema=True)

            arrays = []
            for z in range(len(self.exp_names[key]['time_agent_arrays'])):
                vals = np.array(self.exp_names[key]['time_agent_arrays'][z][1:-1]) - \
                       np.array(self.exp_names[key]['time_agent_arrays'][z][0:-2])
                arrays.extend(list(vals))
            axs[1][0].violinplot(arrays, positions=[i],
                                 showmeans=False, showmedians=True, showextrema=True)

            arrays = []
            for z in range(len(self.exp_names[key]['agent_counts'])):
                arrays.extend(self.exp_names[key]['agent_counts'][z])
            axs[1][1].violinplot(arrays, positions=[i],
                                 showmeans=False, showmedians=True, showextrema=True)

            arrays = []
            for z in range(len(self.exp_names[key]['agent_steps_counts'])):
                arrays.extend(self.exp_names[key]['agent_steps_counts'][z])
            axs[1][2].violinplot(arrays, positions=[i],
                                 showmeans=False, showmedians=True, showextrema=True)

            arrays = []
            for z in range(len(self.exp_names[key]['agent_speeds'])):
                arrays.extend(self.exp_names[key]['agent_speeds'][z])
            axs[2][0].violinplot(arrays, positions=[i],
                                 showmeans=False, showmedians=True, showextrema=True)

            #axs[1][2].boxplot(self.exp_names[key]['agent_steps_counts'], positions=[i])
            #axs[2][0].boxplot(self.exp_names[key]['agent_speeds'], positions=[i])


        axs[0][0].set_title("Connect time finished")
        axs[0][1].set_title("Number of total agents in episode")
        axs[0][2].set_title("Average difference of time car states")
        axs[1][0].set_title("Average difference of time agent array")
        axs[1][1].set_title("Average of agent_counts per time")
        axs[1][2].set_title("Duration of each agents in episode")
        axs[2][0].set_title("Agent speed")

        # add x-tick labels
        # plt.setp(axs, xticks=[y for y in range(len(self.exp_names))],
        #          xticklabels=self.exp_names.keys())

        plt.setp(axs, xticks=[])
        fig.legend(labels=self.exp_names.keys(), loc='lower center', ncol=3)
        print([ax.get_legend_handles_labels() for ax in fig.axes])
        #fig.tight_layout()
        plt.legend()
        plt.show()


if __name__ == "__main__":
    from average_statistics_of_1_data import filter_txt_files, collect_txt_files

    FOLDER_PATHS = [
        '/home/cunjun/driving_data/JAN/gamma_tick_3Hz_ts0_1_8e3/',
        '/home/cunjun/driving_data/JAN/gamma_tick_3Hz_ts0_1_8e3_allHzscale/',
        '/home/cunjun/driving_data/JAN/gamma_tick_3Hz_ts0_1_8e4/',
        '/home/cunjun/driving_data/JAN/gamma_tick_3Hz_ts0_1_8e4_allHzscale/',
        '/home/cunjun/driving_data/JAN/gamma_tick_3Hz_ts1_8e4/',
        '/home/cunjun/driving_data/JAN/gamma_tick_30Hz_ts1_8e4/'
    ]

    FOLDER_PATHS = [
        #'/home/cunjun/driving_data/JAN/gamma_tick_3Hz_ts0_1_8e3_allHzscale/',
        #'/home/cunjun/driving_data/JAN/gamma_tick_3Hz_ts0_1_8e4_allHzscale/',
        '/home/cunjun/driving_data/JAN/gamma_tick_3Hz_ts0_1_8e4/',
        '/home/cunjun/driving_data/JAN/gamma_tick_3Hz_ts0_1_8e4_allHzscale/',
        #'/home/cunjun/driving_data/DEL/knndefault_3Hz_ts0_1_allHzscale/',
        #'/home/cunjun/driving_data/DEL/knnsocial_3Hz_ts0_1_allHzscale/',
        #'/home/cunjun/driving_data/DEL/lstmdefault_3Hz_ts0_1_allHzscale/',
        #'/home/cunjun/driving_data/DEL/lstmsocial_3Hz_ts0_1_allHzscale/',
        '/home/cunjun/driving_data/DEL/cv15hz_t0_5/',
        '/home/cunjun/driving_data/DEL/cv10hz_t0_33/',
        '/home/cunjun/driving_data/DEL/ca10hz_t0_33/',
        '/home/cunjun/driving_data/DEL/cv10hz_t0_33_slowdownHz3times/',
        '/home/cunjun/driving_data/DEL/ca10hz_t0_33_slowdownHz3times/',

    ]

    stats = GetTimeSyncAndAgentsInfo()
    for folder_path in FOLDER_PATHS:
        file_path_list = filter_txt_files(folder_path,
                                      collect_txt_files(folder_path, flag="", ignore_flag="nothing"))

        experiment_name = folder_path.split('/')[5]

        for file_path in file_path_list:
            with open(file_path, 'r') as f:

                for line in f.readlines():
                    stats.read(line)

            stats.finish(experiment_name)

    stats.draw()