import os
import sys
import pdb
import math
import pickle
import random
import argparse
import numpy as np
import matplotlib
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import pandas as pd
import statsmodels.api as sm
import numpy as np
import textwrap
from matplotlib import colors
from scipy import stats
from math import pi
from common_performance_stats import get_dynamic_ade

# Weights for each metric within a category
safety_weights = {
    'collision_rate': 0.0,
    'near_miss_rate': 1.0,
    'near_distance_rate': 0.0,
    'mean_min_ttc': 0.0,
    'mean_min_ttr': 0.0
}
safety_directions = {
    'collision_rate': 'lower',
    'near_miss_rate': 'lower',
    'near_distance_rate': 'lower',
    'mean_min_ttc': 'higher',
    'mean_min_ttr': 'higher'
}
comfort_weights = {
    'jerk': 1.0,
    'lateral_acceleration': 0.0,
    'acceleration': 0.0
}
comfort_directions = {
    'jerk': 'lower',
    'lateral_acceleration': 'lower',
    'acceleration': 'lower'
}
efficiency_weights = {
    'avg_speed': 1.0,
    'tracking_error': 0.0,
    'efficiency_time': 0.0,
    'distance_traveled': 0.0, 
}
efficiency_directions = {
    'avg_speed': 'higher',
    'tracking_error': 'lower',
    'efficiency_time': 'lower',
    'distance_traveled': 'higher'
}

color = 'red'

pred_len = 30

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze")

    # Mode 'data' is to generate data from .txt files
    # Mode 'performance' is to read data pickle file to generate prediction and performance, tree metrics
    # Mode 'plot' is to read from metric pickle file and plot
    parser.add_argument('--mode', help='Generate file or only plot the relation', required=True)

    args = parser.parse_args()

    return args

def save_dict_to_file(dictionary, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(dictionary, f)

# Load the dictionary from a file
def load_dict_from_file(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def remove_outliers_iqr(X, y1, multiplier=1):
    """
    Remove outliers using the interquartile range (IQR) method.
    
    Args:
        X (list): Independent variable.
        y (list): Dependent variable.
        multiplier (float): Multiplier to determine the threshold for outlier detection.
            Default is 1.5, which is a common value used in practice.
    
    Returns:
        X_clean (list): Independent variable with outliers removed.
        y_clean (list): Dependent variable with outliers removed.
    """
    # Convert X and y lists to numpy arrays
    X = np.array(X)
    y1 = np.array(y1)
    
    # Calculate the IQR for X and y
    Q1_X = np.percentile(X, 25)
    Q3_X = np.percentile(X, 75)
    IQR_X = Q3_X - Q1_X

    Q1_y1 = np.percentile(y1, 25)
    Q3_y1 = np.percentile(y1, 75)
    IQR_y1 = Q3_y1 - Q1_y1


    # Define the upper and lower bounds for outlier detection
    upper_bound_X = Q3_X + multiplier * IQR_X
    lower_bound_X = Q1_X - multiplier * IQR_X

    upper_bound_y1 = Q3_y1 + multiplier * IQR_y1
    lower_bound_y1 = Q1_y1 - multiplier * IQR_y1

    # Filter the data to keep only the values within the bounds
    X_clean = X[(X >= lower_bound_X) & (X <= upper_bound_X) & 
                (y1 >= lower_bound_y1) & (y1 <= upper_bound_y1) & (y1!=0)]
    y1_clean = y1[(X >= lower_bound_X) & (X <= upper_bound_X) & 
                (y1 >= lower_bound_y1) & (y1 <= upper_bound_y1) & (y1!=0)]

    return X_clean, y1_clean

def remove_nans(list_of_lists):
    
    # Initialize an empty list to hold the indices of elements to remove
    indices_to_remove = []

    # Iterate through each inner list and find indices containing nan values
    for inner_list in list_of_lists:
        for i, element in enumerate(inner_list):
            if math.isnan(element):
                indices_to_remove.append(i)

    # Remove duplicate indices from the list and reverse sort it
    indices_to_remove = sorted(set(indices_to_remove), reverse=True)

    # Iterate through the list of indices to remove and remove elements from the inner lists
    for i in indices_to_remove:
        for inner_list in list_of_lists:
            del inner_list[i]

    return list_of_lists

def normalize(arr, max_val = 1.0, min_val = 0.0):
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    
    return (arr - np.array(min_val)) / (np.array(max_val) - np.array(min_val)+1e-6)

def weighted_average(data, weights, directions, max_min_list):
    norm_data = [normalize(arr, max_min[0], max_min[1]) if direction == 'higher'
                    else 1 - normalize(arr, max_min[0], max_min[1]) for arr, direction, max_min in zip(data, directions, max_min_list)]
    return np.sum([w * d for w, d in zip(weights, norm_data)], axis=0)

def weighted_average_no_normalize(data, weights, directions, max_min_list):
    norm_data = [normalize(arr, 1, 0) for arr, direction, max_min in zip(data, directions, max_min_list)]
    return np.sum([w * d for w, d in zip(weights, norm_data)], axis=0)

def find_max_min(dpm, methods_to_plot):
    max_min_method = {}
    a_method_in_dpm = list(dpm.keys())[0]
    for category in dpm[a_method_in_dpm].keys():
        max_min_method[category] = {}
        for metric in dpm[a_method_in_dpm][category].keys():
            max_min_method[category][metric] = {'max':-np.inf, 'min':np.inf}
    
    for i, method in enumerate(methods_to_plot):
        for category in dpm[method].keys():
            for metric in dpm[method][category].keys():
                max_min_method[category][metric]['max'] = max(np.nanmax(dpm[method][category][metric]), 
                                                                max_min_method[category][metric]['max'])
                max_min_method[category][metric]['min'] = min(np.nanmin(dpm[method][category][metric]),
                                                                max_min_method[category][metric]['min'])   
    return max_min_method

def get_normalized_data(method, prediction_performance, driving_performance, max_min_method, prediction_metric):

    # Compute the weighted averages for each category
    max_min_safety = []
    w = []
    d = []
    assert len(list(max_min_method['safety'].keys())) == 5, "Must be 5"
    assert list(max_min_method['safety'].keys()) == ["collision_rate", "near_miss_rate", 
                                                     "near_distance_rate", "mean_min_ttc" , "mean_min_ttr"]
    for k in ["collision_rate", "near_miss_rate", "near_distance_rate", "mean_min_ttc" , "mean_min_ttr"]:
        max_min_safety.append(tuple(max_min_method['safety'][k].values()))
        w.append(safety_weights[k])
        d.append(safety_directions[k])

    safety_data = weighted_average(
        [driving_performance[method]['safety']['collision_rate'],
        driving_performance[method]['safety']['near_miss_rate'],
        driving_performance[method]['safety']['near_distance_rate'],
        driving_performance[method]['safety']['mean_min_ttc'],
        driving_performance[method]['safety']['mean_min_ttr']],
        w, d, max_min_safety)

    max_min_comfort = []
    w = []
    d = []
    assert len(list(max_min_method['comfort'].keys())) == 3, "Must be 3"
    assert list(max_min_method['comfort'].keys()) == ["jerk", "lateral_acceleration", "acceleration"]
    for k in ["jerk", "lateral_acceleration", "acceleration"]:
        max_min_comfort.append(tuple(max_min_method['comfort'][k].values()))
        w.append(comfort_weights[k])
        d.append(comfort_directions[k])
    
    comfort_data = weighted_average(
        [driving_performance[method]['comfort']['jerk'],
        driving_performance[method]['comfort']['lateral_acceleration'],
        driving_performance[method]['comfort']['acceleration']],
        w, d, max_min_comfort)

    max_min_efficiency = []
    w = []
    d = []
    assert len(list(max_min_method['efficiency'].keys())) == 4, "Must be 4"
    assert list(max_min_method['efficiency'].keys()) == ["avg_speed", "tracking_error", "efficiency_time", "distance_traveled"]
    for k in ["avg_speed", "tracking_error", "efficiency_time", "distance_traveled"]:
        max_min_efficiency.append(tuple(max_min_method['efficiency'][k].values()))
        w.append(efficiency_weights[k])
        d.append(efficiency_directions[k])
    
    efficiency_data = weighted_average(
        [driving_performance[method]['efficiency']['avg_speed'],
        driving_performance[method]['efficiency']['tracking_error'],
        driving_performance[method]['efficiency']['efficiency_time'],
        driving_performance[method]['efficiency']['distance_traveled']],
        w, d, max_min_efficiency)
    
    prediction_data = np.array(prediction_performance[method][prediction_metric])

    driving_performance_data = (efficiency_data+safety_data+comfort_data)/3

    return prediction_data, safety_data, comfort_data, efficiency_data, driving_performance_data



def plot_static_ade_driving_performance_radar(prediction_performance_map, 
                                                   driving_performance_map, metric='minFDE'):

    
    performance_summit_ade = np.array([2.938, 2.989, 3.099, 3.196, 1.692, 1.944, 2.410, 2.480])
    performance_summit_fde = np.array([5.950, 6.040, 6.505, 6.633, 3.524, 4.189, 5.061, 5.152])

    performance_summit_made = np.array([1.965, 2.003, 1.667, 1.740, 0.807, 1.016, 2.348, 2.396])
    performance_summit_mfde = np.array([3.962, 3.986, 2.742, 2.853, 1.064, 1.503, 4.924, 4.954])
    
    if metric == "ADE":
        performance_summit = performance_summit_ade
    elif metric == "FDE":
        performance_summit = performance_summit_fde
    elif metric == "minADE":
        performance_summit = performance_summit_made
    elif metric == "minFDE":
        performance_summit = performance_summit_mfde

    methods_to_plot = ['cv', 'ca', 'knndefault', 'knnsocial', 'hivt', 'lanegcn', 'lstmdefault', 'lstmsocial']
    method_name = ['CV', 'CA', 'KNN', 'S-KNN', 'HiVT', 'LaneGCN', 'LSTM', 'S-LSTM']
    
    method_performance = get_method_performance(prediction_performance_map, driving_performance_map, 
                                                     methods_to_plot, "ade")
    
    # Extract values from the performance dictionary
    efficiency = [method_performance[method]['efficiency'] for method in methods_to_plot]
    comfort = [method_performance[method]['comfort'] for method in methods_to_plot]
    safety = [method_performance[method]['safety'] for method in methods_to_plot]
    driving = [method_performance[method]['driving'] for method in methods_to_plot]

    performance_summit_ade_inverse = 1/performance_summit_ade
    performance_summit_fde_inverse = 1/performance_summit_fde

    print(performance_summit_ade_inverse)

    data = [performance_summit_ade_inverse, performance_summit_fde_inverse, driving, efficiency, comfort, safety]
    import pdb;pdb.set_trace()

    normalized_data = []
    index = 0
    for metric in data:
        metric_min = min(metric)
        metric_max = max(metric)
        index+=1
        normalized_data.append([(value - metric_min) / (metric_max - metric_min) + 0.2 for value in metric])

    print(normalized_data)

    # Create radar plot
    num_vars = len(normalized_data)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += [angles[0]]

    # Helper function to plot a method on a radar plot
    def plot_method(ax, angles, data, color, label):
        data += [data[0]]
        ax.plot(angles, data, color=color, linewidth=2, label=label)
        ax.fill(angles, data, color=color, alpha=0.1)
    
    # plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 16})
    
    for i, method in enumerate(methods_to_plot):

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
        # fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['1/ADE','1/FDE', 'Driving', 'Effi', 'Comfort', 'Safety'], color='black', fontsize=24)
        ax.tick_params(axis='x', which='major', pad=24, labelsize=24, labelcolor='black')
        ax.set_yticklabels([])
        plot_method(ax, angles, [row[i] for row in normalized_data], 'blue', method)
        ax.set_title(method_name[i], y=1.1, fontsize=40, color='black')
        ax.set_ylim(0, 1.25)
        plt.tight_layout()
        plt.savefig(f"0_Radarplot_{i}.pdf", dpi = 300)
        plt.clf()


    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    plt.grid(True)
    for i, method in enumerate(methods_to_plot):
        ax.scatter(performance_summit_ade[i], driving[i], color = 'red', s=50)
        ax.text(performance_summit_ade[i], driving[i], method, fontsize=16)
    
    sorted_indices = sorted(range(len(performance_summit_ade)), key=lambda k: performance_summit_ade[k])
    sorted_array1 = [performance_summit_ade[i] for i in sorted_indices]
    sorted_array2 = [driving[i] for i in sorted_indices]

    from scipy.interpolate import interp1d

    x_smooth = np.linspace(min(sorted_array1), max(sorted_array1), 100)

    p = np.polyfit(sorted_array1, sorted_array2, deg=2)
    y_smooth = np.poly1d(p)(x_smooth)

    ax.plot(x_smooth, y_smooth, color = 'red', label = "Reality")
    ax.plot(sorted_array1, [(23.5-sorted_array1[i])/28 for i in range(len(sorted_array1))], color = 'black', label = "Assumption")

    ax.set_xlabel("ADE", fontsize=24)
    ax.set_ylabel("Driving Performance", fontsize=24)
    ax.set_title('ADE vs Driving Performance', y=1.1, fontsize=40, color='black')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"0_ade_vs_driving.pdf", dpi = 300)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    plt.grid(True)
    for i, method in enumerate(methods_to_plot):
        ax.scatter(performance_summit_fde[i], driving[i], color = 'red', s=50)
        ax.text(performance_summit_fde[i], driving[i], method, fontsize=16)
    
    sorted_indices = sorted(range(len(performance_summit_fde)), key=lambda k: performance_summit_fde[k])
    sorted_array1 = [performance_summit_fde[i] for i in sorted_indices]
    sorted_array2 = [driving[i] for i in sorted_indices]

    from scipy.interpolate import interp1d

    x_smooth = np.linspace(min(sorted_array1), max(sorted_array1), 100)

    p = np.polyfit(sorted_array1, sorted_array2, deg=2)

    y_smooth = np.poly1d(p)(x_smooth)

    ax.plot(x_smooth, y_smooth, color = 'red', label = "Reality")
    ax.plot(sorted_array1, [(51.5-sorted_array1[i])/62 for i in range(len(sorted_array1))], color = 'black', label = "Assumption")

    ax.set_xlabel("FDE", fontsize=24)
    ax.set_ylabel("Driving Performance", fontsize=24)
    ax.set_title('FDE vs Driving Performance', y=1.10, fontsize=40, color='black')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"0_fde_vs_driving.pdf", dpi = 300)


def plot_dynamic_ade_vs_driving_performance(prediction_performance_map, 
                                            driving_performance_map, metric='minFDE'):
    
    methods_to_plot = ['cv', 'ca', 'knndefault', 'knnsocial', 'hivt', 'lanegcn', 'lstmdefault', 'lstmsocial']
    r_values_dict = {}

    # Find the best-fit-line
    for best_fit_metric in prediction_performance_map['cv'].keys():
        if "ade" not in best_fit_metric and "fde" not in best_fit_metric:
            continue
        driving_performance = get_method_performance(prediction_performance_map, driving_performance_map, 
                                                        methods_to_plot, best_fit_metric)
        dynamic_metric = [np.nanmean(prediction_performance_map[key][best_fit_metric]) for key in methods_to_plot]

        x = dynamic_metric
        y = driving_performance

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Store the r-value in the dictionary
        r_values_dict[best_fit_metric] = r_value

        # Print the r-value for the current method
        print(f"Method: {best_fit_metric}, R-value: {r_value} p-value {p_value}")

    # Sort the methods based on their r-values (ascending order)
    sorted_r_values = sorted(r_values_dict.items(), key=lambda x: x[1], reverse=False)

    # Print the best, 2nd best, and worst methods based on r-values
    for i in range(10):
        print(f"{i}th method: {sorted_r_values[i][0]}, R-value: {sorted_r_values[i][1]}")
    print(f"Worst method: {sorted_r_values[-1][0]}, R-value: {sorted_r_values[-1][1]}")

    
    ## Dynamic ADE vs Driving Performance##
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)

    metric = 'fde_20meters_closest_predlen30'
    driving_performance = get_method_performance(prediction_performance_map, driving_performance_map, 
                                                        methods_to_plot, 'ade')
    dynamic_metric = [np.nanmean(prediction_performance_map[key][metric]) for key in methods_to_plot]
   
    # Create a scatter plot with different colors for each point
    colors = plt.cm.rainbow(np.linspace(0, 1, len(methods_to_plot)))
    fig, ax = plt.subplots(figsize=(5, 5)) # adjust figure size as needed

    for i, method in enumerate(methods_to_plot):
        plt.scatter(dynamic_metric[i], driving_performance[i], color=colors[i], label=method)
        plt.annotate(method, (dynamic_metric[i], driving_performance[i]), 
                     textcoords="offset points", xytext=(-2,7), ha='center')
    print('xx')

    # Add labels and legend
    plt.xlabel(f'Dynamic {metric}')
    plt.ylabel('Driving Performance')
    plt.ylim(top=max(driving_performance)*1.01)  # Adjust the ylim to provide more space for the legend
    plt.xlim(right=max(dynamic_metric)*1.06)
    #plt.xlim(left  = min(min(performance_summit_ade), min(performance_summit_fde), min(performance_summit_made), min(performance_summit_mfde))*0.95,
    #         right = max(max(performance_summit_ade), max(performance_summit_fde), max(performance_summit_made), max(performance_summit_mfde))*1.05)
    plt.title(f'Relation of Dynamic {metric} \n vs Driving Performance of RVO Planner')

    ax = plot_best_fit_line(dynamic_metric, driving_performance, ax)

    plt.tight_layout()
    
    plt.savefig(f"5_Dynamic_{metric}_vs_Driving_Performance_RVO.png")
    plt.clf()


def get_method_performance(prediction_performance, driving_performance, methods_to_plot, prediction_metric):  

    max_min_method = find_max_min(driving_performance, methods_to_plot)
    method_performance = {}
    for i, method in enumerate(methods_to_plot):
        _, safety_data, comfort_data, efficiency_data, performance = get_normalized_data(method, prediction_performance, 
                        driving_performance, max_min_method, prediction_metric)
        
        method_performance[method] = {'safety': np.nanmean(safety_data), 
                                      'comfort': np.nanmean(comfort_data),
                                 'efficiency': np.nanmean(efficiency_data), 
                                 'driving': np.nanmean(performance)}

    return method_performance

if __name__ == "__main__":

    args = parse_args()

    directories_map_RVO = {
            'cv': '/home/phong/driving_data/official/gamma_planner_path1_vel3/cv',
            'ca': '/home/phong/driving_data/official/gamma_planner_path1_vel3/ca',
            'hivt': '/home/phong/driving_data/official/gamma_planner_path1_vel3/hivt',
            'lanegcn': '/home/phong/driving_data/official/gamma_planner_path1_vel3/lanegcn',
            'lstmdefault': '/home/phong/driving_data/official/gamma_planner_path1_vel3/lstmdefault',
            'lstmsocial': '/home/phong/driving_data/official/gamma_planner_path1_vel3/lstmsocial5Hz',
            'knnsocial': '/home/phong/driving_data/official/gamma_planner_path1_vel3/knnsocial',
            'knndefault': '/home/phong/driving_data/official/gamma_planner_path1_vel3/knndefault'
        }

    prediction_performance_RVO = {}
    driving_performance_RVO = {}
    algorithm_performance_RVO = {}
    
    # Storing to a pickle file
    if  args.mode == "data":

        for key in directories_map_RVO.keys():
            print(f"Processing {key}")
            prediction_performance, driving_performance, tree_performance = get_dynamic_ade(directories_map_RVO[key])
            prediction_performance_RVO[key] = prediction_performance
            driving_performance_RVO[key] = driving_performance
            algorithm_performance_RVO[key] = tree_performance

            save_dict_to_file(prediction_performance_RVO, f'pickle_filesrvo/prediction_performance_RVO_{key}.pickle')
            save_dict_to_file(driving_performance_RVO, f'pickle_filesrvo/driving_performance_RVO_{key}.pickle')
            save_dict_to_file(algorithm_performance_RVO, f'pickle_filesrvo/algorithm_performance_RVO_{key}.pickle')
            print(f"Finish saving to pickle file. Exit")
        
        exit(0)

    elif args.mode == "plot":
        for key in directories_map_RVO.keys():
            prediction_performance_RVO[key] = load_dict_from_file(f'pickle_filesrvo/prediction_performance_RVO_{key}.pickle')[key]
            driving_performance_RVO[key] = load_dict_from_file(f'pickle_filesrvo/driving_performance_RVO_{key}.pickle')[key]
            algorithm_performance_RVO[key] = load_dict_from_file(f'pickle_filesrvo/algorithm_performance_RVO_{key}.pickle')[key]
    else:
        assert False, f"Not available option form {args.mode}"    

    # Done
    #plot_dynamic_vs_staticade(prediction_performance_RVO, list(prediction_performance_RVO.keys()))

    plot_static_ade_driving_performance_radar(prediction_performance_RVO, driving_performance_RVO)

    #plot_dynamic_ade_vs_driving_performance(prediction_performance_RVO, driving_performance_RVO)

    

