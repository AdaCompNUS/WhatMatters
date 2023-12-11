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
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r'\usepackage{newtxtext,newtxmath}'
import pandas as pd
import statsmodels.api as sm
import numpy as np
import textwrap
from matplotlib import colors
from scipy import stats

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

def get_normalized_data(method, prediction_performance, driving_performance, max_min_method, prediction_metric,
                        safety_w = 1/3.0, efficiency_w = 1/3.0, comfort_w = 1/3.0):

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

    #driving_performance_data = (efficiency_data+safety_data+comfort_data)/3
    driving_performance_data = comfort_w * comfort_data + efficiency_w * efficiency_data + safety_w * safety_data;

    return prediction_data, safety_data, comfort_data, efficiency_data, driving_performance_data


def plot_dynamic_vs_staticade_scatter(prediction_performance_RVO,
                                      prediction_performance_DESPOT, methods_to_plot):

    methods_to_plot = ['CV', 'CA', 'KNN', 'S-KNN', 'HiVT', 'LaneGCN', 'LSTM', 'S-LSTM']
    # Remap for unified name of plotting througout the papers
    remap_methods = {
        'cv': 'CV',
        'ca': 'CA',
        'knndefault': 'KNN',
        'knnsocial': 'S-KNN',
        'hivt': 'HiVT',
        'lanegcn': 'LaneGCN',
        'lstmdefault': 'LSTM',
        'lstmsocial': 'S-LSTM'
    }
    for k, v in remap_methods.items():
        prediction_performance_RVO[v] = prediction_performance_RVO[k]
        del prediction_performance_RVO[k]

        prediction_performance_DESPOT[v] = prediction_performance_DESPOT[k]
        del prediction_performance_DESPOT[k]

    performance_summit_ade = np.array([2.938, 2.989, 3.099, 3.196, 1.692, 1.944, 2.410, 2.480])
    performance_summit_fde = np.array([5.950, 6.040, 6.505, 6.633, 3.524, 4.189, 5.061, 5.152])
    performance_summit_made = np.array([1.965, 2.003, 1.667, 1.740, 0.807, 1.016, 2.348, 2.396])
    performance_summit_mfde = np.array([3.962, 3.986, 2.742, 2.853, 1.064, 1.503, 4.924, 4.954])
    # Using static ADE

    static_metric = 'ADE'
    if static_metric == "ADE":
        performance_summit = performance_summit_ade
    elif static_metric == "FDE":
        performance_summit = performance_summit_fde
    elif static_metric == "minADE":
        performance_summit = performance_summit_made
    elif static_metric == "minFDE":
        performance_summit = performance_summit_mfde

    # This is the scatter plot with best-fit-line for each dynamic metric vs. static
    # All list below are predlen 30
    dynamic_ade_dict_DESPOT = {'ade_predlen30': np.zeros(len(methods_to_plot)), 
                        'ade_obs20_predlen30': np.zeros(len(methods_to_plot)),
                        'ade_20meters_closest_predlen30': np.zeros(len(methods_to_plot))}
    dynamic_ade_dict_RVO = {'ade_predlen30': np.zeros(len(methods_to_plot)), 
                        'ade_obs20_predlen30': np.zeros(len(methods_to_plot)),
                        'ade_20meters_closest_predlen30': np.zeros(len(methods_to_plot))}
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Printing out tree performance
    for key in prediction_performance_DESPOT.keys():
        idx = methods_to_plot.index(key)
        for which_dynamic_ade in dynamic_ade_dict_DESPOT.keys():
            dynamic_ade_dict_DESPOT[which_dynamic_ade][idx] = np.nanmean(prediction_performance_DESPOT[key][which_dynamic_ade])
    for key in prediction_performance_RVO.keys():
        idx = methods_to_plot.index(key)
        for which_dynamic_ade in dynamic_ade_dict_RVO.keys():
            dynamic_ade_dict_RVO[which_dynamic_ade][idx] = np.nanmean(prediction_performance_RVO[key][which_dynamic_ade])
   

    offsets=0.4
    space_between_methods = 1.2
    position = np.arange(len(methods_to_plot)) * (len(methods_to_plot) * offsets / 2 + space_between_methods)

    # Create a scatter plot with different colors for each point
    for which_dynamic_ade in dynamic_ade_dict_RVO.keys():
        colors = plt.cm.rainbow(np.linspace(0, 1, len(methods_to_plot)))
        fig, ax = plt.subplots(figsize=(5, 5)) # adjust figure size as needed

        plt.rcParams.update({'font.size': 12})
        for i, method in enumerate(methods_to_plot):
            #plt.scatter(performance_summit[i], dynamic_ade_dict_RVO[which_dynamic_ade][i], color='black', label=method, marker='o')
            #plt.annotate(method, (performance_summit[i], dynamic_ade_dict_RVO[which_dynamic_ade][i]), 
            #            textcoords="offset points", xytext=(-2,7), ha='center')

            plt.scatter(performance_summit[i], dynamic_ade_dict_DESPOT[which_dynamic_ade][i], color='black', label=method)
            plt.annotate(method, (performance_summit[i], dynamic_ade_dict_DESPOT[which_dynamic_ade][i]), 
                        textcoords="offset points", xytext=(-2,7), ha='center', color='black')

        # Add labels and legend
        plt.xlabel(f'Static ADE', fontsize=12)
        plt.ylabel(f'Dynamic ADE', fontsize=12)
        plt.title("Static ADE vs. Dynamic ADE of DESPOT Planner", fontsize=12)
        plt.ylim(top=max(dynamic_ade_dict_DESPOT[which_dynamic_ade])*1.05)  # Adjust the ylim to provide more space for the legend
        #plt.xlim(left = min(performance_summit) * 0.95, right=max(performance_summit)*1.03)

        ax = plot_best_fit_line(performance_summit, dynamic_ade_dict_DESPOT[which_dynamic_ade], ax, color="blue")
        #ax = plot_best_fit_line(performance_summit, dynamic_ade_dict_DESPOT[which_dynamic_ade], ax, color="red")

        plt.tight_layout()
        plt.grid()
        plt.savefig(f"3_Static_Dynamic_{which_dynamic_ade.upper()}_DESPOT.pdf", dpi=300)
        plt.clf()
        plt.close()


def plot_dynamic_vs_staticade_line(prediction_performance_RVO,
                                      prediction_performance_DESPOT, methods_to_plot):

    methods_to_plot = ['CV', 'CA', 'KNN', 'S-KNN', 'HiVT', 'LaneGCN', 'LSTM', 'S-LSTM']
    # Remap for unified name of plotting througout the papers
    remap_methods = {
        'cv': 'CV',
        'ca': 'CA',
        'knndefault': 'KNN',
        'knnsocial': 'S-KNN',
        'hivt': 'HiVT',
        'lanegcn': 'LaneGCN',
        'lstmdefault': 'LSTM',
        'lstmsocial': 'S-LSTM'
    }
    for k, v in remap_methods.items():
        prediction_performance_RVO[v] = prediction_performance_RVO[k]
        del prediction_performance_RVO[k]

        prediction_performance_DESPOT[v] = prediction_performance_DESPOT[k]
        del prediction_performance_DESPOT[k]

    performance_summit_ade = np.array([2.938, 2.989, 3.099, 3.196, 1.692, 1.944, 2.410, 2.480])
    performance_summit_fde = np.array([5.950, 6.040, 6.505, 6.633, 3.524, 4.189, 5.061, 5.152])
    performance_summit_made = np.array([1.965, 2.003, 1.667, 1.740, 0.807, 1.016, 2.348, 2.396])
    performance_summit_mfde = np.array([3.962, 3.986, 2.742, 2.853, 1.064, 1.503, 4.924, 4.954])
    # Using static ADE

    static_metric = 'ADE'
    if static_metric == "ADE":
        performance_summit = performance_summit_ade
    elif static_metric == "FDE":
        performance_summit = performance_summit_fde
    elif static_metric == "minADE":
        performance_summit = performance_summit_made
    elif static_metric == "minFDE":
        performance_summit = performance_summit_mfde

    # This is the scatter plot with best-fit-line for each dynamic metric vs. static
    # All list below are predlen 30
    dynamic_ade_dict_DESPOT = {'ade_predlen30': np.zeros(len(methods_to_plot)), 
                        'ade_obs20_predlen30': np.zeros(len(methods_to_plot)),
                        'ade_20meters_closest_predlen30': np.zeros(len(methods_to_plot))}
    dynamic_ade_dict_RVO = {'ade_predlen30': np.zeros(len(methods_to_plot)), 
                        'ade_obs20_predlen30': np.zeros(len(methods_to_plot)),
                        'ade_20meters_closest_predlen30': np.zeros(len(methods_to_plot))}
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Printing out tree performance
    for key in prediction_performance_DESPOT.keys():
        idx = methods_to_plot.index(key)
        for which_dynamic_ade in dynamic_ade_dict_DESPOT.keys():
            dynamic_ade_dict_DESPOT[which_dynamic_ade][idx] = np.nanmean(prediction_performance_DESPOT[key][which_dynamic_ade])
    for key in prediction_performance_RVO.keys():
        idx = methods_to_plot.index(key)
        for which_dynamic_ade in dynamic_ade_dict_RVO.keys():
            dynamic_ade_dict_RVO[which_dynamic_ade][idx] = np.nanmean(prediction_performance_RVO[key][which_dynamic_ade])

    # Create a line plot with different colors for each point
    fig, ax = plt.subplots(figsize=(5, 5)) # adjust figure size as needed
    plt.rcParams.update({'font.size': 12})
    method_index = np.arange(len(methods_to_plot))
    ax.plot(method_index, performance_summit, label="Static ADE")
    ax.plot(method_index, dynamic_ade_dict_RVO['ade_predlen30'], label='Dynamic ADE')
    #for which_dynamic_ade in dynamic_ade_dict_RVO.keys():
    #    ax.plot(method_index, dynamic_ade_dict_RVO[which_dynamic_ade], label=which_dynamic_ade)

    # Add labels and legend
    ax.legend()
    plt.xlabel(f'Methods', fontsize=12)
    plt.ylabel(f'ADE', fontsize=12)
    plt.title("Different Dynamic ADE of RVO Planner", fontsize=12)
    #plt.ylim(top=max(dynamic_ade_dict_RVO[which_dynamic_ade])*1.02)  # Adjust the ylim to provide more space for the legend
    #plt.xlim(left = min(performance_summit) * 0.95, right=max(performance_summit)*1.03)

    # Set the x-tick labels to the method names
    ax.set_xticks(method_index)
    ax.set_xticklabels(methods_to_plot)

    plt.tight_layout()
    plt.grid()
    plt.savefig(f"C_Static_vs_Dynamic_ADE.pdf", dpi=300)
    plt.clf()
    plt.close()

def plot_dynamic_vs_staticade_bar(prediction_performance_RVO,
                                      prediction_performance_DESPOT, methods_to_plot):

    methods_to_plot = ['CV', 'CA', 'KNN', 'S-KNN', 'HiVT', 'LaneGCN', 'LSTM', 'S-LSTM']
    # Remap for unified name of plotting througout the papers
    remap_methods = {
        'cv': 'CV',
        'ca': 'CA',
        'knndefault': 'KNN',
        'knnsocial': 'S-KNN',
        'hivt': 'HiVT',
        'lanegcn': 'LaneGCN',
        'lstmdefault': 'LSTM',
        'lstmsocial': 'S-LSTM'
    }
    for k, v in remap_methods.items():
        prediction_performance_RVO[v] = prediction_performance_RVO[k]
        del prediction_performance_RVO[k]

        prediction_performance_DESPOT[v] = prediction_performance_DESPOT[k]
        del prediction_performance_DESPOT[k]

    performance_summit_ade = np.array([2.938, 2.989, 3.099, 3.196, 1.692, 1.944, 2.410, 2.480])
    performance_summit_fde = np.array([5.950, 6.040, 6.505, 6.633, 3.524, 4.189, 5.061, 5.152])
    performance_summit_made = np.array([1.965, 2.003, 1.667, 1.740, 0.807, 1.016, 2.348, 2.396])
    performance_summit_mfde = np.array([3.962, 3.986, 2.742, 2.853, 1.064, 1.503, 4.924, 4.954])
    # Using static ADE

    static_metric = 'ADE'
    if static_metric == "ADE":
        performance_summit = performance_summit_ade
    elif static_metric == "FDE":
        performance_summit = performance_summit_fde
    elif static_metric == "minADE":
        performance_summit = performance_summit_made
    elif static_metric == "minFDE":
        performance_summit = performance_summit_mfde

    # This is the scatter plot with best-fit-line for each dynamic metric vs. static
    # All list below are predlen 30
    dynamic_ade_dict_DESPOT = {'ade_predlen30': np.zeros(len(methods_to_plot)), 
                        'ade_obs20_predlen30': np.zeros(len(methods_to_plot)),
                        'ade_20meters_closest_predlen30': np.zeros(len(methods_to_plot))}
    dynamic_ade_dict_RVO = {'ade_predlen30': np.zeros(len(methods_to_plot)), 
                        'ade_obs20_predlen30': np.zeros(len(methods_to_plot)),
                        'ade_20meters_closest_predlen30': np.zeros(len(methods_to_plot))}
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Printing out tree performance
    for key in prediction_performance_DESPOT.keys():
        idx = methods_to_plot.index(key)
        for which_dynamic_ade in dynamic_ade_dict_DESPOT.keys():
            dynamic_ade_dict_DESPOT[which_dynamic_ade][idx] = np.nanmean(prediction_performance_DESPOT[key][which_dynamic_ade])
    for key in prediction_performance_RVO.keys():
        idx = methods_to_plot.index(key)
        for which_dynamic_ade in dynamic_ade_dict_RVO.keys():
            dynamic_ade_dict_RVO[which_dynamic_ade][idx] = np.nanmean(prediction_performance_RVO[key][which_dynamic_ade])

    offsets=0.4
    space_between_methods = 0.5
    position = np.arange(len(methods_to_plot)) * (len(methods_to_plot) * offsets / 2 + space_between_methods)


    fig, ax = plt.subplots(figsize=(6.5,5))
    plt.rcParams.update({'font.size': 12})
    #fig.set_size_inches(18, 9)
    ax.bar(position-1*offsets, dynamic_ade_dict_DESPOT['ade_predlen30'], width=offsets, color=(*colors.to_rgba(default_colors[0])[:3], 0.8), align='center', label='Dyn. ADE')
    ax.bar(position+0*offsets, dynamic_ade_dict_DESPOT['ade_20meters_closest_predlen30'], width=offsets, color=(*colors.to_rgba(default_colors[1])[:3], 0.8), align='center', label='Dyn. ADE Closest')
    ax.bar(position+1*offsets, dynamic_ade_dict_DESPOT['ade_obs20_predlen30'], width=offsets, color=(*colors.to_rgba(default_colors[2])[:3], 0.8), align='center', label='Dyn. ADE 20 Obs')
    # Add labels and legend
    plt.ylabel(f'Dynamic ADE', fontsize=12)
    plt.title("Different Dynamic ADE of DESPOT Planner", fontsize=12)
    #plt.ylim(top=max(dynamic_ade_dict_RVO[which_dynamic_ade])*1.02)  # Adjust the ylim to provide more space for the legend
    #plt.xlim(left = min(performance_summit) * 0.95, right=max(performance_summit)*1.03)

    ax.legend(ncol=3, loc='upper center', fontsize='small')
    ax.set_ylim(0, 5.2)  # Adjust the ylim to provide more space for the legend
    # Set the x-tick labels to the method names
    ax.set_xticks(position)
    ax.set_xticklabels(methods_to_plot)

    plt.tight_layout()
    plt.grid()
    plt.savefig(f"C_Different_Dynamic_DESPOT.pdf", dpi=300)
    plt.clf()
    plt.close()

# Plot both static ADE vs driving performance; and dynamic ADE vs driving performance
def plot_static_ade_vs_driving_performance(prediction_performance_RVO, driving_performance_RVO,
                                           prediction_performance_DESPOT, driving_performance_DESPOT):

    ### Parameters
    planner_plot = 'RVO'
    if planner_plot == 'RVO':
        prediction_performance_map = prediction_performance_RVO
        driving_performance_map = driving_performance_RVO
    else:
        prediction_performance_map = prediction_performance_DESPOT
        driving_performance_map = driving_performance_DESPOT
    static_metric = 'ADE'
    #dynamic_metric = 'ade_obs20_closest_predlen30'
    dynamic_metric = 'ade_obs20_closest_predlen30'
    static_or_dynamic = 'dynamic'
    ### Done parameters

    performance_summit_ade = np.array([2.938, 2.989, 3.099, 3.196, 1.692, 1.944, 2.410, 2.480])
    performance_summit_fde = np.array([5.950, 6.040, 6.505, 6.633, 3.524, 4.189, 5.061, 5.152])
    performance_summit_made = np.array([1.965, 2.003, 1.667, 1.740, 0.807, 1.016, 2.348, 2.396])
    performance_summit_mfde = np.array([3.962, 3.986, 2.742, 2.853, 1.064, 1.503, 4.924, 4.954])

    if static_metric == "ADE":
        performance_summit = performance_summit_ade
    elif static_metric == "FDE":
        performance_summit = performance_summit_fde
    elif static_metric == "minADE":
        performance_summit = performance_summit_made
    elif static_metric == "minFDE":
        performance_summit = performance_summit_mfde

    methods_to_plot = ['cv', 'ca', 'knndefault', 'knnsocial', 'hivt', 'lanegcn', 'lstmdefault', 'lstmsocial']
    
    method_performance = get_method_performance(prediction_performance_map, driving_performance_map, 
                                                     methods_to_plot, dynamic_metric)
    
    if static_or_dynamic == 'dynamic':

        r_values_dict = {}

        # Find the best-fit-line
        for best_fit_metric in prediction_performance_map['cv'].keys():
            if "ade" not in best_fit_metric and "fde" not in best_fit_metric:
                continue
            driving_performance_P = get_method_performance(prediction_performance_map, driving_performance_map, 
                                                            methods_to_plot, best_fit_metric)
            driving_performance_P = [driving_performance_P[method]['driving'] for method in methods_to_plot]
            dynamic_metric_P = [np.nanmean(prediction_performance_map[key][best_fit_metric]) for key in methods_to_plot]

            x = dynamic_metric_P
            y = driving_performance_P

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Store the r-value in the dictionary
            r_values_dict[best_fit_metric] = r_value

            # Print the r-value for the current method
            #print(f"Method: {best_fit_metric}, R-value: {r_value} p-value {p_value}")

        # Sort the methods based on their r-values (ascending order)
        sorted_r_values = sorted(r_values_dict.items(), key=lambda x: x[1], reverse=False)

        # Print the best, 2nd best, and worst methods based on r-values
        for i in range(len(sorted_r_values)):
            print(f"{i}th method: {sorted_r_values[i][0]}, R-value: {sorted_r_values[i][1]} R^2-value: {sorted_r_values[i][1]**2}")
    
    #for performance_metric in ['driving', 'safety', 'efficiency', 'comfort']:
    for performance_metric in ['driving']:
        ## Static ADE vs Driving Performance##

        # Create a scatter plot with different colors for each point
        fig, ax = plt.subplots(figsize=(5, 5)) # adjust figure size as needed
        plt.rcParams.update({'font.size': 12})

        for i, method in enumerate(methods_to_plot):
            remap_methods = {
                'cv': 'CV',
                'ca': 'CA',
                'knndefault': 'KNN',
                'knnsocial': 'S-KNN',
                'hivt': 'HiVT',
                'lanegcn': 'LaneGCN',
                'lstmdefault': 'LSTM',
                'lstmsocial': 'S-LSTM'
            }
            if static_or_dynamic == 'static':
                plt.scatter(performance_summit[i], method_performance[method][performance_metric], color='black')
                if method == 'lstmdefault' and planner_plot == 'DESPOT':
                    y = -15
                    x_add = -5
                else:
                    y = 7
                    x_add = 0
                if method == 'lstmsocial' and planner_plot == 'DESPOT':
                    x_add = +10
                plt.annotate(remap_methods[method], (performance_summit[i], method_performance[method][performance_metric]), 
                            textcoords="offset points", xytext=(-2+x_add,y), ha='center')
            else:
                plt.scatter(method_performance[method][dynamic_metric], method_performance[method][performance_metric], color='black')
                y = 7
                x_add = 0
                if planner_plot == 'RVO':
                    x_add = 0
                    y = 8
                if planner_plot == 'RVO' and method == 'hivt':
                    x_add = 7
                if planner_plot == 'RVO' and method == 'lanegcn' and dynamic_metric=='ade_closest_predlen30':
                    x_add = -4
                if planner_plot == 'RVO' and method == 'lanegcn' and dynamic_metric=='fde_closest_predlen30':
                    x_add = -12
                if planner_plot == 'RVO' and method == 'hivt' and dynamic_metric=='fde_closest_predlen30':
                    x_add = +15
                plt.annotate(remap_methods[method], (method_performance[method][dynamic_metric], method_performance[method][performance_metric]), 
                            textcoords="offset points", xytext=(-2+x_add,y), ha='center')

        # Add labels and legend
        if static_or_dynamic == 'static':
            plt.xlabel(f'Static {static_metric}', fontsize=16)
        else:
            plt.xlabel(f'Dynamic {dynamic_metric[:3].upper()}', fontsize=16)

        # This line to ylabel the paper
        #plt.ylabel(f'{performance_metric.capitalize()} Performance', fontsize=16)
        # This line to ylabel my thesis
        plt.ylabel('$P_{drive}$', fontsize=16)
        plt.ylim(top=np.max([method_performance[x][performance_metric] for x in methods_to_plot])*1.03)  # Adjust the ylim to provide more space for the legend
        if static_or_dynamic == 'static':
            plt.xlim(left = min(performance_summit)*0.93, right=max(performance_summit)*1.06)
            x_fit_line = performance_summit
        else:
            x_fit_line = [method_performance[method][dynamic_metric] for method in methods_to_plot]
            plt.xlim(left = min(x_fit_line)*0.89, right=max(x_fit_line)*1.06)

        y_fit_line = [method_performance[x][performance_metric] for x in methods_to_plot] 
        ax = plot_best_fit_line(x_fit_line, y_fit_line, ax, color="blue")

        slope, intercept, r_value, p_value, std_err = stats.linregress(x_fit_line, y_fit_line)
        plt.title('$\mathrm{R}^2$' + f': {r_value **2:.2f} $\;$ $p$-value: {p_value:.3f}', fontsize=16)
        plt.tight_layout()
        plt.grid(True, linewidth=0.5, alpha=0.7)

        if static_or_dynamic == 'static':
            plt.savefig(f"B_Static_{static_metric}_vs_{performance_metric.capitalize()}_Performance_{planner_plot}.pdf", dpi=300) 
            #plt.savefig(f"B_Static_{static_metric}_vs_$P_{{drive}}$_{planner_plot}.pdf", dpi=300)
        else:
            plt.savefig(f"A_Dynamic_{dynamic_metric.upper()}_vs_{performance_metric.capitalize()}_Performance_{planner_plot}.pdf", dpi=300)
        plt.clf()
        plt.close()

def plot_best_fit_line(x, y, ax, color="blue", color_text = 'black'):
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Calculate the best-fit line
    line = slope * np.array(x) + intercept

    # Plot the best-fit line
    ax.plot(x, line, color=color, label='Best-fit line', alpha=1.0)

    # Calculate the 95% confidence interval
    #t = stats.t.ppf(1 - 0.025, len(x) - 2)  # 95% confidence level
    #slope_ci = t * std_err
    #intercept_ci = t * (std_err * np.sqrt(np.sum(x**2) / len(x)))


    # Calculate the endpoints for the confidence interval lines
    #lower_line = (slope - slope_ci) * np.array(x) + (intercept - intercept_ci)
    #upper_line = (slope + slope_ci) * np.array(x) + (intercept + intercept_ci)


    # Plot the confidence interval lines
    #ax.plot(x, lower_line, color=color, linestyle='--', alpha=0.7, label='Lower 95% CI')
    #ax.plot(x, upper_line, color=color, linestyle='--', alpha=0.7, label='Upper 95% CI')

    # Add regression statistics to the plot
    stats_text = f"""\
    R^2: {r_value ** 2:.2f}
    p-value: {p_value:.4f}
    """
    #ax.text(0.05, 0.25, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', color=color_text)

    return ax



def get_method_performance(prediction_performance, driving_performance, methods_to_plot, prediction_metric):  

    max_min_method = find_max_min(driving_performance, methods_to_plot)
    method_performance = {}
    for i, method in enumerate(methods_to_plot):
        ade_fde, safety_data, comfort_data, efficiency_data, performance = get_normalized_data(method, prediction_performance, 
                        driving_performance, max_min_method, prediction_metric)

        method_performance[method] = {'safety': np.nanmean(safety_data), 
                                      'comfort': np.nanmean(comfort_data),
                                 'efficiency': np.nanmean(efficiency_data), 
                                 'driving': np.nanmean(performance),
                                 prediction_metric: np.nanmean(ade_fde)}
        
        method_performance[i] = np.nanmean(performance)

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
    
    directories_map_DESPOT = {
            'cv': '/home/phong/driving_data/official/despot_planner/same_computation/cv2Hz',
            'ca': '/home/phong/driving_data/official/despot_planner/same_computation/ca2Hz',
            'hivt': '/home/phong/driving_data/official/despot_planner/same_computation/hivt02Hz',
            'lanegcn': '/home/phong/driving_data/official/despot_planner/same_computation/lanegcn02Hz',
            'lstmdefault': '/home/phong/driving_data/official/despot_planner/same_computation/lstmdefault05Hz',
            'lstmsocial': '/home/phong/driving_data/official/despot_planner/same_computation/lstmsocial03Hz/',
            'knnsocial': '/home/phong/driving_data/official/despot_planner/same_computation/knnsocial005Hz/',
            'knndefault': '/home/phong/driving_data/official/despot_planner/same_computation/knndefault001Hz/',
        }
    prediction_performance_DESPOT = {}
    driving_performance_DESPOT = {}
    algorithm_performance_DESPOT = {}
    # Storing to a pickle file

    process_rvo_or_despot = 'rvo'
    if  args.mode == "data":
        if process_rvo_or_despot == 'rvo':

            for key in directories_map_RVO.keys():
                print(f"Processing {key}")
                prediction_performance, driving_performance, tree_performance = get_dynamic_ade(directories_map_RVO[key])
                prediction_performance_RVO[key] = prediction_performance
                driving_performance_RVO[key] = driving_performance
                algorithm_performance_RVO[key] = tree_performance

                save_dict_to_file(prediction_performance_RVO, f'pickle_filesrvo3/prediction_performance_RVO_{key}.pickle')
                save_dict_to_file(driving_performance_RVO, f'pickle_filesrvo3/driving_performance_RVO_sameHz_{key}.pickle')
                save_dict_to_file(algorithm_performance_RVO, f'pickle_filesrvo3/algorithm_performance_RVO_{key}.pickle')
                print(f"Finish saving to pickle file. Exit")
        
        exit(0)

    for key in directories_map_RVO.keys():
        # 1 is offial, 2 is no use, 3 is with ob20_closest_predlen30
        prediction_performance_RVO[key] = load_dict_from_file(f'pickle_filesrvo/prediction_performance_RVO_{key}.pickle')[key]
        driving_performance_RVO[key] = load_dict_from_file(f'pickle_filesrvo/driving_performance_RVO_{key}.pickle')[key]
        algorithm_performance_RVO[key] = load_dict_from_file(f'pickle_filesrvo/algorithm_performance_RVO_{key}.pickle')[key]

    for key in directories_map_DESPOT.keys():
        prediction_performance_DESPOT[key] = load_dict_from_file(f'pickle_filesdespot1/prediction_performance_DESPOT_{key}.pickle')[key]
        driving_performance_DESPOT[key] = load_dict_from_file(f'pickle_filesdespot1/driving_performance_DESPOT_{key}.pickle')[key]
        algorithm_performance_DESPOT[key] = load_dict_from_file(f'pickle_filesdespot1/algorithm_performance_DESPOT_{key}.pickle')[key]

    # Done
    # Plot the line for static vs dynamic. 
    # Comment: Line plot cannot show relationship.
    # Conclusion: Using scatter plot for static vs dynamic
    #plot_dynamic_vs_staticade_scatter(prediction_performance_RVO, prediction_performance_DESPOT,
    #                                  list(prediction_performance_RVO.keys()))
    
    plot_static_ade_vs_driving_performance(prediction_performance_RVO, driving_performance_RVO,
                                           prediction_performance_DESPOT, driving_performance_DESPOT)
    
    #plot_dynamic_vs_staticade_bar(prediction_performance_RVO, prediction_performance_DESPOT,
    #                                  list(prediction_performance_RVO.keys()))
    
    # Plot the bar for 3 dynamics
