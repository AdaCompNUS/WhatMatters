
import numpy as np
import matplotlib.pyplot as plt

safety_weights = {
            'collision_rate': 0.3,
            'near_miss_rate': 0.7,
            'mean_min_ttc': 0.0
        }
safety_directions = {
            'collision_rate': 'lower',
            'near_miss_rate': 'lower',
            'mean_min_ttc': 'higher'
        }
comfort_weights = {
            'jerk': 0.5,
            'lateral_acceleration': 0.5,
            'acceleration': 0.0
        }
comfort_directions = {
            'jerk': 'lower',
            'lateral_acceleration': 'lower',
            'acceleration': 'lower'

        }
efficiency_weights = {
            'avg_speed': 0.3,
            'tracking_error': 0.3,
            'efficiency_time': 0.2,
            'distance_traveled': 0.2
        }
efficiency_directions = {
            'avg_speed': 'higher',
            'tracking_error': 'lower',
            'efficiency_time': 'lower',
            'distance_traveled': 'higher'
        }


def scatter_plot_multi_predictions(prediction_performance, driving_performance, methods_to_plot):  
    # Function to compute the weighted average
    def normalize(arr):
        if np.max(arr) - np.min(arr) == 0:
            return np.zeros_like(arr)
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    def weighted_average(data, weights, directions):
        norm_data = [normalize(arr) if direction == 'lower' else 1 - normalize(arr) for arr, direction in zip(data, directions)]
        return np.sum([w * d for w, d in zip(weights, norm_data)], axis=0)

    def plot_method(method, prediction_performance, driving_performance, row, axes):
        # Weights for each metric within a category
        
        # Compute the weighted averages for each category
        safety_data = weighted_average(
            [driving_performance[method]['safety']['collision_rate'],
            driving_performance[method]['safety']['near_miss_rate'],
            driving_performance[method]['safety']['mean_min_ttc']],
            list(safety_weights.values()), list(safety_directions.values()))

        comfort_data = weighted_average(
            [driving_performance[method]['comfort']['jerk'],
            driving_performance[method]['comfort']['lateral_acceleration']],
            list(comfort_weights.values()), list(comfort_directions.values()))

        efficiency_data = weighted_average(
            [driving_performance[method]['efficiency']['avg_speed'],
            driving_performance[method]['efficiency']['tracking_error'],
            driving_performance[method]['efficiency']['efficiency_time'],
            driving_performance[method]['efficiency']['distance_traveled']],
            list(efficiency_weights.values()), list(efficiency_directions.values()))

        # Plot the scatter plots
        axes[row, 0].scatter(prediction_performance[method]['ade'], comfort_data)
        axes[row, 0].set_title(f'{method} - Comfort')
        axes[row, 0].set_xlabel('ADE')
        axes[row, 0].set_ylabel('Weighted Average')


        axes[row, 1].scatter(prediction_performance[method]['ade'], safety_data)
        axes[row, 1].set_title(f'{method} - Safety')
        axes[row, 1].set_xlabel('ADE')

        axes[row, 2].scatter(prediction_performance[method]['ade'], efficiency_data)
        axes[row, 2].set_title(f'{method} - Efficiency')
        axes[row, 2].set_xlabel('ADE')

    # List of methods to plot
    #methods_to_plot = ['method1', 'method2']

    fig, axes = plt.subplots(len(methods_to_plot), 3, figsize=(18, 6 * len(methods_to_plot)))

    for i, method in enumerate(methods_to_plot):
        plot_method(method, prediction_performance, driving_performance, i, axes)

    plt.tight_layout()
    plt.show()