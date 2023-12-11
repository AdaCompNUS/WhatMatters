import numpy as np
import copy

def smooth_l1_loss(y_true, y_pred, delta=1.0):
    assert y_true.shape == y_pred.shape, "y_true and y_pred should have the same shape"
    
    error = y_true - y_pred
    abs_error = np.abs(error)
    
    loss = np.where(abs_error < delta, 0.5 * np.square(error), delta * (abs_error - 0.5 * delta))
    return np.mean(loss)


def consistency(pred, pred_1forward, time_shifting):

    future = np.array(pred_1forward[:,:(-time_shifting)])
    target = np.array(pred[:,time_shifting:])
    output = smooth_l1_loss(future, target)

    #return output.item()
    return output

def calculate_consistency(exos_list, pred_exo_list, pred_len):
        
    compare, compare_1forward = [], []
    compare_closest, compare_1forward_closest = [], []
    
    for timestep in exos_list.keys():
        if timestep in exos_list.keys() and timestep in pred_exo_list.keys() \
            and (timestep+1) in exos_list.keys() and (timestep+1) in pred_exo_list.keys() \
            and (timestep+2) in exos_list.keys() and (timestep+2) in pred_exo_list.keys():

            pred = dict()
            pred_1forward = dict()
            pred_closest = dict()
            pred_1forward_closest = dict()
            
            for agent_index in range(len(exos_list[timestep])):
                agent_id = exos_list[timestep][agent_index]['id']
                pre = np.array([pred_exo_list[timestep][j][agent_index]['pos'] for j in range(pred_len)])
                pred[agent_id] = pre
                if agent_index == 0:
                    pred_closest[agent_id] = pre

            
            for agent_index in range(len(exos_list[timestep+1])):
                id_1forward = exos_list[timestep+1][agent_index]['id']
                pre_1forward = np.array([pred_exo_list[timestep+1][j][agent_index]['pos'] for j in range(pred_len)])
                pred_1forward[id_1forward] = pre_1forward
                if agent_index == 0:
                    pred_1forward_closest[id_1forward] = pre_1forward
            
            for id in pred.keys():
                if id in pred_1forward:
                    compare.append(pred[id])
                    compare_1forward.append(pred_1forward[id])
            
            for id in pred_closest.keys():
                if id in pred_1forward_closest:
                    compare_closest.append(pred_closest[id])
                    compare_1forward_closest.append(pred_1forward_closest[id])
    
    compare = np.array(compare)
    compare_1forward = np.array(compare_1forward)
    compare_closest = np.array(compare_closest)
    compare_1forward_closest = np.array(compare_1forward_closest)

    tem_cos = consistency(compare,compare_1forward,1)
    tem_cos_closest = consistency(compare_closest,compare_1forward_closest,1)

    return tem_cos, tem_cos_closest
    
