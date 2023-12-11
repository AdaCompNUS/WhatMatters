import numpy as np

def displacement_error(pred, gt):
    return np.linalg.norm(np.array(pred) - np.array(gt))

def displacement_error_2d(pred_list, gt_list):
    pred_array = np.array(pred_list)
    gt_array = np.array(gt_list)
    return np.mean(np.sqrt(((pred_array-gt_array)**2).sum(1)))

def ade_fde(pred_car_list, pred_exo_list, ego_list, exos_list):

    obs_len = 20 # Parameter
    pred_len = 20 # Parameter

    ego_ade = []
    ego_fde = []
    ego_obs_list = {} # exo observation

    # Calculate ADE and FDE for ego agent
    # for timestep in list(ego_list.keys()):

    #     errors = []
    #     num_pred = len(pred_car_list[timestep])
    #     ego_obs_list[timestep] = []
        
    #     for j in range(num_pred):
    #         if (timestep + j+1) in ego_list.keys():
    #             ego_pred = pred_car_list[timestep][j]['pos']
    #             ego_gt_next = ego_list[timestep+j+1]['pos'] # +1 because the prediction is for the next timestep
    #             error = displacement_error(ego_pred, ego_gt_next)
    #             errors.append(error)
    #             if j < 20 and (timestep +j - 20) in ego_list.keys():
    #                 ego_obs_list[timestep].append(ego_list[timestep + j - 20]['pos'])
    #     if len(errors) > 0:
    #         ego_ade.append(np.mean(errors))
    #         ego_fde.append(errors[-1])


    # Calculate ADE and FDE for each exo agent
    exo_ade = [] #
    exo_ade_predlen3 = [] #
    exo_ade_predlen10 = []#
    exo_ade_predlen30 = []#
    exo_ade_obs20 = []#
    exo_ade_obs20_predlen3 = [] #
    exo_ade_obs20_predlen10 = []#
    exo_ade_obs20_predlen30 = []#
    exo_ade_closest = [] #
    exo_ade_closest_predlen3 = []
    exo_ade_closest_predlen10 = []
    exo_ade_closest_predlen30 = []
    exo_ade_20meters_closest = []
    exo_ade_20meters_closest_predlen3 = []
    exo_ade_20meters_closest_predlen10 = []
    exo_ade_20meters_closest_predlen30 = []
    
    exo_ade_obs20_closest_predlen30 = []
    exo_ade_obs20_20meters_closest_predlen30 = []

    exo_fde = [] #
    exo_fde_predlen3 = []#
    exo_fde_predlen10 = []#
    exo_fde_predlen30 = []#
    exo_fde_obs20 = [] #
    exo_fde_obs20_predlen3 = [] #
    exo_fde_obs20_predlen10 = [] #
    exo_fde_obs20_predlen30 = [] #
    exo_fde_closest = [] #
    exo_fde_closest_predlen3 = []
    exo_fde_closest_predlen10 = []
    exo_fde_closest_predlen30 = []
    exo_fde_20meters_closest = []
    exo_fde_20meters_closest_predlen3 = []
    exo_fde_20meters_closest_predlen10 = []
    exo_fde_20meters_closest_predlen30 = []

    exo_fde_obs20_closest_predlen30 = []
    exo_fde_obs20_20meters_closest_predlen30 = []


    for timestep in list(exos_list.keys()):

        #if len(exo_ade) == 60: # Do not get too much
            #break

        agent_distance_to_ego = {}

        for agent_index, exo_agent in enumerate(exos_list[timestep]):


            agent_info = exo_agent
            agent_id = agent_info['id']

            all_agent_ids_at_this_timestep = [agent['id'] for agent in exos_list[timestep]]
            agent_distance_to_ego[agent_index] = np.linalg.norm(np.array(exos_list[timestep][all_agent_ids_at_this_timestep.index(agent_id)]['pos']) -\
                                                                np.array(ego_list[timestep]['pos']))
            #assert len(agent_distance_to_ego) == 1 or \
            # agent_distance_to_ego[agent_index] >= agent_distance_to_ego[agent_index-1], f"{agent_distance_to_ego}"
            

            if timestep not in pred_exo_list.keys():
                continue

            num_pred = len(pred_exo_list[timestep])
            exo_pred_at_timestep = pred_exo_list[timestep] # A list of 30 predictions. Each prediction is a list of agents
            
            exo_pred_list = [] # exo prediction
            exo_ggt_list = [] # exo ground truth
            exo_obs_list = [] # exo observation
            
            for j in range(num_pred):
                if (timestep + j+1) not in exos_list.keys():
                    break
                all_agent_ids_at_next_timestep = [agent['id'] for agent in exos_list[timestep+j+1]]

                if timestep >= 20:
                    all_agent_ids_at_prev_20_timestep = [agent['id'] for agent in exos_list[timestep+j-20]]
                
                # If agent is not in the next timestep, then there is no point in calculating the error
                if agent_id not in all_agent_ids_at_next_timestep:
                    break
                
                # If agent not in prev 20 timestep, then it not have enough history to calculate the error
                #if agent_id not in all_agent_ids_at_prev_20_timestep:
                #    break
                
                if  timestep >= 20 and j < 20 and agent_id in all_agent_ids_at_prev_20_timestep:
                    exo_obs_list.append(exos_list[timestep+j-20][all_agent_ids_at_prev_20_timestep.index(agent_id)]['pos'])

                if agent_id in all_agent_ids_at_next_timestep:
                    agent_index_at_next_timestep = all_agent_ids_at_next_timestep.index(agent_id)
                    exo_pred = exo_pred_at_timestep[j][agent_index]['pos']
                    exo_gt_next = exos_list[timestep+j+1][agent_index_at_next_timestep]['pos']
                    exo_pred_list.append(exo_pred)
                    exo_ggt_list.append(exo_gt_next)
                    #error = displacement_error(exo_pred, exo_gt_next)
                    #errors.append(error)
            
            if len(exo_pred_list) > 0:
                ade = displacement_error_2d(exo_pred_list, exo_ggt_list)
                fde = displacement_error_2d(exo_pred_list[-1:], exo_ggt_list[-1:])
                exo_ade.append(ade)
                exo_fde.append(fde)

                ade_predlen3 = None
                ade_predlen10 = None
                ade_predlen30 = None
                fde_predlen3 = None
                fde_predlen10 = None
                fde_predlen30 = None

                if len(exo_pred_list) >= 3:
                    ade_predlen3 = displacement_error_2d(exo_pred_list[0:3], exo_ggt_list[0:3])
                    fde_predlen3 = displacement_error_2d([exo_pred_list[2]], [exo_ggt_list[2]])
                   
                if len(exo_pred_list) >= 10:
                    ade_predlen10 = displacement_error_2d(exo_pred_list[0:10], exo_ggt_list[0:10])
                    fde_predlen10 = displacement_error_2d([exo_pred_list[9]], [exo_ggt_list[9]])
                
                if len(exo_pred_list) == 30:
                    ade_predlen30 = ade
                    fde_predlen30 = fde

                if ade_predlen3 != None:
                    exo_ade_predlen3.append(ade_predlen3)
                if ade_predlen10 != None:
                    exo_ade_predlen10.append(ade_predlen10)
                if ade_predlen30 != None:
                    exo_ade_predlen30.append(ade_predlen30)
                if fde_predlen3 != None:
                    exo_fde_predlen3.append(fde_predlen3)
                if fde_predlen10 != None:
                    exo_fde_predlen10.append(fde_predlen10)
                if fde_predlen30 != None:
                    exo_fde_predlen30.append(fde_predlen30) 

                if len(exo_obs_list) == obs_len:
                    exo_ade_obs20.append(ade)
                    exo_fde_obs20.append(fde)

                    if ade_predlen3 != None:
                        exo_ade_obs20_predlen3.append(ade_predlen3)
                    if ade_predlen10 != None:
                        exo_ade_obs20_predlen10.append(ade_predlen10)
                    if ade_predlen30 != None:
                        exo_ade_obs20_predlen30.append(ade_predlen30)
                    if fde_predlen3 != None:
                        exo_fde_obs20_predlen3.append(fde_predlen3)
                    if fde_predlen10 != None:
                        exo_fde_obs20_predlen10.append(fde_predlen10)
                    if fde_predlen30 != None:
                        exo_fde_obs20_predlen30.append(fde_predlen30) 
                
                if agent_index in [0,1,2]:
                    exo_ade_closest.append(ade)
                    exo_fde_closest.append(fde)

                    if ade_predlen3 != None:
                        exo_ade_closest_predlen3.append(ade_predlen3)
                    if ade_predlen10 != None:
                        exo_ade_closest_predlen10.append(ade_predlen10)
                    if ade_predlen30 != None:
                        exo_ade_closest_predlen30.append(ade_predlen30)
                        if len(exo_obs_list) == obs_len:
                            exo_ade_obs20_closest_predlen30.append(ade_predlen30)
                    if fde_predlen3 != None:
                        exo_fde_closest_predlen3.append(fde_predlen3)
                    if fde_predlen10 != None:
                        exo_fde_closest_predlen10.append(fde_predlen10)
                    if fde_predlen30 != None:
                        exo_fde_closest_predlen30.append(fde_predlen30)
                        if len(exo_obs_list) == obs_len:
                            exo_fde_obs20_closest_predlen30.append(fde_predlen30)

                        
                
                if agent_distance_to_ego[agent_index] < 20:
                    exo_ade_20meters_closest.append(ade)
                    exo_fde_20meters_closest.append(fde)

                    if ade_predlen3 != None:
                        exo_ade_20meters_closest_predlen3.append(ade_predlen3)
                    if ade_predlen10 != None:
                        exo_ade_20meters_closest_predlen10.append(ade_predlen10)
                    if ade_predlen30 != None:
                        exo_ade_20meters_closest_predlen30.append(ade_predlen30)
                        if len(exo_obs_list) == obs_len:
                            exo_ade_obs20_20meters_closest_predlen30.append(ade_predlen30)
                    if fde_predlen3 != None:
                        exo_fde_20meters_closest_predlen3.append(fde_predlen3)
                    if fde_predlen10 != None:
                        exo_fde_20meters_closest_predlen10.append(fde_predlen10)
                    if fde_predlen30 != None:
                        exo_fde_20meters_closest_predlen30.append(fde_predlen30)
                        if len(exo_obs_list) == obs_len:
                            exo_fde_obs20_20meters_closest_predlen30.append(fde_predlen30)
                    
        
        #print(f"Agent distance {[round(v,2) for o, v in agent_distance_to_ego.items()]}")
    
    if len(exo_ade) == 0:
        assert False

    # Calculate average ADE for each exo agent
    # We average all time steps for each agent. Then we average all agents.
    exo_ade_mean = np.nanmean(exo_ade)
    exo_ade_predlen3_mean = np.nanmean(exo_ade_predlen3)
    exo_ade_predlen10_mean = np.nanmean(exo_ade_predlen10)
    exo_ade_predlen30_mean = np.nanmean(exo_ade_predlen30)
    exo_ade_obs20_mean = np.nanmean(exo_ade_obs20)
    exo_ade_obs20_predlen3_mean = np.nanmean(exo_ade_obs20_predlen3)
    exo_ade_obs20_predlen10_mean = np.nanmean(exo_ade_obs20_predlen10)
    exo_ade_obs20_predlen30_mean = np.nanmean(exo_ade_obs20_predlen30)
    exo_ade_closest_mean = np.nanmean(exo_ade_closest)
    exo_ade_closest_predlen3_mean = np.nanmean(exo_ade_closest_predlen3)
    exo_ade_closest_predlen10_mean = np.nanmean(exo_ade_closest_predlen10)
    exo_ade_closest_predlen30_mean = np.nanmean(exo_ade_closest_predlen30)
    exo_ade_20meters_closest_mean = np.nanmean(exo_ade_20meters_closest)
    exo_ade_20meters_closest_predlen3_mean = np.nanmean(exo_ade_20meters_closest_predlen3)
    exo_ade_20meters_closest_predlen10_mean = np.nanmean(exo_ade_20meters_closest_predlen10)
    exo_ade_20meters_closest_predlen30_mean = np.nanmean(exo_ade_20meters_closest_predlen30)

    exo_ade_obs20_closest_predlen30_mean = np.nanmean(exo_ade_obs20_closest_predlen30)
    exo_ade_obs20_20meters_closest_predlen30_mean = np.nanmean(exo_ade_obs20_20meters_closest_predlen30)


    exo_fde_mean = np.nanmean(exo_fde)
    exo_fde_predlen3_mean = np.nanmean(exo_fde_predlen3)
    exo_fde_predlen10_mean = np.nanmean(exo_fde_predlen10)
    exo_fde_predlen30_mean = np.nanmean(exo_fde_predlen30)
    exo_fde_obs20_mean = np.nanmean(exo_fde_obs20)
    exo_fde_obs20_predlen3_mean = np.nanmean(exo_fde_obs20_predlen3)
    exo_fde_obs20_predlen10_mean = np.nanmean(exo_fde_obs20_predlen10)
    exo_fde_obs20_predlen30_mean = np.nanmean(exo_fde_obs20_predlen30)
    exo_fde_closest_mean = np.nanmean(exo_fde_closest)
    exo_fde_closest_predlen3_mean = np.nanmean(exo_fde_closest_predlen3)
    exo_fde_closest_predlen10_mean = np.nanmean(exo_fde_closest_predlen10)
    exo_fde_closest_predlen30_mean = np.nanmean(exo_fde_closest_predlen30)
    exo_fde_20meters_closest_mean = np.nanmean(exo_fde_20meters_closest)
    exo_fde_20meters_closest_predlen3_mean = np.nanmean(exo_fde_20meters_closest_predlen3)
    exo_fde_20meters_closest_predlen10_mean = np.nanmean(exo_fde_20meters_closest_predlen10)
    exo_fde_20meters_closest_predlen30_mean = np.nanmean(exo_fde_20meters_closest_predlen30)

    exo_fde_obs20_closest_predlen30_mean = np.nanmean(exo_fde_obs20_closest_predlen30)
    exo_fde_obs20_20meters_closest_predlen30_mean = np.nanmean(exo_fde_obs20_20meters_closest_predlen30)

    
    return exo_ade_mean,\
    exo_ade_predlen3_mean ,\
    exo_ade_predlen10_mean ,\
    exo_ade_predlen30_mean,\
    exo_ade_obs20_mean,\
    exo_ade_obs20_predlen3_mean,\
    exo_ade_obs20_predlen10_mean ,\
    exo_ade_obs20_predlen30_mean ,\
    exo_ade_closest_mean ,\
    exo_ade_closest_predlen3_mean,\
    exo_ade_closest_predlen10_mean ,\
    exo_ade_closest_predlen30_mean ,\
    exo_ade_20meters_closest_mean ,\
    exo_ade_20meters_closest_predlen3_mean ,\
    exo_ade_20meters_closest_predlen10_mean,\
    exo_ade_20meters_closest_predlen30_mean ,\
    exo_ade_obs20_closest_predlen30_mean, \
    exo_ade_obs20_20meters_closest_predlen30_mean, \
    exo_fde_mean,\
    exo_fde_predlen3_mean ,\
    exo_fde_predlen10_mean,\
    exo_fde_predlen30_mean,\
    exo_fde_obs20_mean ,\
    exo_fde_obs20_predlen3_mean ,\
    exo_fde_obs20_predlen10_mean ,\
    exo_fde_obs20_predlen30_mean ,\
    exo_fde_closest_mean ,\
    exo_fde_closest_predlen3_mean ,\
    exo_fde_closest_predlen10_mean ,\
    exo_fde_closest_predlen30_mean ,\
    exo_fde_20meters_closest_mean ,\
    exo_fde_20meters_closest_predlen3_mean ,\
    exo_fde_20meters_closest_predlen10_mean ,\
    exo_fde_20meters_closest_predlen30_mean ,\
    exo_fde_obs20_closest_predlen30_mean, \
    exo_fde_obs20_20meters_closest_predlen30_mean, \
    exo_ade,\
    exo_ade_predlen30, \
    exo_ade_closest,\
    exo_ade_closest_predlen30,\
    exo_ade_20meters_closest,\
    exo_ade_20meters_closest_predlen30