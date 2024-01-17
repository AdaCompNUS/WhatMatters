#!/home/phong/anaconda3/envs/HiVT/bin/python

import Pyro4
import numpy as np  # Your motion prediction service module
import argparse
import logging
import sys
import time
import logging.handlers

import pathlib
current_dir = pathlib.Path(__file__).parent.resolve()

MAX_HISTORY_MOTION_PREDICTION = 20

print(f"Python executable: {sys.executable}")

from moped_implementation.planner_wrapper import PlannerWrapper

max_bytes = 500 * 1024 * 1024  # 500Mb
backup_count = 1  # keep only the latest file
filename = f"{current_dir}/logpyro4mopedknnsocial.txt"

max_bytes = 1024 * 1024
backup_count = 10

handler = logging.handlers.RotatingFileHandler(
    filename=filename,
    maxBytes=max_bytes,
    backupCount=backup_count,
    mode='a'
)

formatter = logging.Formatter(
    '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class MotionPredictionService(object):

    def __init__(self):
        self.pred_len = 30 # Predict 10 frames into the future as 10 * 0.3 = 3 seconds so it is good enough
        self.planner = PlannerWrapper(pred_len=self.pred_len)

    def predict(self, agents_data):
        '''
            :param data: a dictionary {'agent_id': int, 'agent_history': [(x1,y1), (x2,y2), ...], 'agent_type': int, 'is_ego': bool}
                    Length of agent_history is any arbitrary but must be less than MAX_HISTORY_MOTION_PREDICTION
            
            Return
            :return: a dictionary {'agent_id': int, 'agent_prediction': [(x1,y1), (x2,y2), ...], 'agent_prob': float}
                    Length of agent_prediction is self.pred_len
                    We also add 2 keys to dictionary to ensure the reproducible between simulator and predictor
                    'observation_array': numpy array (number_agents, observation_len, 2)
                    'is_error': bool to indicate if any error happens
        '''

        # Step 1. From agents_data, build numpy array (number_agents, observation_len, 2)

        try:
            agent_id_list = []
            xy_pos_list = []
            ego_id = -1
            padedd_observation_array = {} # to store padded observation array and return for reproducible results
            for agent_id, agent_data in agents_data.items():
                if agent_data['is_ego'] == False: # -1 is the ego agent, so we add at last
                    xy_pos = np.array(agent_data['agent_history'])
                    # NOTE, In pyro4 version, I padd at the beginning
                    # first axis, we pad width=0 at end and pad width=MAX_HISTORY_MOTION_PREDICTION-xy_pos.shape[0] at the beginning
                    # second axis (which is x and y axis), we do not pad anything as it does not make sense to pad anything
                    xy_pos = np.pad(xy_pos, pad_width=((MAX_HISTORY_MOTION_PREDICTION - xy_pos.shape[0], 0), (0, 0)),
                                    mode="edge")
                    xy_pos_list.append(xy_pos)
                    agent_id_list.append(agent_id)
                    padedd_observation_array[agent_id] = [(x, y) for x, y in xy_pos]
                else:
                    ego_id = agent_id
            #print('aaaa')
            # Add ego agent at last
            ego_agent_data = agents_data[ego_id] # Ego agent has id -1
            xy_pos = np.array(ego_agent_data['agent_history'])
            xy_pos = np.pad(xy_pos, pad_width=((0, MAX_HISTORY_MOTION_PREDICTION - xy_pos.shape[0]), (0, 0)),
                                    mode="edge")
            padedd_observation_array[ego_id] = [(x, y) for x, y in xy_pos]
            #print('vbbb')

            xy_pos_list.append(xy_pos)
            agent_id_list.append(ego_id)

            # Creating history
            agents_history = np.stack(xy_pos_list)  # Shape (number_agents, observation_len, 2)
            #print('cccc with shape ', agents_history.shape)
            #print(f"Obs are are {agents_history}")

            #print(f"Agents history is {agents_history}")
            #print(f"Shape of agents history is {agents_history.shape}")

        except Exception as e:
            logger.info(f"Error in building numpy array: {e} with inputs {agents_data}")
            return {'is_error': True, 'observation_array': padedd_observation_array}
        
        try:
            probs, predictions = self.planner.do_predictions(agents_history)
            #logger.info(f"Predictions are {predictions}")
        except Exception as e:
            probs = np.ones(agents_history.shape[0])
            # Predictions is the last known position but with shape (number_agents, self.pred_len, 2)
            predictions = agents_history[:, -self.pred_len:, :]
            logger.info(f"Error in prediction: {e} with inputs {agents_history}")
            return {'is_error': True}

        #print(f"Predictions are {predictions}")
        #print('eee')
        #print('probs', probs)
        #print('pred shape', predictions.shape)

        #print('agent_id_list', agent_id_list)
        # Build response:
        try:
            data_response = {}
            for i, agentID in enumerate(agent_id_list):
                prob_info = probs[i]
                agent_predictions = [tuple([float(row[0]), float(row[1])]) for row in predictions[i]]
                data_response[agentID] = {'agent_prediction': agent_predictions, 'agent_prob': float(prob_info), 'agent_id': agentID}

            # Adding 3 keys
            data_response['observation_array'] = padedd_observation_array
            data_response['is_error'] = False
            data_response['moped_model'] = self.planner.model_running
            
        except:
            logger.info(f"Error in output data, predictions shape: {predictions.shape}, "
                        f"probs shape: {probs.shape}, agent_id_list: {agent_id_list} Inputs shape: {agents_history.shape}")
            data_response = {'is_error': True}
        
        return data_response
        
    
def find_value_types(data):
    types = set()
    for key, value in data.items():
        if isinstance(value, dict):
            types.update(find_value_types(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, list):
                    types.update(find_value_types({"": item}))
                else:
                    types.add(type(item))
                    print(f"Type of {key} is {type(value)}")
        else:
            print(f"Type of {key} is {type(value)}")
            types.add(type(value))
    return types

def main(args):
    # with Pyro4.Daemon(host=args.host, port=args.mopedpyroport) as daemon:
    #     service = MotionPredictionService()
    #     uri = daemon.register(service)
    #     print(f"URI: {uri}")
    #     with Pyro4.locateNS() as ns:
    #         ns.register("mopedservice.warehouse", uri)
    #     daemon.requestLoop()
    Pyro4.Daemon.serveSimple(
        {
            MotionPredictionService: "mopedservice.warehouse",
        },
        host=args.host,
        port=args.mopedpyroport,
        ns=False,
    )

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--mopedpyroport',
        metavar='P',
        default=8300,
        type=int,
        help='TCP port to listen to (default: 8300)')
    args = argparser.parse_args()

    main(args)
