#!/home/phong/anaconda3/envs/HiVT/bin/python

import Pyro4

def create_sample_data():

    # Build request
    raw_data = {-1: {"x": [150,152,160], "y":[250.2, 252.2, 255.2], "type": 1},
            20: {"x": [151,159,165, 170, 180], "y":[255.2, 259.2, 260.2, 240, 245], "type": 1},
            3: {"x": [160.1, 166], "y": [256, 257], "type": 2}}
        
    sample_data = {}
    for agent_id, agent_info in raw_data.items():
        agent_type = agent_info["type"]  # Change this if needed
        observations = [(x, y) for x, y in zip(agent_info["x"], agent_info["y"])]
        sample_data[agent_id] = {'agent_id': agent_id, 'agent_type': agent_type, 'agent_history': observations, 'is_ego': False}
    return sample_data

def main():
    agent_data = create_sample_data()  # Modify the number of agents if needed
    agent_predictor = Pyro4.Proxy("PYRO:mopedservice.warehouse@localhost:8300")

    print(f"Data is {agent_data}")

    agent_data = {32: {'is_ego': False, 'agent_history': [(2599.301513671875, 90.0221939086914)], 'agent_type': 0, 'agent_id': 32}, 51: {'is_ego': False, 'agent_history': [(2733.703125, 93.19654846191406)], 'agent_type': 0, 'agent_id': 51}, 2: {'is_ego': True, 'agent_history': [(2788.658447265625, 87.225830078125)], 'agent_type': 0, 'agent_id': 2}, 49: {'is_ego': False, 'agent_history': [(2893.82763671875, 55.74800109863281)], 'agent_type': 0, 'agent_id': 49}, 63: {'is_ego': False, 'agent_history': [(2733.586181640625, 34.97210693359375)], 'agent_type': 0, 'agent_id': 63}, 10: {'is_ego': False, 'agent_history': [(3001.787353515625, 90.50440979003906)], 'agent_type': 0, 'agent_id': 10}, 43: {'is_ego': False, 'agent_history': [(2774.17578125, 94.55093383789062)], 'agent_type': 0, 'agent_id': 43}, 44: {'is_ego': False, 'agent_history': [(2694.781494140625, 94.5885009765625)], 'agent_type': 0, 'agent_id': 44}, 46: {'is_ego': False, 'agent_history': [(2993.1767578125, 98.74962615966797)], 'agent_type': 0, 'agent_id': 46}, 93: {'is_ego': False, 'agent_history': [(2750.826416015625, 34.5711555480957)], 'agent_type': 0, 'agent_id': 93}, 16: {'is_ego': False, 'agent_history': [(2996.24267578125, 79.84114837646484)], 'agent_type': 0, 'agent_id': 16}, 81: {'is_ego': False, 'agent_history': [(2589.534423828125, 94.38941955566406)], 'agent_type': 0, 'agent_id': 81}, 82: {'is_ego': False, 'agent_history': [(2564.489990234375, 127.64022064208984)], 'agent_type': 0, 'agent_id': 82}, 19: {'is_ego': False, 'agent_history': [(2624.123779296875, 93.11006927490234)], 'agent_type': 0, 'agent_id': 19}, 25: {'is_ego': False, 'agent_history': [(2784.35888671875, 64.37129974365234)], 'agent_type': 0, 'agent_id': 25}, 53: {'is_ego': False, 'agent_history': [(2919.052734375, 70.62605285644531)], 'agent_type': 0, 'agent_id': 53}, 87: {'is_ego': False, 'agent_history': [(2885.57763671875, 55.94069290161133)], 'agent_type': 0, 'agent_id': 87}, 57: {'is_ego': False, 'agent_history': [(2624.237060546875, 72.6158218383789)], 'agent_type': 0, 'agent_id': 57}, 59: {'is_ego': False, 'agent_history': [(2690.09423828125, 104.12584686279297)], 'agent_type': 0, 'agent_id': 59}, 74: {'is_ego': False, 'agent_history': [(2984.004638671875, 45.55058670043945)], 'agent_type': 0, 'agent_id': 74}, 21: {'is_ego': False, 'agent_history': [(2721.044677734375, 89.6438980102539)], 'agent_type': 0, 'agent_id': 21}}

    future_frames = agent_predictor.predict(agent_data)

    print(f"Status: {future_frames['is_error']} model running: {future_frames['moped_model']}")
    for agent_id, agent_data in future_frames.items():
        if type(agent_id) == int:
            print(f"Agent ID {agent_id}: {agent_data['agent_id']}  Agent prob: {agent_data['agent_prob']} Agent pred: {agent_data['agent_prediction']}")

if __name__ == "__main__":
    main()
