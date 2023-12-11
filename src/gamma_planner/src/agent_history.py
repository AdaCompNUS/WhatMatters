import time
import numpy as np

class AgentType:
    car = 0 # Car
    ped = 1 # Pedestrian
    num_values = 2 # Other

class History():
    def __init__(self, max_observations = 20, time_interval = 0.3, time_threshold = 6.0):
        '''
        max_observations: maximum number of observations to store
        time_interval: time interval between two observations
        time_threshold: time threshold to determine if the agent is stuck
        ego_history: history of ego agent, list of (x, y) tuples
        '''
        self.max_observations = max_observations
        self.time_interval = time_interval
        self.time_threshold = time_threshold
        self.exo_history = {}
        self.exo_history_time = {} # time of each update, for printing only
        self.ego_history = []
        self.ego_history_time = {} # time of each update, for printing only
        self.last_exo_update = {}
        self.last_ego_update = None
        self.ego_id = -1

    def add_exo_observation(self, agent_id, x, y):
        '''
        Add an exo observation
        '''
        current_time = time.time()
        if agent_id not in self.last_exo_update or (current_time - self.last_exo_update[agent_id]) >= 0.9*self.time_interval:
            if (current_time - self.last_exo_update.get(agent_id, 0)) > self.time_threshold:
                self.exo_history[agent_id] = []
                self.exo_history_time[agent_id] = [] # for printing only

            if len(self.exo_history[agent_id]) >= self.max_observations:
                self.exo_history[agent_id].pop(0)
                self.exo_history_time[agent_id].pop(0) # for printing only
            self.exo_history[agent_id].append((x, y))
            self.exo_history_time[agent_id].append(current_time) # for printing only
            self.last_exo_update[agent_id] = current_time

            
    def add_ego_observation(self, x, y):
        current_time = time.time()
        if self.last_ego_update is None or (current_time - self.last_ego_update) >= 0.9*self.time_interval:
            if (current_time - (self.last_ego_update or 0)) > self.time_threshold:
                self.ego_history = []
                self.ego_history_time = [] # for printing only

            if len(self.ego_history) >= self.max_observations:
                self.ego_history.pop(0)
                self.ego_history_time.pop(0) # for printing only
            self.ego_history.append((x, y))
            self.ego_history_time.append(current_time) # for printing only
            self.last_ego_update = current_time

    def get_exo_history(self, agent_id):
        return self.exo_history.get(agent_id, [])

    def get_ego_history(self):
        return self.ego_history
    
    def build_request(self):
        '''
        Build request for MOPED
        '''

        # Purge history if the agent is stuck for a long time
        current_time = time.time()
        for agent_id, last_update in self.last_exo_update.items():
            if (current_time - last_update) > 2 * self.time_threshold: # multiply 2 to be safe
                self.exo_history.pop(agent_id, None)
                self.last_exo_update.pop(agent_id, None)
                self.exo_history_time.pop(agent_id, None)

        request = {}
        request_time = {}
        for agent_id, history in self.exo_history.items():
            # For the sake of LaneGCN, we consider moving cars only, static one will be ignored
            # I believe this one should be commented out, as it will not fair for other methods
            # when other methods are able to use the static cars
            #if np.sum(np.abs(np.diff(np.array(history), axis=0))) < 1e-8:
            #    continue
            request[agent_id] = {'agent_id': agent_id, 'agent_type': AgentType.car, 'agent_history': history, 'is_ego': False}
            request_time[agent_id] = {'agent_id': agent_id, 'agent_time': self.exo_history_time[agent_id]}

        request[self.ego_id] = {'agent_id': self.ego_id, 'agent_type': AgentType.car, 'agent_history': self.ego_history, 'is_ego': True}
        request_time[self.ego_id] = {'agent_id': self.ego_id, 'agent_time': self.ego_history_time}

        return request, request_time
    
    def set_ego_id(self, ego_id):
        self.ego_id = ego_id

    def __str__(self):
        return "Exo History: {}\n Ego History: {}".format(self.exo_history, self.ego_history)