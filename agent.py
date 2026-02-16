import numpy as np

class MDPArchitect:
    def __init__(self, w_wait=0.8, w_queue=0.2):
        # Configuration based on your network.net.xml
        self.incoming_lanes = ['N_in_0', 'E_in_0', 'S_in_0', 'W_in_0']
        self.max_lane_capacity = 25  # Estimated based on 92.8m length
        self.w_wait = w_wait
        self.w_queue = w_queue

    def normalize_observation(self, raw_obs):
        """
        Input: Raw vehicle counts/densities from SUMO
        Output: Scaled values [0, 1] for the PPO neural network
        """
        # raw_obs from sumo-rl usually contains density and queue info
        normalized = np.array(raw_obs, dtype=np.float32) / self.max_lane_capacity
        return np.clip(normalized, 0, 1)

    def reward_function(self, traffic_signal):
        """
        Custom reward logic to be passed to SumoEnvironment.
        Matches the tlLogic 'A0' in your XML.
        """
        # 1. Penalize Accumulated Waiting Time (Seconds vehicles are stopped)
        wait_time = traffic_signal.get_accumulated_waiting_time()
        
        # 2. Penalize Queue Length (Total vehicles currently in queue)
        queue = sum(traffic_signal.get_lanes_queue())
        
        # Reward is the negative weighted sum (Maximizing -Wait is Minimizing Wait)
        reward = -(self.w_wait * wait_time + self.w_queue * queue)
        return reward

# For your Final Report:
# Reward Formula: R = -(0.8 * Total_Wait + 0.2 * Total_Queue)