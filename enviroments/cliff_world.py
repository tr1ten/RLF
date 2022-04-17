import re
from matplotlib.pyplot import step
import numpy as np
import os
import sys
agentpath = "../agents/"
sys.path.append(os.path.abspath(agentpath))
from expectedsarsa_agent import ExpectedSarsaAgent

DEFAULT_ENV = {
    "width": 4,
    "length": 12
}
DEFAULT_AGENT = {"num_actions": 4, "num_states": 48, "epsilon": 0.1,
                 "step_size": 0.1, "discount": 1.0, "seed": 0}


class CliffEnv():
    def __init__(self, env_info=DEFAULT_ENV):
        self.grid_w = env_info["width"]
        self.grid_l = env_info["length"]
        # 0 - left 1 - right 2-up 3-down
        self.action_nums = 4
        self.start_loc = (0, self.grid_w-1)
        self.goal_loc = (self.grid_l-1, self.grid_w-1)
        self.agent_loc = self.start_loc
        # all cliffs positions
        self.cliff = [(x, self.grid_w-1) for x in range(1, self.grid_l-1)]

    def start(self):
        reward = 0
        state = self.state(self.start_loc)
        terminated = False
        obs = (reward, state, terminated)
        return obs

    def state(self, loc):
        return loc[1]*self.grid_l + loc[0]

    def step(self, action):
        x, y = self.agent_loc
        if(action == 0):
            x -= 1
        elif(action == 1):
            x += 1
        elif (action == 2):
            y -= 1
        else:
            y += 1

        if(x >= self.grid_l or y >= self.grid_w or x<0 or y<0):
            x, y = self.agent_loc

        self.agent_loc = (x, y)
        # reward time
        terminated = False
        reward = -1
        if((x, y) in self.cliff):
            self.agent_loc = self.start_loc
            reward = -100
        if((x, y) == self.goal_loc):
            terminated = True
        return (reward, self.state(self.agent_loc), terminated)

    def reset(self):
        self.agent_loc = self.start_loc

if(__name__ == "__main__"):
    env = CliffEnv()
    agent = ExpectedSarsaAgent()
    num_eps = 1000
    for eps in range(num_eps):
        reward_sum = 0
        steps_sum = 0
        (reward, state, terminated) = env.start()
        action = agent.agent_start(state)
        (reward, state, terminated) = env.step(action)
        reward_sum +=reward
        steps_sum+=1
        while not terminated:
            action = agent.agent_step(reward, state)
            (reward, state, terminated) = env.step(action)
            reward_sum +=reward
            steps_sum+=1
        agent.agent_end(reward)
        if(eps>num_eps-10):
            print(f"steps:{steps_sum} & reward in #{eps}:{reward_sum}")
        env.reset()
    

