import gym
from math import pi
from gym import spaces
import numpy as np
import random
import yaml
import pybullet as p
from .Assets.GridEnv import GridEnvironment
from .BaseUAMEnv import BaseUAMEnv

class BaseUAMObsEnv(BaseUAMEnv):
    
    def __init__(self,controller = None,load_obstacle = True):
        
        super().__init__(controller)
        
        self.state_size = 39

        self.grid = GridEnvironment(min_coords=-self.pos_bound,max_coords=self.pos_bound,resolution=0.1)
        
        if load_obstacle:
            with open("/Users/shaswatgarg/Documents/WaterlooMASc/StateSpaceUAV/Config/obstacle_1.yaml", "r") as stream:
                try:
                    obstacles = yaml.safe_load(stream)["obstacle_coordinates"]
                except yaml.YAMLError as exc:
                    print(exc)

            self.generate_state_space(obstacles)

    def step(self, action):

        self.state, reward, done, self.info = super().step(action)
        
        if self.check_collision():
            reward -= 300
            if self.info["constraint"] < 0:
                self.info["constraint"] = 20
            else:
                self.info["constraint"] += 10

        if self.info["constraint"] > 0:
            self.info["safe_cost"] = 1
            self.info["engage_reward"] = 10
            self.info["negative_safe_cost"] = -1

        nrb_region = self.grid.encode_state(self.uav_pos)
        self.state = np.concatenate((self.state,nrb_region))
        
        return self.state, reward, done, self.info
        
    def reset(self):

        super().reset()
        #initial conditions
        while True:
            if self.check_goal_collision(self.pos_des):
                self.pos_des = np.random.randint(-4,5,size=(3,))
            else:
                break

        nrb_region = self.grid.encode_state(self.uav_pos)
        self.state = np.concatenate((self.state,nrb_region))
        return self.state
    
    def check_collision(self):

        for obstacle_id in self.obstacle_id_list:
            contact_info = p.getContactPoints(bodyA=self.robot_id, bodyB=obstacle_id)

            if contact_info is not None and len(contact_info) > 0:
                return True
        
        return False
    
    def check_goal_collision(self,position):

        position = list(position)
        for coordinate in self.coordinate:

            min_crd = coordinate[0]
            max_crd = coordinate[1]

            if np.all(position > min_crd) and np.all(position < max_crd):
                return True
        
        return False

    def generate_state_space(self,obstacles):

        self.obstacle_id_list = []
        self.coordinate = []
        for key,coordinate in obstacles.items():

            obstacle_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
            obstacle_position = coordinate
            obstacle_orientation = p.getQuaternionFromEuler([0, 0, 0])  # No rotation
            self.obstacle_id_list.append(p.createMultiBody(baseCollisionShapeIndex=obstacle_id,
                            basePosition=obstacle_position,
                            baseOrientation=obstacle_orientation))
            
            ctr_coordinate = np.array(coordinate)
            self.grid.add_obstacle(ctr_coordinate)

            min_max_coordinate = [ctr_coordinate - self.grid.obs_dim/2,ctr_coordinate + self.grid.obs_dim/2]
            self.coordinate.append(min_max_coordinate)

            print(f"Adding Obstacle : {key}")
