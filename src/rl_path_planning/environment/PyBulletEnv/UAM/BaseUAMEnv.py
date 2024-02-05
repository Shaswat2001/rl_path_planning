import gym
from math import pi
from gym import spaces
import numpy as np
import random
import time
import pybullet as p
from scipy.spatial.transform import Rotation

class BaseUAMEnv(gym.Env):
    
    def __init__(self,controller = None): 

        p.connect(p.GUI)

        self.robot_id = p.loadURDF("/home/shaswatgarg/ros_ws/src/rl_path_planning/rl_path_planning/environment/PyBulletEnv/UAM/Assets/urdf/model.urdf", [0, 0, 0.0],flags=p.URDF_USE_SELF_COLLISION)

        self.state = None
        self.state_size = 7
        self.action_max = np.array([0.1,0.1,0.1,0.01,0.01,0.01,0.01])
        self.safe_action_max = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.05,0.05,0.05]*3)

        self.q = None

        self.max_time = 10
        self.dt = 0.01
        self.current_time = 1

        self.current_iteration = 0
        self.current_space = 1

        self.q_vel_bound = np.array([3,3,3,1.5,1.5,1.5,1.5,1.5,1.5,1.5])
        self.max_q_bound = np.array([12,12,23,2*np.pi,0.4,3,1.3])
        self.min_q_bound = np.array([-12,-12,0.6,-2*np.pi,-2.7,-0.2,-3.4])

        self.max_q_safety = np.array([8,8,8,2*np.pi,0,2.5,1])
        self.min_q_safety = np.array([-8,-8,2,-2*np.pi,-2,0,-3])
        # self.max_q_safety = None
        # self.min_q_safety = None

        self.max_safety_engage = np.array([5.5,5.5,5.5])
        self.min_safety_engage = np.array([-5.5,-5.5,0.8])

        self.safe_action_max = np.array([8,8,8,2*np.pi,0,2.5,1])
        self.safe_action_min = np.array([-8,-8,2,-2*np.pi,-2,0,-3])

        # self.controller = UAM.UAM(self.publish_simulator,self.min_q_bound,self.max_q_bound)
        self.action_space = spaces.Box(-self.action_max,self.action_max,dtype=np.float64)

        self.joint_indices = {}
        self.world_indices = {}
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            self.joint_indices[joint_info[1].decode("utf-8")] = i

            p.setCollisionFilterPair(bodyUniqueIdA=self.robot_id,
                            linkIndexA=0,
                            bodyUniqueIdB=self.robot_id,
                            linkIndexB=i,
                            enableCollision=0)
            
        print(self.joint_indices)
    
    def step(self, action):
        # print(f"Old state : {self.q}")
        action = action[0]
        # print(f"Action is : {action}")
        self.q = self.q + action[:7]
        self.q[3:7] = np.clip(self.q[3:7],self.min_q_bound[3:7],self.max_q_bound[3:7])
        # print(f"New state : {self.q}")
        # print(f"New velocity : {new_q_vel}")
        # self.q,self.qdot = self.controller.solve(new_q,new_q_vel)
        self.changeSimState(self.q)
        self.man_pos = np.round(p.getLinkState(self.robot_id,self.joint_indices["gripper_left_joint"])[0],2)

        self.const_broken = self.constraint_broken()
        self.pose_error = self.get_error()
        reward,done = self.get_reward()
        constraint = self.get_constraint()
        info = self.get_info(constraint)
        if done:
            print(f"Constraint Broken : {self.const_broken}")
            print(f"The position error at the end : {self.pose_error}")
            print(f"The end pose is : {self.man_pos}")
            print(f"The end pose of UAV is : {self.q[:3]}")

        # pose_diff = np.clip(self.q_des - self.man_pos,np.array([-1,-1,-1]),np.array([1,1,1]))
        pose_diff = self.q_des - self.man_pos
        prp_state = np.concatenate((pose_diff,self.q[3:7]))
        prp_state = prp_state.reshape(1,-1)

        self.current_time += 1
        return prp_state, reward, done, info

    def get_reward(self):
        
        done = False
        pose_error = self.pose_error

        if not self.const_broken:

            if pose_error < 0.01:
                done = True
                reward = 1000
            elif pose_error < 0.05:
                done = True
                reward = 100
            elif pose_error < 0.1:
                done = True
                reward = 50
            # elif pose_error < 0.5:
            #     reward = 10
            # elif pose_error < 0.7:
            #     reward = 50
            # elif pose_error < 1:
            #     reward = 0
            else:
                reward = -pose_error*10

            if self.current_time > self.max_time:
                done = True
                reward -= 2
        
        else:
            reward = -100
            done = True

        return reward,done
    
    def get_constraint(self):
        
        constraint = 0
        if self.const_broken:

            for i in range(self.q.shape[0]):
                if self.q[i] > self.max_q_bound[i]:
                    constraint+= (self.q[i] - self.max_q_bound[i])*10
                elif self.q[i] < self.min_q_bound[i]:
                    constraint+= abs(self.q[i] - self.min_q_bound[i])*10

            if constraint <= 0:
                constraint = 10
        else:

            for i in range(self.q.shape[0]):
                constraint+= (abs(self.q[i]) - self.max_q_bound[i])*10

        return constraint

    def get_info(self,constraint):

        info = {}
        info["constraint"] = constraint
        info["safe_reward"] = -constraint
        info["safe_cost"] = 0
        info["negative_safe_cost"] = 0
        info["engage_reward"] = -10

        if np.any(self.q > self.max_q_safety) or np.any(self.q < self.min_q_safety):
            info["engage_reward"] = 10
            
        if constraint > 0:
            info["safe_cost"] = 1
            info["negative_safe_cost"] = -1

        return info

    def constraint_broken(self):

        if self.checkSelfContact():
            print("ROBOT COLLIDED")
            return True
        
        if np.any(self.q[:3] > self.max_q_bound[:3]) or np.any(self.q[:3] < self.min_q_bound[:3]):
            return True
    
        return False
    
    def get_error(self):

        pose_error =  np.linalg.norm(self.man_pos - self.q_des) 

        return pose_error
        
    def reset(self):

        #initial conditions
        self.q = np.array([0,0,2,0.01,0.01,0.01,0.01]) # starting location [x, y, z] in inertial frame - meters
        # self.qdot = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]) #initial velocity [x; y; z] in inertial frame - m/s
        self.q_des = np.random.randint([-1,-1,1],[2,2,4])
        # if self.current_iteration%2000 == 0:
        #     self.q_des = np.random.randint([-self.current_space,-self.current_space,1],[self.current_space+1,self.current_space+1,self.current_space+2])
        #     self.current_space+=1
        # self.q_des = np.array([-2,-2,8])
        # self.qdot_des = np.zeros(self.qdot.shape)
        # self.qdotdot_des = np.zeros(self.qdot.shape)
        print(f"The target pose is : {self.q_des}")
        # Add initial random roll, pitch, and yaw rates
        
        # self.controller.reset(self.q,self.qdot,self.qdot)
        self.changeSimState(self.q)
        self.resetSimState()
        self.man_pos = p.getLinkState(self.robot_id,self.joint_indices["gripper_left_joint"])[0]
        # print(f"the man pose : {self.man_pos}")
        pose_diff = self.q_des - self.man_pos
        # pose_diff = np.clip(self.q_des - self.man_pos,np.array([-1,-1,-1]),np.array([1,1,1]))
        prp_state = np.concatenate((pose_diff,self.q[3:7]))
        prp_state = prp_state.reshape(1,-1)
        self.current_time = 1
        self.current_iteration+=1

        return prp_state
    
    def reset_eval(self,q,q_des):

        #initial conditions
        self.q = q
        # self.qdot = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]) #initial velocity [x; y; z] in inertial frame - m/s
        self.q_des = q_des
        # if self.current_iteration%2000 == 0:
        #     self.q_des = np.random.randint([-self.current_space,-self.current_space,1],[self.current_space+1,self.current_space+1,self.current_space+2])
        #     self.current_space+=1
        # self.q_des = np.array([-2,-2,8])
        # self.qdot_des = np.zeros(self.qdot.shape)
        # self.qdotdot_des = np.zeros(self.qdot.shape)
        print(f"The target pose is : {self.q_des}")
        # Add initial random roll, pitch, and yaw rates
        
        # self.controller.reset(self.q,self.qdot,self.qdot)
        self.changeSimState(self.q)
        self.man_pos = p.getLinkState(self.robot_id,self.joint_indices["gripper_left_joint"])[0]
        # print(f"the man pose : {self.man_pos}")
        pose_diff = self.q_des - self.man_pos
        # pose_diff = np.clip(self.q_des - self.man_pos,np.array([-1,-1,-1]),np.array([1,1,1]))
        prp_state = np.concatenate((pose_diff,self.q[3:7]))
        prp_state = prp_state.reshape(1,-1)
        self.current_time = 1
        self.current_iteration+=1

        return prp_state
    
    def changeSimState(self,q):

        p.resetBasePositionAndOrientation(self.robot_id, q[:3],[0,0,0,1])
        
        p.setJointMotorControl2(self.robot_id, self.joint_indices["arm_1_joint"], p.POSITION_CONTROL, targetPosition=q[3])
        p.setJointMotorControl2(self.robot_id, self.joint_indices["arm_2_joint"], p.POSITION_CONTROL, targetPosition=q[4])
        p.setJointMotorControl2(self.robot_id, self.joint_indices["arm_3_joint"], p.POSITION_CONTROL, targetPosition=q[5])
        p.setJointMotorControl2(self.robot_id, self.joint_indices["arm_4_joint"], p.POSITION_CONTROL, targetPosition=q[6])
        p.setJointMotorControl2(self.robot_id, self.joint_indices["gripper_left_joint"], p.POSITION_CONTROL, targetPosition=0.01)
        p.setJointMotorControl2(self.robot_id, self.joint_indices["gripper_right_joint"], p.POSITION_CONTROL, targetPosition=0.01)
        p.stepSimulation()

    def resetSimState(self):

        p.resetJointState(self.robot_id, self.joint_indices["arm_1_joint"], 0.01)
        p.resetJointState(self.robot_id, self.joint_indices["arm_2_joint"], 0.01)
        p.resetJointState(self.robot_id, self.joint_indices["arm_3_joint"], 0.01)
        p.resetJointState(self.robot_id, self.joint_indices["arm_4_joint"], 0.01)

    def checkSelfContact(self):

        is_contact = False
        contact_info = p.getContactPoints(self.robot_id,self.robot_id)
        if contact_info is not None and len(contact_info) > 0:
            is_contact = True

        base_lk_pose = self.q[:3]
        manip_lk_pose = p.getLinkState(self.robot_id,self.joint_indices["gripper_left_joint"])[0]

        if manip_lk_pose[-1] > base_lk_pose[-1]:
            is_contact = True
        
        return is_contact
    
    def store_joint_data(self,joint_angles):
        
        for i in range(len(joint_angles)):
            self.manip_joint_list[i].append(joint_angles[i])
     