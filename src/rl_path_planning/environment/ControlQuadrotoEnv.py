import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
from rl_path_planning.environment.Dynamics import Quadrotor

class QuadrotorEnv(gym.Env):
    
    def __init__(self,safeRL = False): 
        
        self.Quadrotor = Quadrotor.Quadrotor(np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3))
        self.state = None

        self.safeRL = safeRL

        minT = self.Quadrotor.minT
        maxT = self.Quadrotor.maxT
        # maxT = 0.1
        # minT = -0.1
        k1 = self.Quadrotor.k1
        k2 = self.Quadrotor.k2
        L = self.Quadrotor.L

        self.max_angle = np.array([np.pi/6,np.pi/6,2*np.pi])

        self.max_time = 10
        self.dt = self.Quadrotor.dt
        self.current_time = 0

        self.pos_bound = np.array([30,30,30])

        # self.max_action = np.array([4*maxT,L*(maxT - minT),L*(maxT - minT),2*k2/k1*(maxT - minT)])
        # self.min_action = np.array([4*minT,L*(minT - maxT),L*(minT - maxT),2*k2/k1*(minT - maxT)])

        self.max_action = np.array([maxT]*(int(1)))
        self.min_action = np.array([minT]*(int(1)))

        # self.max_action = np.array([30,0.5,0.5,0.5])
        # self.min_action = np.array([4.96,-0.5,-0.5,-0.5])
        self.max_delta = np.array([0.8,0.8,1,1])
        # self.observation_space = spaces.Box(-4*self.Quadrotor.minT,4*self.Quadrotor.minT, dtype=np.float64)
        self.action_space = spaces.Box(self.min_action,self.max_action,dtype=np.float64)

    def step(self, action):

        prev_action = self.wheel_speed[0]
        self.wheel_speed = np.array([action[0]]*4)
        # thrust_needed = action[0]
        # torque_needed = action[1:]
        self.wheel_speed = np.clip(self.wheel_speed,[0.5]*4,[16.5]*4)
        wheel_speed = self.wheel_speed/self.Quadrotor.k1
        self.Quadrotor.stepRL(wheel_speed)
        # print(self.wheel_speed)
        pos = self.Quadrotor.pos
        # print(pos)
        ang = self.Quadrotor.ort
        vel = self.Quadrotor.lnr_vel
        pos_error = self.Quadrotor.getPoseError(pos)
        vel_error = self.Quadrotor.getLnrVelError(vel)
        self.state = np.concatenate((pos,vel,pos_error,vel_error))
        done = False
        reward = -np.linalg.norm(pos_error)*0.01 - np.linalg.norm(vel_error)*0.01 - np.linalg.norm(prev_action - self.wheel_speed[0])*0.01
        reward = 6 - np.exp(np.linalg.norm(pos_error)/np.linalg.norm(self.Quadrotor.pos_des))

        constraint = np.abs(pos) - self.pos_bound
        constraint = constraint[-1]*0.1
        self.info["constraint"] = constraint
        self.info["safe_cost"] = 0
        self.info["negative_safe_cost"] = 0
        if constraint > 0:
            self.info["safe_cost"] = 1
            self.info["negative_safe_cost"] = -1

        if (np.linalg.norm(pos_error) < 0.1) and (np.linalg.norm(vel_error)< 0.01):
            reward = 100
            print(f"error is {np.linalg.norm(pos_error)}")
            print("SOLVED")
            done = True
        elif (np.linalg.norm(pos_error) < 0.5) and (np.linalg.norm(vel_error)< 0.1):
            reward = 50
            print(f"error is {np.linalg.norm(pos_error)}")
            print("SOLVED-partially")
            done = True
        # elif np.any(abs(pos) > self.pos_bound):
        #     reward = -100
        #     # print("Robot out of bound")
        #     # done = True 
        if not done and self.current_time > self.max_time/self.dt:
            print("cycle end")
            print(f"error is {np.linalg.norm(pos_error)}")
            reward -= -10
            done = True
        
        # if self.safeRL:
        #     if constraint > 0:
        #         print("Constraint broken")
        #         print(f"error is {np.linalg.norm(pos_error)}")
        #         reward = -100
        #         done = True

        self.current_time += 1
        return self.state, reward, done, self.info
        

    def reset(self):

        height = np.random.randint(0,10)
        r_ref = np.array([0., 0., height]) # desired position [x, y, z] in inertial frame - meters

        #initial conditions
        # pos = np.array([0., 0., 0.]) # starting location [x, y, z] in inertial frame - meters
        #pos = [0.5, 1., 2.] # starting location [x, y, z] in inertial frame - meters
        pos = np.array([0,0, 0.]) # starting location [x, y, z] in inertial frame - meters
        vel = np.array([0., 0., 0.]) #initial velocity [x; y; z] in inertial frame - m/s
        ang = np.array([0., 0., 0.]) #initial Euler angles [phi, theta, psi] relative to inertial frame in deg


        # Add initial random roll, pitch, and yaw rates
        deviation = 10 # magnitude of initial perturbation in deg/s
        random_set = np.array([random.random(), random.random(), random.random()])
        ang_vel = np.deg2rad(2* deviation * random_set - deviation) #initial angular velocity [phi_dot, theta_dot, psi_dot]
        
        self.Quadrotor = Quadrotor.Quadrotor(pos,vel,ang,ang_vel,r_ref)
        self.wheel_speed = np.zeros(self.Quadrotor.num_motors)

        self.state = np.concatenate((pos,vel,self.Quadrotor.getPoseError(pos),self.Quadrotor.getLnrVelError(vel)))
        self.current_time = 0
        self.info = {}

        return self.state

    def get_obs(self):

        return self.state

if __name__ == '__main__':

    #Environment unit test!
    #Cant be executed before environment is registered with gym, in envs/__init__.py
    env = QuadrotorEnv()

    env.reset().shape

    env.step(env.action_space.sample())

    print(env.action_space.high)

     