#!/usr/bin/env python3

import os
import pickle
import argparse
import sys
import numpy as np
import time
import rospy
from laser_assembler.srv import AssembleScans
from gazebo_msgs.srv import GetLinkState
from geometry_msgs.msg import Twist

from rl_path_planning.agent import DDPG,TD3,SAC,SoftQ,RCRL,SEditor,USL,SAAC,IDEA1,IDEA2,IDEA3,IDEA4
from rl_path_planning.pytorch_model import GaussianPolicyNetwork, PolicyNetwork,QNetwork,VNetwork,PhasicPolicyNetwork,PhasicQNetwork,ConstraintNetwork,MultiplierNetwork,SafePolicyNetwork,RealNVP,FeatureExtractor

from rl_path_planning.replay_buffer.Uniform_RB import ReplayBuffer,VisionReplayBuffer
from rl_path_planning.replay_buffer.Auxiliary_RB import AuxReplayBuffer
from rl_path_planning.replay_buffer.Constraint_RB import ConstReplayBuffer,CostReplayBuffer

from rl_path_planning.exploration.OUActionNoise import OUActionNoise

from rl_path_planning.environment.GazeboEnv.Quadrotor.BaseGazeboUAVVelObsEnv1 import BaseGazeboUAVVelObsEnv1
from rl_path_planning.environment.GazeboEnv.Quadrotor.BaseGazeboUAVVelObsEnv1PCD import BaseGazeboUAVVelObsEnv1PCD

def build_parse():

    parser = argparse.ArgumentParser(description="RL Algorithm Variables")

    parser.add_argument("Environment",nargs="?",type=str,default="uav_vel_obs_gazebo1pcd",help="Name of OPEN AI environment")
    parser.add_argument("input_shape",nargs="?",type=int,default=[],help="Shape of environment state")
    parser.add_argument("n_actions",nargs="?",type=int,default=[],help="shape of environment action")
    parser.add_argument("max_action",nargs="?",type=float,default=[],help="Max possible value of action")
    parser.add_argument("min_action",nargs="?",type=float,default=[],help="Min possible value of action")

    parser.add_argument("Algorithm",nargs="?",type=str,default="DDPG",help="Name of RL algorithm")
    parser.add_argument('tau',nargs="?",type=float,default=0.005)
    parser.add_argument('gamma',nargs="?",default=0.99)
    parser.add_argument('actor_lr',nargs="?",type=float,default=0.0001,help="Learning rate of Policy Network")
    parser.add_argument('critic_lr',nargs="?",type=float,default=0.0001,help="Learning rate of the Q Network")
    parser.add_argument('mult_lr',nargs="?",type=float,default=0.0001,help="Learning rate of the LAG constraint")

    parser.add_argument("mem_size",nargs="?",type=int,default=100000,help="Size of Replay Buffer")
    parser.add_argument("batch_size",nargs="?",type=int,default=64,help="Batch Size used during training")
    parser.add_argument("n_episodes",nargs="?",type=int,default=50000,help="Total number of episodes to train the agent")
    parser.add_argument("target_update",nargs="?",type=int,default=2,help="Iterations to update the target network")
    parser.add_argument("vision_update",nargs="?",type=int,default=5,help="Iterations to update the vision network")
    parser.add_argument("delayed_update",nargs="?",type=int,default=100,help="Iterations to update the second target network using delayed method")
    parser.add_argument("enable_vision",nargs="?",type=bool,default=False,help="Whether you want to integrate sensor data")
    
    # SOFT ACTOR PARAMETERS
    parser.add_argument("temperature",nargs="?",type=float,default=0.2,help="Entropy Parameter")
    parser.add_argument("log_std_min",nargs="?",type=float,default=np.log(1e-4),help="")
    parser.add_argument("log_std_max",nargs="?",type=float,default=np.log(4),help="")
    parser.add_argument("aux_step",nargs="?",type=int,default=8,help="How often the auxiliary update is performed")
    parser.add_argument("aux_epoch",nargs="?",type=int,default=6,help="How often the auxiliary update is performed")
    parser.add_argument("target_entropy_beta",nargs="?",type=float,default=-3,help="")
    parser.add_argument("target_entropy",nargs="?",type=float,default=-3,help="")

    # MISC VARIABLES 
    parser.add_argument("save_rl_weights",nargs="?",type=bool,default=True,help="save reinforcement learning weights")
    parser.add_argument("save_results",nargs="?",type=bool,default=True,help="Save average rewards using pickle")

    # USL 
    parser.add_argument("eta",nargs="?",type=float,default=0.05,help="USL eta")
    parser.add_argument("delta",nargs="?",type=float,default=0.1,help="USL delta")
    parser.add_argument("Niter",nargs="?",type=int,default=20,help="Iterations")
    parser.add_argument("cost_discount",nargs="?",type=float,default=0.99,help="Iterations")
    parser.add_argument("kappa",nargs="?",type=float,default=5,help="Iterations")
    parser.add_argument("cost_violation",nargs="?",type=int,default=20,help="Save average rewards using pickle")

    # Safe RL parameters
    parser.add_argument("safe_iterations",nargs="?",type=int,default=5,help="Iterations to run Safe RL once engaged")
    parser.add_argument("safe_max_action",nargs="?",type=float,default=[],help="Max possible value of safe action")
    parser.add_argument("safe_min_action",nargs="?",type=float,default=[],help="Min possible value of safe action")

    # Environment Teaching parameters
    parser.add_argument("safe_iterations",nargs="?",type=int,default=5,help="Iterations to run Safe RL once engaged")
    parser.add_argument("teach_alg",nargs="?",type=str,default="alp_gmm",help="How to change the environment")

    # Environment parameters List
    parser.add_argument("max_obstacles",nargs="?",type=int,default=10,help="Maximum number of obstacles need in the environment")
    parser.add_argument("obs_region",nargs="?",type=float,default=6,help="Region within which obstacles should be added")

    # ALP GMM parameters
    parser.add_argument('gmm_fitness_fun',nargs="?", type=str, default="aic")
    parser.add_argument('warm_start',nargs="?", type=bool, default=False)
    parser.add_argument('nb_em_init',nargs="?", type=int, default=1)
    parser.add_argument('min_k', nargs="?", type=int, default=2)
    parser.add_argument('max_k', nargs="?", type=int, default=11)
    parser.add_argument('fit_rate', nargs="?", type=int, default=250)
    parser.add_argument('alp_buffer_size', nargs="?", type=int, default=500)
    parser.add_argument('random_task_ratio', nargs="?", type=int, default=0.2)
    parser.add_argument('alp_max_size', nargs="?", type=int, default=None)

    args = parser.parse_args("")

    return args

def setup_agent():

    args = build_parse()

    env = BaseGazeboUAVVelObsEnv1PCD(controller=None)

    args.state_size = env.state_size
    args.input_shape = env.state_size
    args.n_actions = env.action_space.shape[0]
    args.max_action = env.action_space.high
    args.min_action = env.action_space.low
    args.safe_max_action = env.safe_action_max
    args.safe_min_action = -env.safe_action_max

    if args.Algorithm == "DDPG":
        agent = DDPG.DDPG(args = args,policy = PolicyNetwork,critic = QNetwork,replayBuff = ReplayBuffer,exploration = OUActionNoise,vision = None)
    elif args.Algorithm == "TD3":
        agent = TD3.TD3(args = args,policy = PolicyNetwork,critic = QNetwork,replayBuff = ReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "SAC":
        agent = SAC.SAC(args = args,policy = GaussianPolicyNetwork,critic = QNetwork,valueNet=VNetwork,replayBuff = ReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "SoftQ":
        agent = SoftQ.SoftQ(args = args,policy = PolicyNetwork,critic = QNetwork,replayBuff = ReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "RCRL":
        agent = RCRL.RCRL(args = args,policy = PolicyNetwork,critic = QNetwork,multiplier=MultiplierNetwork,replayBuff = CostReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "SEditor":
        agent = SEditor.SEditor(args = args,policy = PolicyNetwork,critic = QNetwork,replayBuff = CostReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "USL":
        agent = USL.USL(args = args,policy = PolicyNetwork,critic = QNetwork,replayBuff = CostReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "SAAC":
        agent = SAAC.SAAC(args = args,policy = GaussianPolicyNetwork,critic = QNetwork,valueNet=VNetwork, replayBuff = CostReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "IDEA1":
        agent = IDEA1.IDEA1(args = args,policy = PolicyNetwork,critic = QNetwork,replayBuff = CostReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "IDEA2":
        agent = IDEA2.IDEA2(args = args,policy = SafePolicyNetwork,critic = QNetwork,replayBuff = CostReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "IDEA3":
        agent = IDEA3.IDEA3(args = args,policy = SafePolicyNetwork,critic = QNetwork,replayBuff = CostReplayBuffer,exploration = OUActionNoise)
    elif args.Algorithm == "IDEA4":
        agent = IDEA4.IDEA4(args = args,policy = PolicyNetwork,critic = QNetwork,nvp=RealNVP,replayBuff = CostReplayBuffer,exploration = OUActionNoise)

    return agent

def test(agent,pointcloud,pose,prev_vel):
    '''
    Function that returns the velocity of the UAV

    Parameters:
    - agent: RL policy object
    - pointcloud - PCD data in the form of np.array (Shape is either (1080,) or (1080,1))
    - pose - position of the UAV in the form of np.array (Shape is either (3,) or (3,1))
    - prev_vel - velocity of the UAV at previous timestep (Shape - (3,))

    '''

    if len(pointcloud.shape) == 2:
        pointcloud = pointcloud.reshape(-1)

    if len(pose.shape) == 2:
        pose = pose.reshape(-1)

    prp_state = np.concatenate((pose,pointcloud))
    prp_state = prp_state.reshape(1,-1)

    action = agent.choose_action(prp_state,"testing")

    prev_vel += action[0]
    prev_vel = np.clip(prev_vel,np.array([-0.7,-0.7,-0.7]),np.array([0.7,0.7,0.7]))

    return prev_vel

def get_pointcloud(subscriber,data):

    resp = subscriber(rospy.Time(0,0), rospy.get_rostime())

    if len(resp.cloud.points) == 0:
        return data
    
    points = np.array([[point.x,point.y,point.z] for point in resp.cloud.points])

    data[:points.shape[0],:] = points[:360,:]
    data[points.shape[0]:,:] = data[0,:]

    return data

def get_pose(publisher):

    response = publisher("bebop::base_link","ground_plane::link")
    response = response.link_state

    return np.array([response.pose.position.x,response.pose.position.y,response.pose.position.z])


if __name__=="__main__":

    rospy.init_node('test_rl')

    pointcloud_subscriber = rospy.ServiceProxy('assemble_scans', AssembleScans) 
    uam_publisher = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
    twist_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    pointcloud_data = np.zeros((360,3))
    pose = np.array([0,0,1])
    vel = np.array([0,0,0],dtype=np.float64)
    goal = np.array([1,1,1])
    agent = setup_agent()

    for i in range(100):

        pointcloud_data = get_pointcloud(pointcloud_subscriber,pointcloud_data)
        print(pointcloud_data)
        vel = test(agent,pointcloud_data.reshape(-1),goal - pose,prev_vel=vel)

        velocity = Twist()
        velocity.linear.x = vel[0]
        velocity.linear.y = vel[1]
        # velocity.linear.z = vel[2]
        twist_pub.publish(velocity)

        print(vel)

        time.sleep(0.05)

        pose = get_pose(uam_publisher)