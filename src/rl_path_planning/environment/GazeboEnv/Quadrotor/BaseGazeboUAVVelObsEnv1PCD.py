import gym
import rospy
from std_msgs.msg import String 
from math import pi
from gym import spaces
import numpy as np
import time
from laser_assembler.srv import AssembleScans
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.msg import LinkState
from sensor_msgs.msg import LaserScan
import tf

import tf2_ros
import geometry_msgs.msg

class BaseGazeboUAVVelObsEnv1PCD(gym.Env):
    
    def __init__(self,controller = None): 
        
        self.uam_publisher = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)
        self.lidar_subscriber = rospy.Subscriber('/laser_controller/out', LaserScan, self.lidar_callback)
        self.pointcloud_subscriber = rospy.ServiceProxy('assemble_scans', AssembleScans)
        # self.collision_sub = CollisionSubscriber()

        self.state = None
        self.state_size = 1083
        self.action_max = np.array([0.3,0.3,0.3])

        self.pointcloud_data = np.zeros((360,3))
        self.lidar_range = None
        
        self.q = None
        self.qdot = None
        self.q_des = None
        self.qdot_des = None
        self.qdotdot_des = None
        self.man_pos = None
        self.manip_difference = None

        self.max_time = 10
        self.dt = 0.07
        self.current_time = 0

        self.q_vel_bound = np.array([3,3,3,1.5,1.5,1.5,1.5,1.5,1.5,1.5])
        self.max_q_bound = np.array([1.5,1.5,1.5])
        self.min_q_bound = np.array([-1.5,-1.5,-1.5])

        self.max_q_safety = np.array([8,8,8])
        self.min_q_safety = np.array([-8,-8,2])
        # self.max_q_safety = None
        # self.min_q_safety = None

        self.max_safety_engage = np.array([5.5,5.5,5.5])
        self.min_safety_engage = np.array([-5.5,-5.5,0.8])

        self.safe_action_max = np.array([8,8,8])
        self.safe_action_min = np.array([-8,-8,2])

        self.action_space = spaces.Box(-self.action_max,self.action_max,dtype=np.float64)

    def step(self, action):
        
        action = action[0]
        self.vel = self.vel + action[:3]

        self.vel = np.clip(self.vel,self.min_q_bound,self.max_q_bound)
        self.pose = np.array([self.dt*self.vel[i] + self.pose[i] for i in range(self.vel.shape[0])])
        self.pose = np.clip(self.pose,np.array([-12,-12,0.5]),np.array([12,12,3]))
        self.publish_simulator(self.pose)
        self.send_tf(self.pose)

        _,self.check_contact = self.get_lidar_data()
        
        # self.check_contact = self.collision_sub.get_collision_info()

        # print(f"New pose : {new_q}")
        # print(f"New velocity : {new_q_vel}")
        # self.q,self.qdot = self.controller.solve(new_q,new_q_vel)

        self.const_broken = self.constraint_broken()
        self.pose_error = self.get_error()
        reward,done = self.get_reward()
        constraint = self.get_constraint()
        info = self.get_info(constraint)

        if done:

            # self.publish_simulator(np.array([0.0,0.0,2.0]))
            
            print(f"The constraint is broken : {self.const_broken}")
            print(f"The position error at the end : {self.pose_error}")
            print(f"The end pose of UAV is : {self.pose[:3]}")

        pcd = self.get_pointcloud()
        pose_diff = self.q_des - self.pose
        prp_state = np.concatenate((pose_diff,pcd))
        prp_state = prp_state.reshape(1,-1)
        self.current_time += 1

        if self.const_broken:

            self.get_safe_pose()
            self.publish_simulator(self.previous_pose)
            self.pose = self.previous_pose

            # self.reset_sim.send_request(uav_pos_ort)

            self.vel = self.vel - action[:3]
            # self.publish_simulator(self.vel)

        return prp_state, reward, done, info

    def get_reward(self):
        
        done = False
        pose_error = self.pose_error
        reward = 0
        if not self.const_broken:
            self.previous_pose = self.pose
            # if pose_error < 0.01:
            #     done = True
            #     reward = 1000
            # elif pose_error < 0.05:
            #     done = True
            #     reward = 100
            if pose_error < 0.1:
                done = True
                reward = 10
            # elif pose_error < 1:
            #     done = True
            #     reward = 0
            else:
                reward = -(pose_error*10)
        
        else:
            reward = -20
            if self.algorithm == "SAC" and self.algorithm == "SoftQ":
                done = True

        if self.current_time > self.max_time:
            done = True
            reward -= 2

        return reward,done
    
    def get_constraint(self):
        
        constraint = 0
        if self.const_broken:

            for i in range(self.vel.shape[0]):
                if self.vel[i] > self.max_q_bound[i]:
                    constraint+= (self.vel[i] - self.max_q_bound[i])*10
                elif self.vel[i] < self.min_q_bound[i]:
                    constraint+= abs(self.vel[i] - self.min_q_bound[i])*10

            if constraint < 0:
                constraint = 10
        else:

            for i in range(self.vel.shape[0]):
                constraint+= (abs(self.vel[i]) - self.max_q_bound[i])*10

        return constraint

    def get_info(self,constraint):

        info = {}
        info["constraint"] = constraint
        info["safe_reward"] = -constraint
        info["safe_cost"] = 0
        info["negative_safe_cost"] = 0
        info["engage_reward"] = -10

        if np.any(self.vel > self.max_q_safety) or np.any(self.vel < self.min_q_safety):
            info["engage_reward"] = 10
            
        if constraint > 0:
            info["safe_cost"] = 1
            info["negative_safe_cost"] = -1

        return info

    def constraint_broken(self):
        
        if self.check_contact:
            return True
        
        # if np.any(self.vel[:3] > self.max_q_bound[:3]) or np.any(self.vel[:3] < self.min_q_bound[:3]):
        #     return True
        
        return False
    
    def get_error(self):

        pose_error =  np.linalg.norm(self.pose - self.q_des) 

        return pose_error
        
    def reset(self):

        #initial conditions
        self.pose = np.array([0,0,2])
        self.vel = np.array([0,0,0])
        self.previous_pose = np.array([0,0,2])
        # self.qdot = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]) #initial velocity [x; y; z] in inertial frame - m/s
        
        self.q_des = np.random.randint([-1,-1,1],[2,2,4])
        # self.check_contact = False
        # self.qdot_des = np.zeros(self.qdot.shape)
        # self.qdotdot_des = np.zeros(self.qdot.shape)
        print(f"The target pose is : {self.q_des}")
        self.publish_simulator(self.pose)
        self.send_tf(self.pose)
        _,self.check_contact = self.get_lidar_data()
        pcd = self.get_pointcloud()
        # print(f"the man pose : {self.man_pos}")
        pose_diff = self.q_des - self.pose
        # pose_diff = np.clip(self.q_des - self.man_pos,np.array([-1,-1,-1]),np.array([1,1,1]))
        prp_state = np.concatenate((pose_diff,pcd))
        prp_state = prp_state.reshape(1,-1)
        self.current_time = 0
        self.const_broken = False
        self.max_time = 10
        time.sleep(0.1)

        return prp_state
    
    def reset_test(self,q_des,max_time,algorithm):

        #initial conditions
        self.pose = np.array([0.0,0.0,2.0])
        # self.pose = np.array([0.0,0.0,2.0])
        self.vel = np.array([0,0,0])
        self.previous_pose = self.pose
        self.algorithm = algorithm

        self.q_des = q_des
        self.max_time = max_time
        # self.check_contact = False
        # self.qdot_des = np.zeros(self.qdot.shape)
        # self.qdotdot_des = np.zeros(self.qdot.shape)
        print(f"The target pose is : {self.q_des}")

        self.publish_simulator(self.pose)
        _,self.check_contact = self.get_lidar_data()
        pcd = self.get_pointcloud()
        # print(f"the man pose : {self.man_pos}")
        pose_diff = self.q_des - self.pose
        # pose_diff = np.clip(self.q_des - self.man_pos,np.array([-1,-1,-1]),np.array([1,1,1]))
        prp_state = np.concatenate((pose_diff,pcd))
        prp_state = prp_state.reshape(1,-1)
        self.current_time = 0
        self.const_broken = False
        time.sleep(0.1)

        return prp_state
    
    def publish_simulator(self,q):

        base_link_state = LinkState()
        base_link_state.link_name = "base_link"
        base_link_state.pose.position.x = q[0]
        base_link_state.pose.position.y = q[1]
        base_link_state.pose.position.z = q[2]

        base_link_state.pose.orientation.w = 1

        response = self.uam_publisher(base_link_state)

    def lidar_callback(self, msg):

        self.lidar_range = msg.ranges
    
    def get_state(self):

        if self.lidar_range is None:
            return np.zeros(shape=(360)),False
        
        lidar_data = np.array(self.lidar_range)
        contact = False
        for i in range(lidar_data.shape[0]):
            if lidar_data[i] == np.inf:
                lidar_data[i] = 1
            elif lidar_data[i] < 0.3:
                contact = True

        return lidar_data,contact
        
    def get_lidar_data(self):

        data,contact = self.get_state()
        return data,contact
    
    def get_pointcloud(self):

        resp = self.pointcloud_subscriber(rospy.Time(0,0), rospy.get_rostime())

        if len(resp.cloud.points) == 0:
            return self.pointcloud_data.reshape(-1)
        
        points = np.array([[point.x,point.y,point.z] for point in resp.cloud.points])

        self.pointcloud_data[:points.shape[0],:] = points[:360,:]
        self.pointcloud_data[points.shape[0]:,:] = self.pointcloud_data[0,:]

        pcd = self.pointcloud_data.reshape(-1)

        return pcd
    
    def send_tf(self,q):

        broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_transformStamped = geometry_msgs.msg.TransformStamped()

        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "world"
        static_transformStamped.child_frame_id = "base_link"

        static_transformStamped.transform.translation.x = float(q[0])
        static_transformStamped.transform.translation.y = float(q[1])
        static_transformStamped.transform.translation.z = float(q[2])

        static_transformStamped.transform.rotation.x = 0
        static_transformStamped.transform.rotation.y = 0
        static_transformStamped.transform.rotation.z = 0
        static_transformStamped.transform.rotation.w = 1

        broadcaster.sendTransform(static_transformStamped)
    
    def get_safe_pose(self):

        # for i in range(len(self.previous_pose) - 1):

        py = self.pose[1] - self.previous_pose[1]
        px = self.pose[0] - self.previous_pose[0]

        if (py > 0 and px > 0) or (py < 0 and px < 0):

            if py > 0:
                self.previous_pose[0]+= 0.05
                self.previous_pose[1]-= 0.05
            else:
                self.previous_pose[0]-= 0.05
                self.previous_pose[1]+= 0.05

        else:

            if py > 0:
                self.previous_pose[0]-= 0.05
                self.previous_pose[1]-= 0.05
            else:
                self.previous_pose[0]+= 0.05
                self.previous_pose[1]+= 0.05