#!/usr/bin/env python3

import rospy
import pickle
from geometry_msgs.msg import Twist

f = open("/home/shaswat/ros_ws/src/bebop_simulator/bebop_gazebo/src/velocity_random.pkl","rb")
vel_val = pickle.load(f)
f.close()

print(vel_val)

def init():
    rospy.init_node('test_rl')
    twist_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    for i in vel_val[:-4]:
        vel = Twist()
        vel.linear.x = i[0]
        vel.linear.y = i[1]
        vel.linear.z = 0
        twist_pub.publish(vel)
        rospy.sleep(0.07)

if __name__ == '__main__':
    init()
