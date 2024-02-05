# PID Controller for quadcopter

# import libraries
import numpy as np
import math
from rl_path_planning.environment.Dynamics import Quadrotor

class CascadeController():

    def __init__(self):
        Kp_pos = [.95, .95, 15.] # proportional [x,y,z]
        Kd_pos = [1.8, 1.8, 15.]  # derivative [x,y,z]
        Ki_pos = [0.2, 0.2, 1.0] # integral [x,y,z]
        Ki_sat_pos = 1.1*np.ones(3)  # saturation for integral controller (prevent windup) [x,y,z]

        # Gains for angle controller
        Kp_ang = [6.9, 6.9, 25.] # proportional [x,y,z]
        Kd_ang = [3.7, 3.7, 9.]  # derivative [x,y,z]
        Ki_ang = [0.1, 0.1, 0.1] # integral [x,y,z]
        Ki_sat_ang = 0.1*np.ones(3)  # saturation for integral controller (prevent windup) [x,y,z]
        self.dt = 0.01
        self.sim_start = 0
        self.sim_end = 15
        self.gravity = 9.8

        self.quadrotor = Quadrotor.Quadrotor(np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3))
        self.orientation_list = []
        self.motor_thrust_list = []
        self.torque_list = []
        self.ang_vel_list = []
        self.pos_list = []
        self.lnr_vel_list = []
        # Create quadcotper with position and angle controller objects
        self.pos_controller = PID_Controller(Kp_pos, Kd_pos, Ki_pos, Ki_sat_pos, self.dt)
        self.angle_controller = PID_Controller(Kp_ang, Kd_ang, Ki_ang, Ki_sat_ang, self.dt)

    def move_robot(self,pos_des,vel_des,ang_vel_des):

        # print(f"the desired pose is : {pos_des}")
        self.quadrotor.set_target(pos_des,vel_des,ang_vel_des)
        time_index = np.arange(self.sim_start, self.sim_end + self.dt, self.dt)
        quadcopter = self.quadrotor
        for time in enumerate(time_index):
            # print(time)
            #find position and velocity error and call positional controller
            pos_error = quadcopter.getPoseError(quadcopter.pos)
            vel_error = quadcopter.getLnrVelError(quadcopter.lnr_vel)
            des_acc = self.pos_controller.control_update(pos_error,vel_error)
            
            #Modify z gain to include thrust required to hover
            des_acc[2] = (self.gravity + des_acc[2])/(math.cos(quadcopter.ort[0]) * math.cos(quadcopter.ort[1]))
            
            #calculate thrust needed  
            thrust_needed = quadcopter.mass * des_acc[2]
            
            #Check if needed acceleration is not zero. if zero, set to one to prevent divide by zero below
            mag_acc = np.linalg.norm(des_acc)
            if mag_acc == 0:
                mag_acc = 1
            
            # print(f"the cos : {math.cos(quadcopter.ort[1])}")

            phi_des = np.clip(-des_acc[1] / mag_acc / math.cos(quadcopter.ort[1]),-1,1)
            theta_des = np.clip(des_acc[0] / mag_acc,-1,1)
            #use desired acceleration to find desired angles since the quad can only move via changing angles
            ang_des = [math.asin(phi_des),
                math.asin(theta_des),
                0]
            # #use desired acceleration to find desired angles since the quad can only move via changing angles
            # ang_des = [math.asin(-des_acc[1] / mag_acc / math.cos(quadcopter.ort[1])),
            #     math.asin(des_acc[0] / mag_acc),
            #     0]

            #check if exceeds max angle
            mag_angle_des = np.linalg.norm(ang_des)
            if mag_angle_des > quadcopter.max_angle:
                ang_des = (ang_des / mag_angle_des) * quadcopter.max_angle

            # call angle controller
            quadcopter.ort_des = ang_des
            ang_error = quadcopter.getOrtError(quadcopter.ort)
            ang_vel_error = quadcopter.getAngVelError(quadcopter.ang_vel)
            tau_needed = self.angle_controller.control_update(ang_error, ang_vel_error)
            #Find motor speeds needed to achieve desired linear and angular accelerations
            quadcopter.getDesSpeed(thrust_needed, tau_needed)
            # Step in time and update quadcopter attributes
            quadcopter.step()
            
            self.pos_list[0].append(quadcopter.pos[0])
            self.pos_list[1].append(quadcopter.pos[1])
            self.pos_list[2].append(quadcopter.pos[2])

            self.lnr_vel_list[0].append(quadcopter.lnr_vel[0])
            self.lnr_vel_list[1].append(quadcopter.lnr_vel[1])
            self.lnr_vel_list[2].append(quadcopter.lnr_vel[2])

            self.orientation_list[0].append(np.rad2deg(quadcopter.ort[0]))
            self.orientation_list[1].append(np.rad2deg(quadcopter.ort[1]))
            self.orientation_list[2].append(np.rad2deg(quadcopter.ort[2]))

            self.motor_thrust_list[0].append(quadcopter.speeds[0]*quadcopter.k1)
            self.motor_thrust_list[1].append(quadcopter.speeds[1]*quadcopter.k1)
            self.motor_thrust_list[2].append(quadcopter.speeds[2]*quadcopter.k1)
            self.motor_thrust_list[3].append(quadcopter.speeds[3]*quadcopter.k1)

            self.torque_list[0].append(quadcopter.tau[0])
            self.torque_list[1].append(quadcopter.tau[1])
            self.torque_list[2].append(quadcopter.tau[2])

            self.ang_vel_list[0].append(quadcopter.ang_vel[0])
            self.ang_vel_list[1].append(quadcopter.ang_vel[1])
            self.ang_vel_list[2].append(quadcopter.ang_vel[2])

            # print(quadcopter.pos)
        return quadcopter.get_states()

    def reset(self,pos,lnr_vel,ort,ang_vel):

        self.quadrotor = Quadrotor.Quadrotor(pos,ort,lnr_vel,ang_vel,np.zeros(3))
        self.orientation_list = [[],[],[]]
        self.motor_thrust_list = [[],[],[],[]]
        self.torque_list = [[],[],[]]
        self.ang_vel_list = [[],[],[]]
        self.lnr_vel_list = [[],[],[]]
        self.pos_list = [[],[],[]]

class PID_Controller:

    def __init__(self, Kp, Kd, Ki, Ki_sat, dt):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.Ki_sat = Ki_sat
        self.dt = dt
        
        # Integration total
        self.int = [0., 0., 0.]

    def control_update(self, pos_error, vel_error):
        
        #Update integral controller
        self.int += pos_error * self.dt

        #Prevent windup
        over_mag = np.argwhere(np.array(self.int) > np.array(self.Ki_sat))
        if over_mag.size != 0:
            for i in range(over_mag.size):
                mag = abs(self.int[over_mag[i][0]]) #get magnitude to find sign (direction)
                self.int[over_mag[i][0]] = (self.int[over_mag[i][0]] / mag) * self.Ki_sat[over_mag[i][0]] #maintain direction (sign) but limit to saturation 

        
        #Calculate controller input for desired acceleration
        des_acc = self.Kp * pos_error + self.Ki * self.int + self.Kd * vel_error
        return des_acc