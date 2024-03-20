import os
import csv
import numpy as np
import random
import math
import time
import traceback
from . import utils


class DroneManager():
    def __init__(self, supervisor, robot, drone_id, grid_size, origin_offset, tolerance=0.01, min_tolerance=0.05,
                  base_forward=0.025, base_altitude=0.1, forward_step=0.1, turn_step=0.5, n_agents=3, check_angle_tolerance=25, 
                  check_disp_tolerance=1.5, verbose=False, raise_exceptions=False, log_trajs=False, log_time_step=100) -> None:
        self.supervisor = supervisor
        self.robot = robot
        self.drone_id = drone_id
        self.grid_size = grid_size
        self.origin_offset = origin_offset
        self.tolerance = tolerance
        self.min_tolerance = min_tolerance
        self.max_forward_step = forward_step
        self.max_turn_step = turn_step
        self.n_agents = n_agents
        self.verbose = verbose
        self.forward_step = forward_step
        self.turn_step = turn_step
        self.raise_exceptions = raise_exceptions

        # Make angle tolerance a multiple of factor so that it increases as we increase speed factor
        self.stopped = True
        self.turned = False
        self.time_step = 0

        # Drone specific base variables
        self.base_forward = base_forward
        self.base_altitude = base_altitude #if drone_id != 1 else 0.15
        self.base_tolerance = tolerance
        self.max_height = 0.3
        self.altitude_diff = 0.05
        self.desired_altitude = self.base_altitude
        self.angle_tolerance = turn_step
        self.check_angle_tolerance = check_angle_tolerance
        self.check_disp_tolerance = check_disp_tolerance

        # Logging trajectories
        self.log_trajs = log_trajs
        self.log_time_step = log_time_step

        # Obstacle avoidance        
        self.avoid_time_step = 0
        self.avoid = False
        # Traj storage list
        self.trajs = []
        # Temp variables to keep tracks
        self.temp = None

        # Origin offset is half in this case. We should consider changing it later
        self.grid_2_continious = utils.grid_to_continuous_mapping(
            self.grid_size, self.origin_offset*2)
        

    # function to initialize the states and paramters of the webots world
    def init_world(self, initial_positions=None, grid_pos=False):
        """
        This function intializes the world and the drones

        It sets the PID object, the node and supervisor objects, etc

        Returns: True if successfully initialized
                 False if there is an exception
        """
        
        try:          
            # Set the pid controller
            self.PID = utils.pid_velocity_fixed_height_controller()
            self.node = self.supervisor.getSelf()
            self.pos = self.node.getField("translation")
            self.rot = self.node.getField("rotation")
            self.past_time = self.robot.getTime()
        
            # Initialize drone motors and devices
            timestep = int(self.robot.getBasicTimeStep())
            self.m1 = self.robot.getDevice("m1_motor")
            self.m2 = self.robot.getDevice("m2_motor")
            self.m3 = self.robot.getDevice("m3_motor")
            self.m4 = self.robot.getDevice("m4_motor")
            _ = [motor.setPosition(float('inf')) 
                 for motor in [self.m1, self.m2, self.m3, self.m4]]
            self.m1.setVelocity(-1)
            self.m2.setVelocity(1)
            self.m3.setVelocity(-1)
            self.m4.setVelocity(1)
            
            self.imu = self.robot.getDevice("inertial_unit")
            self.gps = self.robot.getDevice("gps")
            self.gyro = self.robot.getDevice("gyro")
            self.range_f = self.robot.getDevice("range_front")
            self.range_b = self.robot.getDevice("range_back")
            self.range_l = self.robot.getDevice("range_left")
            self.range_r = self.robot.getDevice("range_right")
            _ = [device.enable(timestep) for device in [
                self.imu, self.gps, self.gyro, self.range_f, self.range_b,
                self.range_l, self.range_r]]

            """
            It is possible to get the properties of all other drones in the environment
            and set them at once. I did not implement this
            """
            #if world_agent_lists is not None:
            #    pass 

            # If a list of initial positions is passed
            if initial_positions:
                # Get the initial position for the agent
                pos = initial_positions[self.drone_id]

                if grid_pos:
                    pos = self.map_positions(pos)
                # 0.015 is the crazyflie drone height
                temp = list(pos) + [0.015]
                # Set the position of the drone from the supplied value
                self.pos.setSFVec3f(temp)
                # Set the old pos as pos
                self.old_pos = pos
            
            # Get the current position of the drone as the old position
            else:
                temp = self.pos.getSFVec3f()
                self.old_pos = tuple(temp[0], temp[1])
                self.new_pos = self.old_pos
            
            # Set the rotation so that all agents are facing the true North (90 degrees)
            self.rot.setSFRotation([0.0,0.0,1.0,1.5708])
            
            # Update the PID inputs and return True if successful
            self.past_time = self.robot.getTime()
            self.past_x_global = self.gps.getValues()[0]
            self.past_y_global = self.gps.getValues()[1]
            self.altitude = self.gps.getValues()[2]
            return self.update_pid_inputs()
        
        except Exception as e:
            print("Exception: ", e)
            if self.raise_exceptions:
                raise
            else:
                traceback.print_exc()
    
        return False


    # Updates using the value stored in self.motor_power
    def update_motors(self):
        try:
            #print("Motor power: ", self.motor_power)
            self.m1.setVelocity(-self.motor_power[0])
            self.m2.setVelocity(self.motor_power[1])
            self.m3.setVelocity(-self.motor_power[2])
            self.m4.setVelocity(self.motor_power[3])
            return True
        
        except Exception as e:
            print("Exception while updating motor values: ", e)
                    
            if self.raise_exceptions:
                raise
            else:
                traceback.print_exc()
                return False
        

    # Compute motor power and apply to the drone
    def power_motors(self, forward=0, sideways=0, yaw=0, height=0):
        self.motor_power = self.PID.pid(self.dt, forward, sideways, yaw, height,
                                        self.roll, self.pitch, self.yaw_rate,
                                        self.altitude, self.v_x, self.v_y)
        '''
        if self.verbose:
            print("Drone:", self.drone_id, "Self.motor power values: ", self.motor_power)
        #'''
        
        return self.update_motors()


    # Stop drone from moving and keep it hovering in a fixed position
    def stop(self):
        
        # Camly bring robot to desired height
        height = self.desired_altitude
        if self.altitude > self.desired_altitude:
            #self.desired_altitude -= self.altitude_diff * self.dt
            height = min(self.desired_altitude*1.1, self.altitude - self.desired_altitude * self.dt)

        # If drone x-velocity is greater than 0, derate it
        if self.verbose: 
            print("v_x:", self.v_x, "altitude:", self.altitude, "desired: ", self.desired_altitude, "height:", height)
        if self.v_x > self.tolerance/4:
            self.forward_step = self.forward_step/2
        else:
            self.forward_step = 0

        if self.power_motors(forward=self.forward_step, height=height):
            self.stopped = True
            return self.stopped
        
        return False
    

    def start(self):
        """
        Attempt to start the drone or make it go higher by calling this function 
        
        It increases the height of the drone and calls the power_motors function
        using the computed values

        Returns: True if altitude is higher than 90% of desired altitude
                 
                 False otherwise

        """
        self.power_motors(height=self.desired_altitude)


    def land(self):
        """
        Attempt to land the drone by calling this function 

        ! Not yet working as intended
        
        It reduces the height of the drone and calls the power_motors function
        using the computed values
        """
        self.desired_altitude -= self.altitude_diff * self.dt
        #if self.verbose: print("Height in land: ", self.desired_altitude)
        self.power_motors(height=self.desired_altitude)


    def update_targets(self):
        """
        Update required positional informations including current position, current angle of drone
        
        This function also computes the reference displacement and angles. 
        These reference values are the absolute values of the continous env
        They are computed using old_conv_pos and new_cont_pos

        Returns: True for successful update, False if any exception occured
        """
        try:
            temp = self.rot.getSFRotation()
            self.curr_angle = utils.radians_to_degrees(temp[-1], z_axis=temp[-2])

            temp = self.pos.getSFVec3f()
            self.curr_pos = temp[:-1]

            self.target_angle, self.target_disp = utils.compute_angle_and_displacement(
                self.curr_pos, self.new_pos
            )
            self.ref_angle, self.ref_disp = utils.compute_angle_and_displacement(
                self.old_pos, self.new_pos
            )

            return True

        except Exception as e:
            print("Exception while updating targets: ", e)
            if self.raise_exceptions:
                raise
            else:
                traceback.print_exc()
                return False
    

    def update_pid_inputs(self):
        """
        Update the PID inputs based on the values from the drone IMU and gps

        Returns: True if successful
                 False - with an exception that can be raised or not - self.raise_exceptions
        """

        try:
            self.dt = self.robot.getTime() - self.past_time
            self.dt = 0.032 if self.dt == 0 else self.dt       # Prevent zero division
            self.roll = self.imu.getRollPitchYaw()[0]
            self.pitch = self.imu.getRollPitchYaw()[1]
            self.yaw = self.imu.getRollPitchYaw()[2]
            self.yaw_rate = self.gyro.getValues()[2]
            self.x_global = self.gps.getValues()[0]
            self.y_global = self.gps.getValues()[1]
            self.altitude = self.gps.getValues()[2]
            v_x_global = (self.x_global - self.past_x_global)/self.dt
            v_y_global = (self.y_global - self.past_y_global)/self.dt

            # Get body fixed velocities
            cos_yaw = math.cos(self.yaw)
            sin_yaw = math.sin(self.yaw)
            self.v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
            self.v_y = - v_x_global * sin_yaw + v_y_global * cos_yaw
        
            return True

        except Exception as e:
            print("Exception while updating PID inputs: ", e)
            if self.raise_exceptions:
                raise
            else:
                traceback.print_exc()
                return False
    

    def update_range_vals(self):
        """
        Compute the range in meters of any obstacle on the drone path

        Return: True for successful computation, False for any Exception
        """
        try:
            self.range_f_val = self.range_f.getValue() / 1000
            self.range_r_val = self.range_r.getValue() / 1000
            self.range_l_val = self.range_l.getValue() / 1000

            if self.verbose:
                print('~~~')
                print("Drone", self.drone_id, "Range vals: front: ", self.range_f_val, "right: ", 
                    self.range_r_val, "left: ", self.range_l_val)
                print('~~~')
            return True

        except Exception as e:
            print("Exception while updating range inputs: ", e)
            if self.raise_exceptions:
                raise
            else:
                traceback.print_exc()
                return False
        

    def set_new_pos(self, pos, grid_pos=False):
        """
        Sets the new position for the drone

        Returns: True
        """

        if grid_pos:
            pos = self.map_positions(pos)
        
        self.new_pos = pos
        self.time_step += 1

        return True
        

    def turn_by_angle(self, check=False, angle=0):
        """        
        Turn the drone using the target angle value computed in the update_targets() function

        Args: check - boolean to determine if we are checking for drift or turning
              angle - angle value to be used for checking drift 
                    - self.check_angle_tolerance is used by default
        
        This function sets the self.turned boolean to specify that a turn has happened or not

        Returns: True - (1) old and new grid values are equal - **No loger used**
                        (2) if target displacement is less than self.tolerance*2
                            this prevents attempts to turn when target angle values are wrong
                            due to floating point values in calculations
                        (3) If angle is within bound of bottom and top angle
                            i.e. target angle + and - angle tolerance
                
                False - (1) If robot altitude is less than 90% of base altitude - 
                            calls the self.start() function
                        (2) Angle is not within bound
                None - if there is an exception or the function ran till the end
        
        Actions:
                Computes the yaw needed to turn the drone and calls the power_motors function
        """  
        #self.turned = False     # Flag for turning 
       
        if self.altitude < self.desired_altitude*0.9:
            if self.verbose: 
                print("Drone", self.drone_id, "Drone not yet at desired altitude.", "Altitude: ", 
                      self.altitude)
                print("Desired altitude is: ", self.desired_altitude)
            self.start()
            return False
        
        # Calculating time for turn
        if self.temp is None and not check:
            self.temp = True
            self.temp1 = self.robot.getTime()
            #if self.verbose: print("Drone ", self.drone_id, "Turn started at : ", self.temp1)

        # The turn ratio - Keep it positive if we are turning CCW i.e left
        yaw = self.turn_step

        # Check if targets and PID inputs have been successfully updated
        if self.update_targets():
            
            """
            This is a crude approach to preventing drone from turning when close to final location
            Sometimes the angle diverges so greatly when the displacement is close to 0.01 or 0.02
            This errornously make stop to turn remain true so we turn it off when checking
            """
            if (self.target_disp < self.tolerance*1.5):
                # Make turn check false if we are getting close to target
                if check: return False

                self.turned = True
                return self.turned

            """
            If we are checking turn angle divergence, use the angle passed in or 20 as default
            Also, if robot is moving, allow up to 20 degrees angle divegence to prevent unneccessary stopping
            If robots keep crashing, allow wide angle divergence so it does not arbritarily try to turn
            without stopping
            """
            target_angle = self.target_angle
            angle_tolerance = self.angle_tolerance

            # Only use the strict angle tolerance if we are stopped - if checking or moving, dont use it
            if check or not self.stopped:
                angle_tolerance = self.check_angle_tolerance if angle == 0 else angle
            
            # Angle to positive ensures negative angles become positive values
            top_angle = utils.angle_to_positive(target_angle + angle_tolerance)
            bottom_angle = utils.angle_to_positive(target_angle - angle_tolerance)

            # Set direction based on angle size - to turn right we set right motor to roll backwars (-1)
            # Trun to the right - Clockwise as angle is larger than target or in the quadrant - set desired yaw negative
            if self.curr_angle > target_angle: #or (self.curr_angle < 90 and target_angle > 270):
                yaw = -yaw
            
            # This is probably buggy and might be the reason turn direction is arbitrary
            # If we are in Q4, then we should go CCW to if we are going to Q1
            if self.curr_angle > 275 and target_angle < 85:
                yaw = abs(yaw)
            
            bottom_q4 = -1
            top_q4 = -1
            # This will only happen when the target is around 0 and bottom becomes like 355, etc
            if bottom_angle > top_angle:
                bottom_q4 = bottom_angle
                top_q4 = 360
                bottom_angle = 0

            # If we do not need to turn we can stop here
            # The second part of this if will always be false except when bottom is greater than top
            if (utils.value_in_range(self.curr_angle, bottom_angle, top_angle) or 
                utils.value_in_range(self.curr_angle, bottom_q4, top_q4)):

                # This means we are not to stop and turn during check - value is within range
                if check: 
                    return False

                if self.temp:
                    if self.verbose:
                        print("Drone ", self.drone_id, "Turned ended at ", self.robot.getTime()-self.temp1)
                    self.temp=None

                # If we are within range, set turned to True
                self.turned = True
                return self.turned
            
            # This means we are to stop and turn if we are checking - value not within range
            if check: 
                return True
            
            # Rotate the drone
            try:
                # We should only attempt to turn while robot is stopped
                if self.stopped:
                    self.power_motors(yaw=yaw, height=self.desired_altitude)
                
                else:
                    if self.verbose:
                        print('^^^^^')
                        print("Drone ID: ", self.drone_id, "Trun failed!!! Drone cannot turn as it is moving")
                        print('^^^^^')
                    self.stop_to_turn()
                
                if self.verbose:
                    print('+++++')
                    print("\nAgent ID: ", self.drone_id, "Curr angle: ", 
                          self.curr_angle, "Target angle: ", self.target_angle, 
                          "Ref angle: ", self.ref_angle)
                    print("Bottom angle: ", bottom_angle, "Top angle: ", top_angle)
                    print("Turned status: ", self.turned, "\n")
                    print('+++++')
                    
                    # We are definately outside range. Set turned to False
                    self.turned = False
                return self.turned
            
            except Exception as e:
                print("Exception while turning by angle: ", e)
                if self.raise_exceptions:
                    raise
                else:
                    traceback.print_exc()
                    return None
            
        return None


    # Move drone by current displacement and target displacement
    def move_by_disp(self):
        """
        Move the drone using the displacement value computed in the update_targets() function

        Returns: True - (1) target displacement is less than provided tolerance
                        (2) old and new grid values are equal
                False - otherwise
                None - if there is an exception or the function ran till the end

                self.stopped as True -  (1) If next action is not equal to current action
                                False otherwise
        """  
        # Wait until drone is in air before moving and start the drone if needed
        # Use 90% of base altitude
        #if not self.start():
        if self.altitude < self.desired_altitude*0.9:
            if self.verbose: 
                print("Drone not yet at desired altitude.", "Altitude: ", self.altitude)
            self.start()

            return False, self.stopped

        if self.temp is None:
            self.temp1 = self.robot.getTime()
            self.temp = True
                     
        # Ensure targets are updated before proceeding 
        if self.update_targets():
            
            # Prevent movement if we are at the final position or temp grid pos
            if self.target_disp < self.tolerance:
                
                self.stopped = True
                if self.temp:
                    if self.verbose: 
                        print("Drone ", self.drone_id, "Move ended at: ", self.robot.getTime() - self.temp1)
                        self.temp = None
                return True, self.stopped
            
            # If we are not at the final position
            try:
                # Reset the forward step if we are to resume motion 
                if self.forward_step < self.base_forward and self.stopped:
                    self.forward_step = self.max_forward_step

                self.stopped = False        # Flag to tell robot to stop 
                forward_step = self.forward_step #if sideways == 0 else 0 

                temp = abs(self.m1.getVelocity())
                sideways = 0
                
                #This function can now correct drone position on the fixed axis while moving
                # if desired altitude is less than desired value, increase steadily
                self.power_motors(forward=forward_step, sideways=sideways, height=self.desired_altitude)

                if self.verbose:
                    print('***')
                    print("Drone ID: ", self.drone_id, "displacement: ", self.target_disp,
                          "curr_pos: ", self.curr_pos, "new pos: ", self.new_pos, 
                          "old pos", self.old_pos, "Stopped: ", self.stopped)
                    print("motor 1 speed: ", temp, "drone v_x: ", self.v_x, "drone v_y: ", self.v_y,
                          "dt: ", self.dt, "\naltitude: ", self.altitude, "forward", forward_step, 
                          "sideways: ", sideways, "height: ", self.desired_altitude)

                return False, self.stopped
            
            except Exception as e:
                print("Exception while moving by displacement: ", e)
                if self.raise_exceptions:
                    raise
                else:
                    traceback.print_exc()
                    return None, None
        
        return None, None
    
    
    def stop_to_turn(self):
        """
        Logic: Return false always except if displacement has become too large than y or 
        angle has strayed x degrees or more from desired angle in either direction

        y = self.check_disp_tolerance
        x = self.check_angle_tolerance

        Returns: True - If drone should stop to turn
                 False - otherwise
        """

        # Update targets before running this - also ensure drone is at desired hieght
        if not self.update_targets() or self.altitude < self.desired_altitude*0.9:
            return
        
        turn_check = self.turn_by_angle(check=True)

        disp_check = self.ref_disp * self.check_disp_tolerance < self.target_disp
 
        # Only set stopped if needed else leave it as is
        if (self.ref_disp > 0 and disp_check) or (turn_check and self.ref_angle > 0):
            self.stopped = True
            self.turned = False
            return True
        
        return False


    def avoid_obstacles(self):
        """
        The idea is to fly higher so that the obstacle can pass, move one grid and then come down
        
        **Not yet working as inteneded
        """
        # If there is an obstacle, go up
        '''
        if self.range_f_val < 0.15 and not self.avoid:
            self.avoid_time_step = self.time_step
            self.desired_altitude = self.base_altitude*1.8
            self.avoid = True
        
        # After going up for one full grid cell, come back down
        if (self.time_step > self.avoid_time_step+2) and self.avoid:
            self.desired_altitude = self.base_altitude
            self.avoid = False
        #'''
        if self.range_f_val < 0.14:
            if not self.avoid and self.altitude > self.desired_altitude*0.9:
                self.avoid = True
                self.stopped = True
                self.turned = True
        
        elif self.range_f_val > 0.3:
            self.avoid = False
        
        """
        TODO - Simple obstacle avoidance logic:

        If the right is free, turn to the right and move one grid else turn to the left
        """
        if self.avoid:
            if self.range_l_val > 1:
                self.curr_angle = self.curr_angle + 90

        return self.avoid


    def map_positions(self, grid_pos):
        """
        Convert grid positions to continous positions by picking the values from the 
        precomputed numpy array

        Args: grid_pos - Tuple of agent position in the gridworld

        Return: Equivalent of grid position in the continious world
        """
        return self.grid_2_continious[int(grid_pos[0]), int(grid_pos[1])] 
    

    def update_past_values(self):
        """
        Update all past values needed for computing self.dt and drone velocities

        This function also log trajectories values to compute positional trace, etc
        """

        self.past_time = self.robot.getTime()
        self.past_x_global = self.x_global
        self.past_y_global = self.y_global

        if self.log_trajs:
            self.trajs.append(self.curr_pos)

        # If we are logging and we are at the end of the episode - or the specified log_time_step
        if self.log_trajs and self.time_step > self.log_time_step:
            path = os.path.join(os.getcwd(), "crazyflie")

            if not os.path.exists(path):
                os.makedirs(path)
            
            f_name = os.path.join(path, f"drone_{self.drone_id}_trajs.csv")

            write_mode = 'w' if not os.path.exists(f_name) else 'a'
            
            # Write data to the CSV file
            with open(f_name, mode=write_mode, newline='') as file:                
                writer = csv.writer(file)
                if write_mode == 'w':
                    writer.writerow(["pos_x", "pos_y"])
                writer.writerows(self.trajs)
            
            # Allow the main controller to save a snapshot of the environment
            if self.drone_id == 0:
                f_name = os.path.join(path, f"world_view.jpg")
                self.supervisor.exportImage(f_name, 100)

            if self.verbose:
                print("------")
                print(f"Trajectories successfully logged for drone {self.drone_id}")
                print(f"Log path: {f_name}")
                print("------")
            self.log_trajs = False


    def stop_and_hover(self, landing=False):
        """
        Stop robot and let it hover in a fixed position

        Args: landing - If true, it does not stop and returns so that drone can land
        """
        if landing: return

        if self.verbose:
            print("Turned:", self.turned, "Stopped:", self.stopped)

        if (self.turned and self.stopped):

            if self.altitude > self.desired_altitude * 0.9:
                self.stop()
            
            else:
                self.start()
                if self.verbose: 
                    print("Drone not at desired altitude. Starting ...")
