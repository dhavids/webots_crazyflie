import os
import sys
import time
import json
from pathlib import Path

# Add the webots_crazyflie folder to path
file_path = Path(__file__)
# The main.py controller is in - /webots_crazyflie/crazyflie/controllers/main/main.py
# So the base path is parent 3
webots_crazyflie_path = file_path.parents[3]
sys.path.append(webots_crazyflie_path.as_posix())

from src.DroneManager import DroneManager
from src import utils
from controller import Robot, Supervisor, Keyboard

FLYING_ATTITUDE = 1

start_t = time.time()
# Load config file - controller path should be - parent 1
config_path = os.path.join(file_path.parents[1], 'config.json')
with open(config_path) as f:
    base_config = json.load(f)
# Update the base config dict using the recursive dict update method
config = utils.recursive_update(
    base_config["default"], base_config["crazyflie"])

drone_id = 0
# time in [ms] of a simulation step
TIME_STEP = config["timestep"]   
grid_size = config["grid_size"]

print("Time to load: ", time.time() - start_t, "ms")


# This should be set to half of the field
origin_offset = config["origin_offset"]
tolerance = config["tolerance"]
n_agents = config["n_agents"]
verbose_list = config["verbose"]
forward_step = config["forward_step"]
turn_step = config["turn_step"]
log_trajs = config["log_trajs"]
min_tolerance = config["min_tolerance"]

# create the Robot instance.
robot = Robot()
# Initialize supervisor
supervisor = Supervisor()
robot_Id = robot.getName()
manager = DroneManager(
    supervisor, robot, drone_id, grid_size, origin_offset, tolerance=tolerance, 
    min_tolerance=min_tolerance, forward_step=forward_step, turn_step=turn_step,
    n_agents=n_agents, log_trajs=log_trajs,
    verbose=verbose_list[drone_id])

# Initial positions - this is for agent 0 only
initial_positions = [(2,5)]
status = manager.init_world(initial_positions, grid_pos=True)
if status:
    print("Manager status: ", status)

# get the time step of the current world.
counter = 1
display = True
pos_x = None
pos_y = None
new_pos = False
just_turned = False

timestep = int(robot.getBasicTimeStep())
# Get keyboard
keyboard = Keyboard()
keyboard.enable(timestep)
camera = robot.getDevice("camera")
camera.enable(timestep)

print("====== Information ======")
print("Only values between 0 - 9 are accepted for now")
print("0,0 is the top left corner of the environment")
print("9,9 is the bottom right corner of the environment")
print("=========================")


# Main loop:
while robot.step(timestep) != -1:

    # Update PID inputs and range vals
    manager.update_pid_inputs()
    manager.update_range_vals()
    # Uncomment to see a camera vidw from the drone
    #camera_data = camera.getImage()

    # First we get the new position
    if not new_pos:
        # Show the values
        if display:
            if pos_x is None:
                print('-----')
                print("Enter X destination for Drone: ")
            else:
                print('-----')
                print("Now Enter Y destination for Drone: ")
            display = False
        
        key = keyboard.getKey()
        while key > 0:
            #print(key)
            """
            TODO
            It is possible to enter float values by accumunating the values in a list
            until the user presses the 'Enter' key and then convert the entered values
            into a float

            This is also needed if the grid_size is greater than 10
            """
            val = int(key - 48)

            # Set the value entered
            if pos_x is None: 
                pos_x = val
                time.sleep(0.2) # Debounce
            else:
                pos_y = val
                time.sleep(0.2) # Debounce
            # Refresh the display boolean
            display = True
            key = keyboard.getKey()

            print("Value Entered:", val)

        if pos_x is not None and pos_y is not None:
            pos = (pos_x, pos_y)
            print("New Position: ", pos)
            new_pos = manager.set_new_pos(pos, grid_pos=True)
    
    # Avoid obstacles if needed - This blocks all other code
    avoid = manager.avoid_obstacles()

    if new_pos and not avoid:

        # If we have not just turned or moved
        if not just_turned:
            turned = manager.turn_by_angle()

            # This means an exception occured
            if turned is None:
                break

            # successful turn or no need for turn
            if turned:
                just_turned = True
        

        # If we have just turned
        if just_turned:

            # If we need to turn, stop the robot
            if manager.stop_to_turn():
                just_turned = False
            
            moved, _ = manager.move_by_disp()

            # If exception while moving
            if moved is None:
                break
        
            # Successful move or need for move
            if moved:
                just_turned = False
                new_pos = False
                pos_x = None
                pos_y = None
                counter += 1

    manager.stop_and_hover()
    manager.update_past_values()
