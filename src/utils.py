import numpy as np
import math

def map_positions(grid_pos, grid_size=10, continuous_range=1):

    global grid_2_continious

    if not grid_2_continious:
        grid_2_continious = grid_to_continuous_mapping(grid_size, continuous_range)

    return grid_2_continious[grid_pos[0], grid_pos[1]]


def map_coordinate(x, y, max_range):
    # Scale the x and y coordinates to the range (0, max_range)
    mapped_x = (x + max_range / 2) / max_range
    mapped_y = 1.0 - (y + max_range / 2) / max_range  # Invert the y-axis to match the coordinate system

    return mapped_x, mapped_y  


def compute_angle_and_displacement(p1, p2, map_points=False):
    """
    Compute the angle and displacement between two points on the x-y plane

    Args: p1 and p2 - tuples of x and y
    
    Returns: angles in degrees
             displacement as float value
    """
    
    x1, y1 = p1 if not map_points else map_coordinate(p1[0], p1[1])
    x2, y2 = p2 if not map_points else map_coordinate(p2[0], p2[1])
    
    # Compute displacement
    displacement = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Compute angle
    angle = math.atan2(y2 - y1, x2 - x1)
    # Round to the nearest tens
    angle_degrees = round(math.degrees(angle), -1)
    
    # Adjust angle to be between 0 and 360 degrees
    angle_degrees = angle_degrees % 360

    return angle_degrees, displacement


def calculate_terminal_velocity(initial_velocity, deceleration_rate, total_distance, total_time):
    """
    Calculate the terminal velocity required to cover a certain distance while decelerating
    and ensuring the total travel time.

    Args:
    - initial_velocity (float): Initial velocity of the vehicle in m/s.
    - deceleration_rate (float): Rate of deceleration in seconds.
    - total_distance (float): Total distance to be covered in meters.
    - total_time (float): Total time of travel in seconds.

    Returns:
    - terminal_velocity (float): Terminal velocity needed to satisfy the conditions in m/s.
    """

    # Calculate time to reach terminal velocity
    time_to_terminal_velocity = (initial_velocity - 0) / deceleration_rate

    # Calculate distance covered during deceleration
    distance_during_deceleration = 0.5 * initial_velocity * time_to_terminal_velocity

    # Calculate remaining distance to cover at terminal velocity
    remaining_distance = total_distance - distance_during_deceleration

    # Calculate terminal velocity
    terminal_velocity = remaining_distance / (total_time - time_to_terminal_velocity)

    return terminal_velocity


def compute_terminal_velocity(u, d, t, dt):
    """
    Compute the terminal velocity required for a vehicle to decelerate to cover distance d in time t.

    Parameters:
    - u: Initial velocity (m/s).
    - d: Total distance to cover (m).
    - t: Total time available for deceleration and constant speed (s).
    - dt: Deceleration rate (m/s^2).

    Returns:
    - v: Terminal velocity required for deceleration (m/s).
    """
    # Calculate the time to decelerate to an arbitrary terminal velocity d/t
    t_decel = (d/t - u) / dt

    s_decel = u * t_decel + 0.5 * dt * t_decel**2
    s_remaining = d - s_decel
    v_terminal = s_remaining / (t - t_decel)

    return v_terminal

# Compute terminal velocity when accelerating from rest
def compute_terminal_velocity_acceleration(d, t, dt):
    """
    Compute the terminal velocity required for a vehicle to accelerate to a given final velocity
    while covering distance d in time t.

    Parameters:
    - d: Total distance to cover (m).
    - t: Total time available for acceleration and constant speed (s).
    - dt: Acceleration rate (m/s^2).

    Returns:
    - v_terminal: Terminal velocity required for acceleration (m/s).
    """
    # Calculate distance covered during acceleration
    s_acceleration = 0.5 * dt * t**2

    # Calculate remaining distance for constant speed phase
    s_remaining = d - s_acceleration

    # Calculate terminal velocity for constant speed phase
    v_terminal = s_remaining / (t - 0.5 * t)

    return v_terminal


def generate_path(agent_pos, old_cont_pos, new_cont_pos, tolerance=0.01, steps=10, step_size=0.01):
    
    # If agent is not in old grid_pos, navigate there first
    if not_within_range(
        agent_pos[0], old_cont_pos[0]-tolerance, old_cont_pos[0]+tolerance) or \
        not_within_range(
        agent_pos[1], old_cont_pos[1]-tolerance, old_cont_pos[1]+tolerance):
        generate_path(agent_pos, agent_pos, old_cont_pos)
    
    movements = generate_random_points(old_cont_pos, new_cont_pos, steps, step_size, origin_offset=0.5)

    return movements

'''
def generate_random_points(old_cont_pos, new_cont_pos, steps, step_size, origin_offset=0.5):
    # Determine the unit vector between the two points
    total_distance = math.dist(old_cont_pos, new_cont_pos)
    unit_vector = np.array([(new_cont_pos[0] - old_cont_pos[0]) / total_distance, 
                            (new_cont_pos[1] - old_cont_pos[1]) / total_distance])
    
    # Generate n random points
    points = []
    for _ in range(steps):
        # Calculate the step distance based on the step size
        step_distance = step_size * np.random.uniform()
        
        # Calculate the coordinates of the point at the step distance along the line segment
        point_x = old_cont_pos + unit_vector[0] * step_distance
        point_y = old_cont_pos + unit_vector[1] * step_distance
        
        # Clip the coordinates to ensure they are within the range [-0.5, 0.5]
        point_x = np.clip(point_x, -origin_offset, origin_offset)
        point_y = np.clip(point_y, -origin_offset, origin_offset)
        
        # Add the point to the list of points
        points.append((point_x, point_y))
    
    return points
#'''

def generate_random_points(old_cont_pos, new_cont_pos, steps, step_size, origin_offset=0.5):
    # Generate n random points
    points = []
    for _ in range(steps):
        # Generate random coordinates within the bounding box defined by old_cont_pos and new_cont_pos
        random_x = np.random.uniform(min(old_cont_pos[0], new_cont_pos[0]), max(old_cont_pos[0], new_cont_pos[0]))
        random_y = np.random.uniform(min(old_cont_pos[1], new_cont_pos[1]), max(old_cont_pos[1], new_cont_pos[1]))
        
        # Clip the coordinates to ensure they are within the range [-0.5, 0.5]
        random_x = np.clip(random_x, -origin_offset, origin_offset)
        random_y = np.clip(random_y, -origin_offset, origin_offset)
        
        # Add the point to the list of points
        points.append((random_x, random_y))
    
    return points


def grid_to_continuous_mapping(grid_size=10, continuous_range=1.0):
    mapping = np.zeros((grid_size, grid_size, 2))  # Initialize a numpy array to store mappings
    
    # Calculate the cell size in the continuous environment
    cell_size = continuous_range / grid_size
    
    # Calculate the starting point of the continuous range
    start_continuous_x = -continuous_range / 2.0
    start_continuous_y = continuous_range / 2.0
    
    # Iterate through each grid cell and calculate the continuous world coordinates
    for i in range(grid_size):
        for j in range(grid_size):
            continuous_x = start_continuous_x + (i + 0.5) * cell_size
            continuous_y = start_continuous_y - (j + 0.5) * cell_size
            mapping[i, j] = [continuous_x, continuous_y]
    
    return mapping


def value_in_range(value, lower_bound, upper_bound):
    """
    Check if the given value falls within the specified range.

    Args:
    - value: The value to check.
    - lower_bound: The lower bound of the range.
    - upper_bound: The upper bound of the range.

    Returns:
    - True if the value is within the range, False otherwise.
    """
    return lower_bound <= value <= upper_bound


def not_within_range(value, lower_bound, upper_bound):
    """
    Check if a value is not within the given range.

    Parameters:
    - value: The value to check.
    - lower_bound: The lower bound of the range.
    - upper_bound: The upper bound of the range.

    Returns:
    - True if the value is not within the range, False otherwise.
    """
    return value < lower_bound or value > upper_bound


def index_of_equal_value(t1, t2):
    # Check if the tuples have the same length
    #print("t1: ", t1, "t2: ", t2)
    if len(t1) != len(t2):
        return None  # Return None if the tuples have different lengths

    # Iterate through the tuples and compare elements at corresponding indices
    for i in range(len(t1)):
        if t1[i] == t2[i]:
            return i  # Return the index if the values are equal at the current position
    
    # Return None if no equal values are found
    return None


def radians_to_degrees(radians, z_axis= 1, neg_pi=True):
    """
    Takes in the radians value and a neg_pi value
    We bot environment specifies radians as 0 - pi and encode direction as z-axis
    If neg_pi is true, values are converted from -pi to pi to 0 to 2pi
    """
    radians = radians*z_axis
    radians = radians % (2*math.pi) if neg_pi else radians
    degrees = round(math.degrees(radians))
    degrees = degrees % 360  # Ensure degrees stay within 0-359 range
    if degrees < 0:
        degrees += 360  # Add 360 to negative degrees to bring them into positive range
    
    # Force values of 360 to 0
    if degrees == 360: degrees = 0
    return degrees


def angle_to_positive(n):
    """
    Transform an angle value to a positive value
    -10 becomes 350, -5 becomes 355 etc
    Positive  values are unchanged

    Args: n - input angle
    Returns: Positive angle values between 0 and 359
    """
    return (360 + (n % 360)) % 360


class pid_velocity_fixed_height_controller():
    def __init__(self):
        self.past_vx_error = 0.0
        self.past_vy_error = 0.0
        self.past_alt_error = 0.0
        self.past_pitch_error = 0.0
        self.past_roll_error = 0.0
        self.altitude_integrator = 0.0
        self.last_time = 0.0

    def pid(self, dt, desired_vx, desired_vy, desired_yaw_rate, desired_altitude, actual_roll, actual_pitch, actual_yaw_rate,
            actual_altitude, actual_vx, actual_vy):
        # Velocity PID control (converted from Crazyflie c code)
        gains = {"kp_att_y": 1, "kd_att_y": 0.5, "kp_att_rp": 0.5, "kd_att_rp": 0.1,
                 "kp_vel_xy": 2, "kd_vel_xy": 0.5, "kp_z": 10, "ki_z": 5, "kd_z": 5}

        # Velocity PID control
        vx_error = desired_vx - actual_vx
        vx_deriv = (vx_error - self.past_vx_error) / dt
        vy_error = desired_vy - actual_vy
        vy_deriv = (vy_error - self.past_vy_error) / dt
        desired_pitch = gains["kp_vel_xy"] * np.clip(vx_error, -1, 1) + gains["kd_vel_xy"] * vx_deriv
        desired_roll = -gains["kp_vel_xy"] * np.clip(vy_error, -1, 1) - gains["kd_vel_xy"] * vy_deriv
        self.past_vx_error = vx_error
        self.past_vy_error = vy_error

        # Altitude PID control
        alt_error = desired_altitude - actual_altitude
        alt_deriv = (alt_error - self.past_alt_error) / dt
        self.altitude_integrator += alt_error * dt
        alt_command = gains["kp_z"] * alt_error + gains["kd_z"] * alt_deriv + \
            gains["ki_z"] * np.clip(self.altitude_integrator, -2, 2) + 48
        self.past_alt_error = alt_error

        # Attitude PID control
        pitch_error = desired_pitch - actual_pitch
        pitch_deriv = (pitch_error - self.past_pitch_error) / dt
        roll_error = desired_roll - actual_roll
        roll_deriv = (roll_error - self.past_roll_error) / dt
        yaw_rate_error = desired_yaw_rate - actual_yaw_rate
        roll_command = gains["kp_att_rp"] * np.clip(roll_error, -1, 1) + gains["kd_att_rp"] * roll_deriv
        pitch_command = -gains["kp_att_rp"] * np.clip(pitch_error, -1, 1) - gains["kd_att_rp"] * pitch_deriv
        yaw_command = gains["kp_att_y"] * np.clip(yaw_rate_error, -1, 1)
        self.past_pitch_error = pitch_error
        self.past_roll_error = roll_error

        # Motor mixing
        m1 = alt_command - roll_command + pitch_command + yaw_command
        m2 = alt_command - roll_command - pitch_command - yaw_command
        m3 = alt_command + roll_command - pitch_command + yaw_command
        m4 = alt_command + roll_command + pitch_command - yaw_command

        # Limit the motor command
        m1 = np.clip(m1, 0, 600)
        m2 = np.clip(m2, 0, 600)
        m3 = np.clip(m3, 0, 600)
        m4 = np.clip(m4, 0, 600)

        return [m1, m2, m3, m4]
    

def pickle_file(fname, file, base_path="", verbose=False):

    import pickle, os
    if base_path != "":
        if os.path.isdir(base_path):
            pass
        else:
            os.mkdir(base_path)

    save_name= str(fname) #+ '-pickle'
    save_path= os.path.join(base_path, save_name)
    with open(save_path, 'wb') as fp:
                pickle.dump(file, fp)
    if verbose: print(f'file {fname} pickled and saved at {save_path}')



def unpickle_file(fname, file = None, verbose=False):
    
    import pickle, os 
    if os.path.isfile(fname):
        with open(fname, 'rb') as fp:
            file= pickle.load(fp) 
        if verbose: print("File loaded")         
        return file
    
    else:
        print('File not found or something else')
    return None


# Recusively update a dict from another dict
def recursive_update(dict1, dict2):
    for key, value in dict2.items():
        if isinstance(value, dict):
            # Recursively update nested dictionaries
            dict1[key] = recursive_update(dict1.get(key, {}), value)
        else:
            # Update non-dictionary values
            dict1[key] = value
    return dict1