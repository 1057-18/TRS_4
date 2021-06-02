#!/usr/bin/env python

import rospy
import numpy as np
from scipy.linalg import block_diag
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from tf.transformations import euler_from_quaternion
from laser_line_extraction.msg import LineSegment, LineSegmentList

prev_position = [0.0, 0.0]
pose = np.array([0, 0, 0]).reshape(-1, )
covar_p = np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])
wheel_radius = 0.033
wheel_track = 0.16
kr = 0.05
kl = 0.05
line_map = None
covar_map = None
validation_gate = 0.2

def create_map():
    global line_map
    global covar_map
    rospy.init_node('localization')
    observation = rospy.wait_for_message('line_segments', LineSegmentList).line_segments
    line_map, covar_map = construct_map_matrices(observation)

def construct_map_matrices(obser):
    angles = []
    radii = []
    covars = []
    for i in range(len(obser)):
        angles.append(obser[i].angle)
        radii.append(obser[i].radius)
        covars.append(np.array(obser[i].covariance).reshape(2, 2))
    return np.vstack((np.array(angles), np.array(radii))), np.array(covars)

def calculate_Fp(delta_s, delta_theta, theta):
    first_row = [1, 0, -delta_s*np.sin(theta + delta_theta/2)]
    second_row = [0, 1, delta_s*np.cos(theta * delta_theta/2)]
    thrid_row = [0, 0, 1]
    return np.array([first_row, second_row, thrid_row])

def calculate_F_delta(delta_s, delta_theta, theta, wheel_track):
    first_row = [(1/2)*np.cos(theta + delta_theta/2) - (delta_s/(2*wheel_track))*np.sin(theta + delta_theta/2), (1/2)*np.cos(theta + delta_theta/2) + (delta_s/(2*wheel_track))*np.sin(theta + delta_theta/2)]
    second_row = [(1/2)*np.sin(theta + delta_theta/2) + (delta_s/(2*wheel_track))*np.cos(theta + delta_theta/2), (1/2)*np.sin(theta + delta_theta/2) - (delta_s/(2*wheel_track))*np.cos(theta + delta_theta/2)]
    thrid_row = [1/wheel_track, -1/wheel_track]
    return np.array([first_row, second_row, thrid_row])

def calculate_Jacobian(line_map):
    H = np.full((line_map.shape[1], 2, 3), np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]))
    H[:, 1, 0] = -np.cos(line_map[0])
    H[:, 1, 1] = -np.sin(line_map[0])
    return H

def calculate_innovation_covariance(H, R):
    covar_innov = []
    for i in range(R.shape[0]):
        for j in range(H.shape[0]):
            covar_innov.append(np.dot(np.dot(H[j, :, :], covar_p), H[j, :, :].T) + R[i, :, :])
    covar_innov = np.array(covar_innov).reshape(R.shape[0], H.shape[0], 2, 2)
    return covar_innov

def transition_function(data):
    global wheel_radius
    global pose
    global covar_p
    global prev_position
    global wheel_track
    global kr
    global kl

    x = pose[0]
    y = pose[1]
    theta = pose[2]

    r_wheel = data.position[0]
    l_wheel = data.position[1]

    delta_sr = (r_wheel - prev_position[0]) * wheel_radius
    delta_sl = (l_wheel - prev_position[1]) * wheel_radius

    prev_position = data.position
 
    delta_s = (delta_sr + delta_sl)/2
    delta_theta = (delta_sr - delta_sl)/wheel_track

    delta_x = delta_s * np.cos(theta + delta_theta/2)
    delta_y = delta_s * np.sin(theta + delta_theta/2)

    pose = pose + np.array([delta_x, delta_y, delta_theta]).reshape(-1, )

    covar_delta = np.array([[kr*abs(delta_sr), 0], [0, kl*abs(delta_sl)]])
    Fp = calculate_Fp(delta_s, delta_theta, theta)
    F_delta = calculate_F_delta(delta_s, delta_theta, theta, wheel_track)
    covar_p = np.dot(np.dot(Fp, covar_p), Fp.T) + np.dot(np.dot(F_delta, covar_delta), F_delta.T)

def measurement_function():
    global pose 
    global line_map
    [x_hat, y_hat, theta_hat] = pose
    z_hat = np.array([line_map[0] - theta_hat, line_map[1] - (x_hat*np.cos(line_map[0]) + y_hat*np.sin(line_map[0]))]).reshape(2, -1)
    H = calculate_Jacobian(line_map)
    return z_hat, H
    
def associate_measurements(z_hat, H):
    global covar_p
    global validation_gate
    rospy.init_node('localization')
    observation = rospy.wait_for_message('line_segments', LineSegmentList).line_segments
    z, R = construct_map_matrices(observation)

    v = {}
    for i in range(z.shape[1]):
        for j in range(z_hat.shape[1]):
            v[(i, j)] = z[:, i] - z_hat[:, j]
    # v = np.array(v).reshape(z.shape[1], z_hat.shape[1], 2, 1)

    covar_innov = calculate_innovation_covariance(H, R)

    d = []
    pairs = []
    for i in range(covar_innov.shape[0]):
        for j in range(covar_innov.shape[1]):
            pairs.append((i, j))
            d.append(np.dot(np.dot(v[(i, j)].T, np.linalg.pinv(covar_innov[i, j, :, :])), v[(i, j)]))
    d = np.array(d).reshape(-1, )
    pairs = np.array(pairs, dtype="i, i")

    log_vec = d <= validation_gate**2
    valid_pairs = pairs[log_vec]

    i_list=[]
    j_list=[]

    for k in range(len(valid_pairs)):
        (i, j) = valid_pairs[k]
        i_list.append(i)
        j_list.append(j)

    H = H[j_list, :, :]
    R = R[i_list, :, :]

    return v, valid_pairs, H, R 

def filter_step(v, valid_pairs, H, R):
    global pose
    global covar_p
    H = np.transpose(H, [0, 2, 1]).reshape(-1, 3)
    R = block_diag(*R)
    covar_innov = np.dot(H, covar_p).dot(H.T) + R
    K = np.dot(covar_p, H.T).dot(np.linalg.pinv(covar_innov))
    v_hat = []
    for k in range(len(valid_pairs)):
        (i, j) = valid_pairs[k]
        v_hat.append(v[(i, j)])
    v_hat = np.array(v_hat)

    pose = pose + np.dot(K, v_hat)
    covar_p = covar_p - np.dot(K, covar_innov).dot(K.T)

    return pose, covar_p


def callback(data):
    transition_function(data)
    z_hat, H = measurement_function()
    v, valid_pairs, H_hat, R_hat = associate_measurements(z_hat, H)
    x, P = filter_step(v, valid_pairs, H_hat, R_hat)
    print(x)

if __name__ == '__main__':
    try:
        create_map()
        rospy.init_node('localization')
        sub = rospy.Subscriber('joint_states', JointState, callback)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass