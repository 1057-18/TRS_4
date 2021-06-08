#!/usr/bin/env python

import rospy
import numpy as np
from scipy.linalg import block_diag
from sensor_msgs.msg import JointState
from laser_line_extraction.msg import LineSegment, LineSegmentList
import os

prev_position = [0.0, 0.0]
pose = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)
covar_p = np.array([[0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, 0.001]])
wheel_radius = 0.033
wheel_track = 0.16
kr = 0.05
kl = 0.05
line_map = None
covar_map = None
validation_gate = 1

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
    return np.array([[1.0, 0.0, -delta_s*np.sin(theta + delta_theta/2)], [0.0, 1.0, delta_s*np.cos(theta + delta_theta/2)], [0.0, 0.0, 1.0]], dtype=float)

def calculate_F_delta(delta_s, delta_theta, theta, wheel_track):
    first_row = [(1 / 2) * np.cos(theta + delta_theta / 2) + (delta_s / (2 * wheel_track)) * np.sin(theta + delta_theta / 2), (1 / 2) * np.cos(theta + delta_theta / 2) - (delta_s / (2 * wheel_track)) * np.sin(theta + delta_theta / 2)]
    second_row = [(1 / 2) * np.sin(theta + delta_theta / 2) - (delta_s / (2 * wheel_track)) * np.cos(theta + delta_theta / 2), (1 / 2) * np.sin(theta + delta_theta / 2) + (delta_s / (2 * wheel_track)) * np.cos(theta + delta_theta / 2)]
    thrid_row = [-1 / wheel_track, 1 / wheel_track]
    return np.array([first_row, second_row, thrid_row], dtype=float)

def calculate_Jacobian(line_map):
    H = np.full((line_map.shape[1], 2, 3), np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]))
    H[:, 1, 0] = -np.cos(line_map[0])
    H[:, 1, 1] = -np.sin(line_map[0])
    return H

def normalizeLineParameters(alpha, r):
    if r < 0:
        alpha = alpha + np.pi
        r = -r
        isRNegated = 1
    else:
        isRNegated = 0

    if alpha > np.pi:
        alpha = alpha - 2 * np.pi
    elif alpha < -np.pi:
        alpha = alpha + 2 * np.pi

    return np.array([alpha, r]), isRNegated

def transition_function(data):
    global wheel_radius
    global pose
    global covar_p
    global prev_position
    global wheel_track
    global kr
    global kl

    [x, y, theta] = pose

    [r_wheel, l_wheel] = data.position

    delta_sr = (r_wheel - prev_position[0]) * wheel_radius
    delta_sl = (l_wheel - prev_position[1]) * wheel_radius

    prev_position = data.position
 
    delta_s = (delta_sr + delta_sl)/2
    delta_theta = (delta_sr - delta_sl)/wheel_track
    delta_x = delta_s * np.cos(theta[0] + delta_theta/2)
    delta_y = delta_s * np.sin(theta[0] + delta_theta/2)

    pose = pose + np.array([delta_x, delta_y, delta_theta]).reshape(-1, 1)

    covar_delta = np.array([[kr*abs(delta_sr), 0], [0, kl*abs(delta_sl)]])
    Fp = calculate_Fp(delta_s, delta_theta, theta)
    F_delta = calculate_F_delta(delta_s, delta_theta, theta, wheel_track)
    covar_p = np.dot(Fp, covar_p).dot(Fp.T) + np.dot(F_delta, covar_delta).dot(F_delta.T)

def measurement_function():
    global pose 
    global line_map
    [x_hat, y_hat, theta_hat] = pose
    z_hat = np.array([line_map[0] - theta_hat, line_map[1] - (x_hat*np.cos(line_map[0]) + y_hat*np.sin(line_map[0]))]).reshape(2, -1)
    H = calculate_Jacobian(line_map)

    for i in range(z_hat.shape[1]):
        pair, isNeg = normalizeLineParameters(z_hat[0, i], z_hat[1, i])
        z_hat[:, i] = pair
        if isNeg:
            H[i, 1, :] = -H[i, 1, :]

    return z_hat, H
    
def associate_measurements(z_hat, H):
    global covar_p
    global validation_gate
    rospy.init_node('localization')
    observation = rospy.wait_for_message('line_segments', LineSegmentList).line_segments
    z, R = construct_map_matrices(observation)

    m = z.shape[1]
    n = z_hat.shape[1]

    v = np.zeros((m, n, 2))
    d = np.zeros((m, n))
    covar_innov = np.zeros((m, n, 2, 2))
    for i in range(m):
        for j in range(n):
            v[i, j, :] = z[:, i] - z_hat[:, j]
            covar_innov[i, j, :, :] = np.dot(H[j, :, :], covar_p).dot(H[j, :, :].T) + R[i, :, :]
            d[i, j] = np.dot(v[i, j, :].T, np.linalg.inv(covar_innov[i, j, :, :])).dot(v[i, j, :])

    log_vec = d <= validation_gate**2

    v = v[log_vec, :]
    H = H[np.sum(log_vec, axis=0) > 0, :, :]
    R = R[np.sum(log_vec, axis=1) > 0, :, :]

    return v, H, R

def filter_step(v, H, R):
    global pose
    global covar_p

    v = v.reshape(-1, 1)
    H = np.transpose(H, [1, 0, 2]).reshape(-1, 3)
    R = block_diag(*R)
    covar_innov = np.dot(H, covar_p).dot(H.T) + R
    K = np.dot(covar_p, H.T).dot(np.linalg.pinv(covar_innov))

    pose = pose + np.dot(K, v).reshape(-1, 1)
    covar_p = covar_p - np.dot(K, covar_innov).dot(K.T)

    return pose, covar_p

def callback(data):
    global pose
    transition_function(data)
    z_hat, H = measurement_function()
    v, H_hat, R_hat = associate_measurements(z_hat, H)
    x, P = filter_step(v, H_hat, R_hat)
    os.system('cls' if os.name == 'nt' else 'clear')
    print(pose)

if __name__ == '__main__':
    try:
        create_map()
        rospy.init_node('localization')
        sub = rospy.Subscriber('joint_states', JointState, callback)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass