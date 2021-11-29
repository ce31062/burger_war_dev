#!python3
#-*- coding: utf-8 -*-

import torch

print("wallAvoid.pyを実行する")

def punish_by_count(lidar, dist_th=0.2, count_th=90):
    # Check LiDAR data to punish for AMCL failure
    punish = 0
    if lidar is not None:
        lidar_1d = lidar.squeeze()
        count_too_close = 0
            
        # Check each laser and count up if too close
        for intensity in lidar_1d:
            if intensity.item() < dist_th:
                count_too_close += 1

        # Punish if too many lasers close to obstacle
        if count_too_close > count_th:
            print("### Too close to the wall, get penalty1 ###")
            punish = -0.5

    return punish


def punish_by_min_dist(lidar, dist_th=0.15):
    # Check LiDAR data to punish for AMCL failure
    punish = 0
    if lidar is not None:
        lidar_1d = lidar.squeeze()
        if lidar_1d.min() < dist_th:
            print("### Too close to the wall, get penalty2 ###")
            punish = -0.5

    return punish


def manual_avoid_wall(lidar, dist_th=0.2, count_th=90):
    # check how stacked for each direction
    front_stacked = torch.sum(lidar[0][0][:45] < dist_th) \
                + torch.sum(lidar[0][0][315:] < dist_th)
    left_stacked = torch.sum(lidar[0][0][45:135] < dist_th)
    rear_stacked = torch.sum(lidar[0][0][135:225] < dist_th)
    right_stacked = torch.sum(lidar[0][0][:315] < dist_th)
    # if total of stacked is larger than threshold, recover stacked status
    if front_stacked + left_stacked + rear_stacked + right_stacked > dist_th:
        print("### stacked ###")
        avoid = True
        SPEED = 0.4
        RAD = 3.14
        # decide where to go for recovering stacked status
        _, linear_x, angular_z = min([
            (front_stacked, SPEED, .0),
            (left_stacked, .0, RAD / 2), 
            (rear_stacked, -SPEED, .0),
            (right_stacked, .0, -RAD / 2),
        ], key=lambda e: e[0])
            
    else:
	print("safety_for_wall")
        avoid = False
        linear_x = None
        angular_z = None

    return avoid, linear_x, angular_z


def manual_avoid_wall_2(lidar, dist_th=0.2, back_vel=0.3):
    lidar = lidar.squeeze()
    lidar_min = torch.min(lidar).item()
    if lidar_min < dist_th:
        min_index = torch.argmin(lidar).item()
        print("### avoid wall - {} deg ###".format(min_index))
        if min_index < 90 - 10:
            rotate_deg = min_index - 90
            linear_x = -back_vel
        elif min_index < 180:
            rotate_deg = min_index - 180
            linear_x = back_vel
        elif min_index < 270 + 10:
            rotate_deg = min_index - 180
            linear_x = back_vel
        else:
            rotate_deg = min_index - 270
            linear_x = -back_vel

        avoid = True
        angular_z = rotate_deg*3.1416/180
    else:
        avoid = False
        linear_x = None
        angular_z = None

    return avoid, linear_x, angular_z
