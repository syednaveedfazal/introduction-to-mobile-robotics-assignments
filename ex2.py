#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import bresenham as bh

def plot_gridmap(gridmap):
    plt.figure()
    plt.imshow(gridmap, cmap='Greys',vmin=0, vmax=1)
    
def init_gridmap(size, res):
    gridmap = np.zeros([int(np.ceil(size/res)), int(np.ceil(size/res))])
    return gridmap

def world2map(pose, gridmap, map_res):
    origin = np.array(gridmap.shape)/2
    new_pose = np.zeros_like(pose)
    new_pose[0] = np.round(pose[0]/map_res) + origin[0];
    new_pose[1] = np.round(pose[1]/map_res) + origin[1];
    return new_pose.astype(int)

def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr    

def ranges2points(ranges):
    # laser properties
    start_angle = -1.5708
    angular_res = 0.0087270
    max_range = 30
    # rays within range
    num_beams = ranges.shape[0]
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    angles = np.linspace(start_angle, start_angle + (num_beams*angular_res), num_beams)[idx]
    points = np.array([np.multiply(ranges[idx], np.cos(angles)), np.multiply(ranges[idx], np.sin(angles))])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, points.shape[1])), axis=0)
    return points_hom

def ranges2cells(r_ranges, w_pose, gridmap, map_res):
    # ranges to points
    r_points = ranges2points(r_ranges)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # covert to map frame
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2,:]
    return m_points

def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose  

def bresenham(x0, y0, x1, y1):
    l = np.array(list(bh.bresenham(x0, y0, x1, y1)))
    return l
    
def prob2logodds(p):
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1 - epsilon)
    return np.log(p / (1 - p))

def logodds2prob(l):
    return 1 - 1 / (1 + np.exp(l))    
    
def inv_sensor_model(cell, endpoint, prob_occ, prob_free):
    """
    Inverse sensor model:
    Returns log-odds update depending on whether the cell is free or occupied.
    """
    if np.array_equal(cell, endpoint):
        return prob2logodds(prob_occ)
    else:
        return prob2logodds(prob_free)
    

def grid_mapping_with_known_poses(poses_raw, ranges_raw, map_res, occ_gridmap, prior, prob_free, prob_occ):
    """
    Performs occupancy grid mapping with known robot poses and range data.
    """
    # Convert prior probability to log-odds
    log_odds_map = prob2logodds(occ_gridmap)

    # Loop through all poses and corresponding range measurements
    for i in range(len(poses_raw)):
        w_pose = poses_raw[i, :]
        r_ranges = ranges_raw[i, :]

        # Convert pose and ranges to map coordinates
        m_robot = poses2cells(w_pose, occ_gridmap, map_res)
        m_points = ranges2cells(r_ranges, w_pose, occ_gridmap, map_res)

        for j in range(m_points.shape[1]):
            endpoint = m_points[:, j]
            # Get all cells along the beam
            line = bresenham(m_robot[0], m_robot[1], endpoint[0], endpoint[1])

            # Update cells along beam
            for k in range(len(line) - 1):
                cell = line[k]
                log_odds_map[cell[1], cell[0]] += inv_sensor_model(cell, endpoint, prob_occ, prob_free) - prob2logodds(prior)
            
            # Update endpoint as occupied
            end_cell = line[-1]
            log_odds_map[end_cell[1], end_cell[0]] += inv_sensor_model(end_cell, endpoint, prob_occ, prob_free) - prob2logodds(prior)

    # Convert back to probabilities
    occ_gridmap = logodds2prob(log_odds_map)
    return occ_gridmap

