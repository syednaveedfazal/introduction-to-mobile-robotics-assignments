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
    return np.log(p / (1 - p))
    
def logodds2prob(l):
    return 1 / (1 + np.exp(-l))    
    
def inv_sensor_model(cell, endpoint, prob_occ, prob_free):
    
    if cell == endpoint:
    # We are checking if the cell is the endpoint ie. the lidar is hitting an obstacle
        return prob2logodds(prob_occ)
    # Or else, the cell is free space
    else:
        return prob2logodds(prob_free)

def grid_mapping_with_known_poses(poses_raw, ranges_raw, map_res, occ_gridmap, prior, prob_free, prob_occ):
    

    log_odds_map = prob2logodds(occ_gridmap)
    # Convert the initial probability map to a log-odds map.
    
    # Get the log-odds values for our sensor model.
    l_occ = prob2logodds(prob_occ)   # Log-odds for "occupied"
    l_free = prob2logodds(prob_free)  # Log-odds for "free"
    l_prior = prob2logodds(prior)   # Log-odds for "prior"
    

    for t in range(len(poses_raw)):
        # Iterate over each time step
        w_pose = poses_raw[t]    # Current world pose (x, y, theta)
        r_ranges = ranges_raw[t] # Current 1D array of range measurements
        # Get the *current* pose and scan for this time step
        
        
        robot_cell = poses2cells(w_pose, occ_gridmap, map_res)
        # The robot's position is in coordinate system. We need to convert it to grid indices 
        
        
        endpoint_cells = ranges2cells(r_ranges, w_pose, occ_gridmap, map_res)
        # endpoint_cells is a matrix containing the grid coordinates for every single laser hit in this scan
        # Each column of endpoint_cells is one (row, col) pair for a laser hit.
    
    
        for i in range(endpoint_cells.shape[1]):
            # we iterate through them one beam at a time
            
            endpoint = endpoint_cells[:, i]
            # Get the (row, col) of the ith endpoint

            beam_cells = bresenham(int(robot_cell[0]), int(robot_cell[1]), 
                                    int(endpoint[0]), int(endpoint[1]))
            # we need to know every single grid cell the laser passed through to get there
            # Get all grid cells along the line from the robot to the endpoint
            
            if len(beam_cells) > 1:
                # make sure the beam actually traveled somewhere. If the robot is inside a wall (distance 0), we skip this

                free_cells = beam_cells[:-1] 
                # The last item is the obstacle. Every cell before the obstacle must be empty (air) because the laser passed through it

                for cell in free_cells:
                    # Update rule: L_t = L_{t-1} + L_inv - L_0
                    log_odds_map[cell[0], cell[1]] += l_free - l_prior
            

            if len(beam_cells) > 0:
                occ_cell = beam_cells[-1] 
                # This implies there is a solid object here. The laser hit something at the last cell.

                log_odds_map[occ_cell[0], occ_cell[1]] += l_occ - l_prior


    final_prob_map = logodds2prob(log_odds_map)
    # Convert the final log-odds map back to a probability map for plotting

    return final_prob_map
