#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def dist(x, y):
    return np.sqrt(((x[0]-y[0])**2)+((x[1]-y[1])**2))

def normpdf(x, mu, sigma):
    u = (x-mu)/abs(sigma)
    y = (1/(np.sqrt(2*np.pi)*abs(sigma)))*np.exp(-u*u/2)
    return y

def landmark_observation_model(z,sigma_r, b, x):
    ##STUDENT_CODE:  #TODO

    b = np.array(b)
    x = np.array(x)
    expected_dist = np.linalg.norm(b - x)
    
    # 2. Calculate the Gaussian Likelihood
    # Formula: (1 / (sigma * sqrt(2*pi))) * exp( - (diff)^2 / (2*sigma^2) )
    
    # Normalization constant (The height of the peak)
    norm_const = 1.0 / (np.sqrt(2 * np.pi) * sigma_r)
    
    # Exponent part
    error = z - expected_dist
    exponent = - (error ** 2) / (2 * sigma_r ** 2)
    
    prob = norm_const * np.exp(exponent)
    
    return prob

    ##END_STUDENT_CODE:

def observation_likelihood(z,sigma_z, b, gridmap):
    ##STUDENT_CODE:  #TODO    """

    # Get the dimensions of the map
    size_x, size_y = gridmap.shape
    
    # Initialize the output likelihood grid
    likelihood_map = np.zeros((size_x, size_y))
    
    # Iterate over every cell in the environment
    for x in range(size_x):
        for y in range(size_y):
            # Current cell is the potential robot position
            robot_pos = np.array([x, y])
            
            # Use the model from 4.1 to calculate probability
            prob = landmark_observation_model(z, sigma_z, b, robot_pos)
            
            # Store the probability in the likelihood map
            likelihood_map[x, y] = prob
            
    return likelihood_map


def joint_observation_likelihood(z, sigma_z, b, gridmap):
    ##STUDENT_CODE:  #TODO    



    ##END_STUDENT_CODE: 
    joint_prob = np.ones_like(gridmap, dtype=float)
    
    # 2. Iterate through each measurement provided
    num_observations = len(z)
    
    for i in range(num_observations):
        zi = z[i]
        sigma = sigma_z[i]
        b_pos = b[i]
        
        # 3. Compute likelihood for this specific sensor reading
        p_z = observation_likelihood(zi, sigma, b_pos, gridmap)
        
        # 4. Multiply into the joint belief (Intersection of probabilities)
        joint_prob *= p_z
        
    return joint_prob



def plot_setup(num_obs=1):
    plt.close()
    fig, axes = plt.subplots(1, num_obs, figsize=(10, 5))
    if num_obs == 1:
        ytitle = 1.05
        ysubtitle = 0.95
    else:
        ytitle = 0.88
        ysubtitle = 0.8
    fig.suptitle("Likelihood Maps of Beacon Observations", fontsize=16, fontweight='bold',y=ytitle)
    fig.text(0.5, ysubtitle, "Solid line = observed distance, dotted lines = ±2σ uncertainty",
            ha='center', fontsize=10)

    return axes

def plot_helper(likelihood,colors,z_all,sigma_z_all, beacon_all, num_obs,ax,n_sigma = 2):

    #If single observation not passed as array
    if isinstance(colors,str):
        colors = [colors]
        z_all = np.array([z_all])
        sigma_z_all = np.array([sigma_z_all])
        beacon_all = np.array([beacon_all])
    else:
        colors = np.tile(colors, int(np.ceil(num_obs/len(colors))))[:num_obs] #Ensures enough colors are defined - wraps around color list

    ax.scatter(beacon_all[:,0],beacon_all[:,1], s=100,c=colors)

    for j in range(num_obs):

        obs_color = colors[j]
        circle = Circle((beacon_all[j,0], beacon_all[j,1]), radius=z_all[j], edgecolor=obs_color, facecolor='none', lw=2)
        ax.add_patch(circle)

        # ±2σ (n_sigma) uncertainty
        inner_radius = max(z_all[j] - n_sigma*sigma_z_all[j],0)
        outer_radius = z_all[j] + n_sigma*sigma_z_all[j]
        
        uncertainty_circle = Circle((beacon_all[j,0], beacon_all[j,1]), radius=inner_radius,
                                    edgecolor=obs_color, facecolor='none',ls='--',lw=0.5)
        ax.add_patch(uncertainty_circle)
        uncertainty_circle = Circle((beacon_all[j,0], beacon_all[j,1]), radius=outer_radius,
                                    edgecolor=obs_color, facecolor='none',ls='--',lw=0.5)
        ax.add_patch(uncertainty_circle)

    #Needs to be transposed to match other plots
    ax.matshow(np.transpose(likelihood), origin="lower", cmap='gray')

