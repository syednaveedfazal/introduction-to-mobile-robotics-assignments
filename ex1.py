#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_belief(belief):
    
    plt.figure()
    
    ax = plt.subplot(2,1,1)
    ax.matshow(belief.reshape(1, belief.shape[0]))
    ax.set_xticks(np.arange(0, belief.shape[0],1))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])
    ax.title.set_text("Grid")
    
    ax = plt.subplot(2, 1, 2)
    ax.bar(np.arange(0, belief.shape[0]), belief)
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.set_ylim([0, 1.05])
    ax.title.set_text("Histogram")


def motion_model(action, belief):
    """
    Updates the belief distribution based on a single motion command.
    
    Args:
        belief (np.array): The current belief distribution.
        command (str): The motion command, 'F' for forward or 'B' for backward.
        
    Returns:
        np.array: The updated belief distribution.
    """
    num_cells = len(belief)
    new_belief = np.zeros(num_cells)
    # we create the an empty array for the new belief to update our motion model
    
    # Define motion probabilities
    prob_correct = 0.75
    prob_stay = 0.15
    prob_opposite = 0.10
    
    # Iterate through each possible starting cell 'i'
    for i in range(num_cells):
        prior_prob = belief[i]
        # We use previous belief to get the prior probability of being in cell 'i'

        if prior_prob == 0:
            continue  # No need to calculate if there's no chance the robot was here
            # we skip the for loop if the prior probability is zero
            
        # Calculate where the probability from cell 'i' moves to
        if action == 'F':
            # Correct move (Forward)
            if i + 1 < num_cells:
                # If not hitting the border
                new_belief[i+1] += prior_prob * prob_correct
            else: # when hitting the right border
                new_belief[i] += prior_prob * prob_correct
                
            # Opposite move (Backward)
            if i - 1 >= 0:
                new_belief[i-1] += prior_prob * prob_opposite
            else: # when hitting the left border
                new_belief[i] += prior_prob * prob_opposite
                
        elif action == 'B':
            # Correct move (Backward)
            if i - 1 >= 0: 
                new_belief[i-1] += prior_prob * prob_correct
            else:  # when hitting the left border
                new_belief[i] += prior_prob * prob_correct
            
            # Opposite move (Forward)
            if i + 1 < num_cells:
                new_belief[i+1] += prior_prob * prob_opposite
            else: # when hitting the right border
                new_belief[i] += prior_prob * prob_opposite

        # No move
        # we always add the probability of staying in the same cell
        new_belief[i] += prior_prob * prob_stay
    return new_belief
    
def sensor_model(observation, belief, world):
    """
    Updates a belief distribution based on a sensor observation.
    
    Args:
        observation (int): The sensor reading (0 for white, 1 for blue).
        belief (np.array): The current belief distribution (after a motion update).
        world (np.array): The map of the world's colors.
        
    Returns:
        np.array: The updated, normalized belief distribution.
    """
    # Sensor characteristics
    prob_see_blue_if_blue = 0.85  # True Positive
    prob_see_white_if_white = 0.75 # True Negative
    
    prob_see_blue_if_white = 1 - prob_see_white_if_white # False Positive
    prob_see_white_if_blue = 1 - prob_see_blue_if_blue # False Negative

    new_belief = np.zeros_like(belief)

    # Loop through each cell in the world
    for i in range(len(world)):
        is_blue = world[i] == 1
        # we check if sensor reading matches the cell color
        
        # Determine the likelihood of the observation given the cell's true color
        if observation == 1: # Observed blue
            likelihood = prob_see_blue_if_blue if is_blue else prob_see_blue_if_white
        else: # Observed white
            likelihood = prob_see_white_if_blue if is_blue else prob_see_white_if_white
        
        # Update the belief for this cell
        new_belief[i] = belief[i] * likelihood
        
    # Normalize the belief so it sums to 1
    total_prob = np.sum(new_belief)
    if total_prob > 0:
        new_belief = new_belief / total_prob
        
    return new_belief


def recursive_bayes_filter(commands, observations, initial_belief, world):
    """Estimates robot position using a recursive Bayes filter.

    Args:
        commands (list): Sequence of movement commands ('F' or 'B').
        observations (np.array): Sequence of sensor readings (0 or 1).
        initial_belief (np.array): Initial belief distribution.
        world (np.array): The world's color map.

    Returns:
        np.array: The final belief distribution after all steps.
    """

    # It prepares the inputs for the tool.
    # There are 10 observations for 9 commands, so we take first oberversation as initial observation
    initial_observation = observations[0]
    
    # The MANAGER calls the tool for the first step.
    belief = sensor_model(initial_observation, initial_belief, world)
    
    for i in range(len(commands)):
        # it prepares the inputs for the motion model
        predicted_belief = motion_model(commands[i], belief)
        
        # it prepares the inputs for the sensor model
        current_observation = observations[i + 1]
        
        # and it calls the sensor model tool again.
        belief = sensor_model(current_observation, predicted_belief, world)
            
    return belief





def load_data(filename):
    """
    Loads a single line of comma-separated integers from a file.
    """
    with open(filename, 'r') as f:
        # Read the first (and only) line from the file
        line = f.readline()
        
        # Remove any extra whitespace from the ends
        line = line.strip()
        
        # Split the string into a list of smaller strings (e.g., ['0', '1', '0', ...])
        string_values = line.split(',')
        
        # Convert each string value into an integer
        int_values = [int(v) for v in string_values]
        
    return int_values