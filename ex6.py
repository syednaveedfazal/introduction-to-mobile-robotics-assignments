#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import PillowWriter
from IPython.display import Image, display, clear_output
import matplotlib
import numpy as np
import os
import pickle
import timeit
import inspect

POINTSIZE_TRESHHOLDS = [(50, 10), (500, 2), (5000, 0.5)]
TODO = None


def sample_motion_model_odometry(pose_t_1, u_t, alpha):
    rot1, trans, rot2 = u_t

    ##STUDENT_CODE #TODO Q1: Compute x_t, y_t and theta_t



    # For this task (compared to EX03) you have been already given the rot1, trans and rot2 components.

    # Please follow the notation of the lecture -> The variance is a sum of absolute values not squares!


    ##STUDENT_CODE
    x, y, theta = pose_t_1
    a1, a2, a3, a4 = alpha

    # 1. Calculate the variance (noise magnitude) for each step.
    # The noise is proportional to how much the robot moved.
    # Note: The prompt specifies variance is the sum of absolute values.
    sigma2_rot1  = a1 * np.abs(rot1)  + a2 * np.abs(trans)
    sigma2_trans = a3 * np.abs(trans) + a4 * (np.abs(rot1) + np.abs(rot2))
    sigma2_rot2  = a1 * np.abs(rot2)  + a2 * np.abs(trans)

    # 2. Sample the "noisy" control parameters.
    # We add random Gaussian noise to the reported odometry.
    # np.random.normal takes (mean, std_dev), so we assume std_dev = sqrt(variance)
    hat_rot1  = rot1  + np.random.normal(0, np.sqrt(sigma2_rot1))
    hat_trans = trans + np.random.normal(0, np.sqrt(sigma2_trans))
    hat_rot2  = rot2  + np.random.normal(0, np.sqrt(sigma2_rot2))

    # 3. Apply the geometric update to the pose.
    # Important: The translation happens in the direction of (theta + hat_rot1)
    x_t = x + hat_trans * np.cos(theta + hat_rot1)
    y_t = y + hat_trans * np.sin(theta + hat_rot1)
    theta_t = theta + hat_rot1 + hat_rot2
    ##END_STUDENT_CODE

    ##END_STUDENT_CODE
    shape_check("x_t",x_t,())
    shape_check("y_t",y_t,())
    shape_check("theta_t",theta_t,())
    return np.array([x_t, y_t, theta_t])


def compute_weights(
    x_pose, z_obs, gridmap, likelihood_map, map_res, p_outside=0.1, max_range=10
):
    ##STUDENT_CODE:
    # 1. Extract ranges and angles from the observation z_obs
    # z_obs[0, :] are angles, z_obs[1, :] are ranges
    all_ranges = z_obs[1, :]
    all_angles = z_obs[0, :]

    # 2. Filter sensor readings that do not lie within max_range
    # We only care about beams that actually hit something within range.
    valid_idx = all_ranges < max_range
    valid_ranges = all_ranges[valid_idx]
    valid_angles = all_angles[valid_idx]

    # 3. Transform observation to gridmap coordinates
    # We pass the filtered ranges, angles, the particle pose, and map info to the helper function.
    # This returns the [x, y] indices on the grid map.
    m_points = ranges2cells(valid_ranges, valid_angles, x_pose, gridmap, map_res)
    
    # m_points is a (2, N) array where row 0 is x (col) and row 1 is y (row)
    m_x = m_points[0, :]
    m_y = m_points[1, :]

    # 4. Count beams that land outside the map
    # We need to check if the coordinates are within the bounds of the likelihood_map
    max_y, max_x = likelihood_map.shape
    
    # Boolean mask: True if inside, False if outside
    is_inside = (m_x >= 0) & (m_x < max_x) & (m_y >= 0) & (m_y < max_y)
    
    # 5. Compute probability for those outside beam occurrences.
    # Count how many False values are in is_inside
    num_outside = np.sum(~is_inside)
    
    # Initialize weight
    weight = 1.0
    
    # Apply p_outside for every beam that landed outside
    if num_outside > 0:
        weight *= np.power(p_outside, num_outside)

    # 6. Obtain probabilities from beams that end (i,j) within the gridmap
    # Get the coordinates that are valid
    inside_x = m_x[is_inside]
    inside_y = m_y[is_inside]
    
    # Look up the likelihoods. Note: arrays are accessed as [row, col] -> [y, x]
    p_ij = likelihood_map[inside_y, inside_x]

    # 7. Compute final weight by combining both subsets of beams.
    if len(p_ij) > 0:
        weight *= np.prod(p_ij)

    ##END_STUDENT_CODE
    shape_check("weight",weight,())
    return weight

def low_variance_resampler(particles, weights):
    ##STUDENT_CODE: #TODO Q3: Implement the low variance resampling strategy



    ##STUDENT_CODE:
    N = particles.shape[0]
    
    # Initialize output arrays
    resampled_particles = np.zeros((N, 3))
    resampled_idx = np.zeros(N, dtype=int)
    
    # 1. Compute random resample_offset in range [0:1/N]
    # This ensures we don't always pick the exact same subset if weights are constant.
    resample_offset = np.random.uniform(0, 1.0/N)
    
    # 2. Iterate over N evenly distributed bins [0,1] shifted by offset_start
    # "c" represents the cumulative weight sum.
    c = weights[0]
    i = 0  # Index of the current particle we are considering
    
    for j in range(N):
        # The "pointer" moves along the line
        u = resample_offset + (j * (1.0/N))
        
        # Move the particle index 'i' forward until the cumulative sum 'c' is greater than 'u'
        while u > c:
            i = i + 1
            # Safety check for numerical instability (floating point errors)
            if i >= N:
                i = N - 1
                break
            c = c + weights[i]
            
        # 3. Store resampled particle and corresponding index
        resampled_particles[j] = particles[i]
        resampled_idx[j] = i

    ##END_STUDENT_CODE




    ##END_STUDENT_CODE
    assert resampled_particles[:, 0].size == resampled_idx.size, (
        "Both arrays need to have the same number of particles!"
    )
    shape_check("resampled_particles",resampled_particles,(N,3))
    shape_check("resampled_idx",resampled_idx,(N,))
    shape_check("resample_offset",resample_offset,float)
    return resampled_particles, resampled_idx, resample_offset


def mc_localization(
    odom,
    z,
    num_particles,
    particles,
    noise,
    gridmap,
    likelihood_map,
    map_res,
    img_map,
    viz_timestep_intervall=None,
    parallel_mode=False,
    surpress_pb=False
):
    # Init GifWriter
    file_name, fig, GifWriter = init_plot_mc_localization(num_particles, parallel_mode)
    init_particles = particles.copy() 

    with GifWriter.saving(fig, file_name, dpi=150):
        
        VECTORIZED = parallel_mode
        
        # Iterate over all timesteps
        for timestep in range(len(odom)):

            # Get current control (u) and measurement (z)
            u_t = odom[timestep]
            z_t = z[timestep]

            if not VECTORIZED:
                # ----------------------------------------
                # Sequential Implementation (Loop)
                # ----------------------------------------
                
                # Initialize weights
                weights = np.zeros(num_particles)

                # Iterate over all particles
                for i in range(num_particles):
                    # 1. Motion Update (Prediction)
                    # USE THE HELPER FUNCTION HERE
                    particles[i, :] = sample_motion_model_odometry(particles[i, :], u_t, noise)
                    
                    # 2. Sensor Update (Correction)
                    # USE THE HELPER FUNCTION HERE
                    weights[i] = compute_weights(particles[i, :], z_t, gridmap, likelihood_map, map_res)

            else:
                # ----------------------------------------
                # Vectorized Implementation (Parallel)
                # ----------------------------------------
                # 1. Motion Update
                particles = sample_motion_model_odometry_parallel(particles, u_t, noise)
                
                # 2. Sensor Update
                weights = compute_weights_parallel(particles, z_t, gridmap, likelihood_map, map_res)

            # 3. Normalization
            # Ensure weights sum to 1
            w_sum = np.sum(weights)
            if w_sum > 0:
                weights /= w_sum
            else:
                # Fallback if track lost
                weights = np.ones(num_particles) / num_particles

            # 4. Resampling
            if not VECTORIZED:
                particles, _, _ = low_variance_resampler(particles, weights)
            else:
                particles, _, _ = low_variance_resampler_parallel(particles, weights)

            # Progress Bar
            if not surpress_pb:
                progress_bar(timestep+1, len(odom))

            # Visualization
            if (
                viz_timestep_intervall is not None
                and timestep % viz_timestep_intervall == 0
            ):
                plot_mc_localization(
                    GifWriter,
                    init_particles,
                    particles,
                    img_map,
                    map_res,
                    f"Timestep: {timestep}",
                    "Runtime particles",
                    ".b",
                )
        
        if not surpress_pb:
            clear_output(wait=True)

        # Final plotting result
        if viz_timestep_intervall is not None:
            plot_mc_localization(
                GifWriter,
                init_particles,
                particles,
                img_map,
                map_res,
                f"Final timestep: {timestep}",
                "Final particles",
                ".g",
            )
            # Create the PNG file (this is the 'final-state' file you saw)
            file_name2 = "FINAL_STATE_" + file_name.replace("gif", "png")
            print(f"Saved GIF: {file_name}\n      IMG: {file_name2}")
            plt.savefig(file_name2)
        plt.close("all")
        
    # Only display if the file was successfully created
    if viz_timestep_intervall is not None:
        if os.path.exists(file_name):
            display(Image(filename=file_name))
        else:
            print("Warning: GIF file was not created. Check PillowWriter implementation.")
    else:
        return particles

def get_sample_parallel(std, count):
    ##STUDENT_CODE #TODO Q5: 
    # Generate (12, count) samples, sum along axis 0 to get (count,)
    samples = np.random.uniform(-std, std, (12, count))
    tot = np.sum(samples, axis=0)
    ##END_STUDENT_CODE
    shape_check("tot",tot,(count,))
    return 0.5 * tot

def v2t_parallel(poses):
    ##STUDENT_CODE #TODO Q5
    n = poses.shape[0]
    c = np.cos(poses[:, 2])
    s = np.sin(poses[:, 2])
    
    # Initialize (N, 3, 3) array
    tr = np.zeros((n, 3, 3))
    
    # Fill matrix elements using array slicing
    tr[:, 0, 0] = c
    tr[:, 0, 1] = -s
    tr[:, 0, 2] = poses[:, 0]
    
    tr[:, 1, 0] = s
    tr[:, 1, 1] = c
    tr[:, 1, 2] = poses[:, 1]
    
    tr[:, 2, 2] = 1.0
    ##END_STUDENT_CODE
    shape_check("tr",tr,(poses.shape[0],3,3))
    return tr

def ranges2points_parallel(ranges, angles):
    ##STUDENT_CODE #TODO Q5
    # Standard polar to cartesian
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    
    # Stack to create (2, K) -> then append ones for homogeneous (3, K)
    points = np.stack([x, y], axis=0)
    points_hom = np.vstack([points, np.ones((1, points.shape[1]))])
    ##END_STUDENT_CODE
    shape_check("points_hom",points_hom,(3,ranges.shape[0]))
    return points_hom

def world2map_parallel(poses, gridmap, map_res):
    ##STUDENT_CODE #TODO Q5
    # Works for both (N, 3) poses and (N, 2, K) points due to broadcasting
    max_y = np.size(gridmap, 0) - 1
    new_poses = np.zeros_like(poses)
    
    # Check dimensions to handle both Poses (N,3) and Points (N,2,K)
    # However, the skeleton defines specific returns. 
    # Usually world2map handles coordinates.
    
    if poses.ndim == 2: # Poses (N, 3)
        new_poses[:, 0] = np.round(poses[:, 0] / map_res)
        new_poses[:, 1] = max_y - np.round(poses[:, 1] / map_res)
    else: # Points (N, 2, K) or similar structure
        # Just apply the formula element-wise
        new_poses[..., 0, :] = np.round(poses[..., 0, :] / map_res)
        new_poses[..., 1, :] = max_y - np.round(poses[..., 1, :] / map_res)
        
    ##END_STUDENT_CODE
    shape_check("new_poses",new_poses,poses.shape)
    return new_poses.astype(int)

def ranges2cells_parallel(r_ranges, r_angles, w_poses, gridmap, map_res):
    ##STUDENT_CODE #TODO Q5
    # 1. Ranges to homogeneous points (3, K)
    r_points = ranges2points_parallel(r_ranges, r_angles)
    
    # 2. Poses to transforms (N, 3, 3)
    w_P = v2t_parallel(w_poses)
    
    # 3. Apply transform (N, 3, 3) @ (3, K) -> (N, 3, K)
    # matmul broadcasts automatically: (N, 3, 3) x (1, 3, K)
    w_points = np.matmul(w_P, r_points)
    
    # 4. World to Map
    # Extract x,y only -> (N, 2, K)
    w_points_2d = w_points[:, 0:2, :]
    
    # Reuse world2map logic explicitly here to ensure shape consistency
    max_y = np.size(gridmap, 0) - 1
    m_points = np.zeros_like(w_points_2d)
    
    m_points[:, 0, :] = np.round(w_points_2d[:, 0, :] / map_res)
    m_points[:, 1, :] = max_y - np.round(w_points_2d[:, 1, :] / map_res)
    
    ##END_STUDENT_CODE
    shape_check("m_points",m_points,(w_poses.shape[0],2,r_ranges.shape[0]))
    return m_points.astype(int)

def sample_motion_model_odometry_parallel(poses_t_1, u_t, alpha):
    ##STUDENT_CODE #TODO Q5
    x = poses_t_1[:, 0]
    y = poses_t_1[:, 1]
    theta = poses_t_1[:, 2]
    
    rot1, trans, rot2 = u_t
    a1, a2, a3, a4 = alpha
    N = poses_t_1.shape[0]

    # 1. Variances (Scalar values, same for all particles based on command)
    sigma_rot1  = np.sqrt(a1 * np.abs(rot1) + a2 * np.abs(trans))
    sigma_trans = np.sqrt(a3 * np.abs(trans) + a4 * (np.abs(rot1) + np.abs(rot2)))
    sigma_rot2  = np.sqrt(a1 * np.abs(rot2) + a2 * np.abs(trans))

    # 2. Sample noise (Vectorized)
    # We can use np.random.normal directly, or the get_sample_parallel helper
    # Using np.random.normal for speed and standard practice, but let's use get_sample_parallel if strictly required
    # noisy_rot1 = rot1 + get_sample_parallel(sigma_rot1, N) ...
    
    hat_rot1  = rot1  + np.random.normal(0, sigma_rot1, N)
    hat_trans = trans + np.random.normal(0, sigma_trans, N)
    hat_rot2  = rot2  + np.random.normal(0, sigma_rot2, N)

    # 3. Update
    x_t = x + hat_trans * np.cos(theta + hat_rot1)
    y_t = y + hat_trans * np.sin(theta + hat_rot1)
    theta_t = theta + hat_rot1 + hat_rot2
    
    ##END_STUDENT_CODE
    shape_check("x_t",x_t,(poses_t_1.shape[0],))
    shape_check("y_t",y_t,(poses_t_1.shape[0],))
    shape_check("theta_t",theta_t,(poses_t_1.shape[0],))
    return np.stack([x_t, y_t, theta_t], axis=1)

def compute_weights_parallel(
    x_poses, z_obs, gridmap, likelihood_map, map_res, p_outside=0.1, max_range=10
):
    ##STUDENT_CODE #TODO Q5: Compute particle weights
    N = x_poses.shape[0]
    
    # 1. Filter valid beams
    ranges = z_obs[1, :]
    angles = z_obs[0, :]
    valid_idx = ranges < max_range
    valid_ranges = ranges[valid_idx]
    valid_angles = angles[valid_idx]
    
    # If no valid beams, return uniform weights
    if len(valid_ranges) == 0:
        return np.ones(N)

    # 2. Transform to map cells: Shape (N, 2, K_valid)
    m_points = ranges2cells_parallel(valid_ranges, valid_angles, x_poses, gridmap, map_res)
    m_x = m_points[:, 0, :] # Shape (N, K)
    m_y = m_points[:, 1, :] # Shape (N, K)

    # 3. Check boundaries (Vectorized)
    max_y, max_x = likelihood_map.shape
    is_inside = (m_x >= 0) & (m_x < max_x) & (m_y >= 0) & (m_y < max_y)

    # 4. Prepare probability matrix
    # Initialize with p_outside. 
    # We will overwrite entries that are INSIDE the map with values from likelihood_map
    probs = np.full(m_x.shape, p_outside)
    
    # 5. Advanced Indexing for Likelihood Lookup
    # We clamp indices to be safe for lookup, then apply mask.
    # (Clamping ensures we don't crash on lookup, even though we discard those values later)
    safe_x = np.clip(m_x, 0, max_x - 1)
    safe_y = np.clip(m_y, 0, max_y - 1)
    
    # Look up all probabilities at once
    map_probs = likelihood_map[safe_y, safe_x]
    
    # 6. Combine: Use map probability if inside, else keep p_outside
    # np.where(condition, x, y) -> if condition True yield x, else yield y
    final_beam_probs = np.where(is_inside, map_probs, p_outside)
    
    # 7. Product over beams (axis 1) to get weight per particle
    weights = np.prod(final_beam_probs, axis=1)

    ##END_STUDENT_CODE
    shape_check("weights",weights,(N,))
    return weights

def low_variance_resampler_parallel(particles, weights):
    ##STUDENT_CODE: #TODO Q3: Implement the low variance resampling strategy
    N = particles.shape[0]
    
    # 1. Compute random resample_offset
    resample_offset = np.random.uniform(0, 1.0/N)
    
    # 2. Create the "Pointer" array U (The comb)
    # Linspace from offset to 1+offset, excluding endpoint depending on implementation
    # Easiest way: offset + [0, 1/N, 2/N, ... (N-1)/N]
    pointers = resample_offset + (np.arange(N) * (1.0/N))
    
    # 3. Cumulative Sum of weights (The scale)
    cumsum_weights = np.cumsum(weights)
    
    # Ensure the last element is exactly 1.0 (or slightly higher due to float errors) to catch the last pointer
    cumsum_weights[-1] = 1.0 + resample_offset + 1e-9 

    # 4. Vectorized Search
    # Find which bin in cumsum_weights each pointer falls into.
    # searchsorted returns the index i such that C[i-1] < pointer <= C[i]
    resampled_idx = np.searchsorted(cumsum_weights, pointers)
    
    # 5. Indexing
    resampled_particles = particles[resampled_idx]

    ##END_STUDENT_CODE
    shape_check("resampled_particles",resampled_particles,(N,3))
    shape_check("resampled_idx",resampled_idx,(N,))
    shape_check("resample_offset",resample_offset,float)
    return resampled_particles, resampled_idx, resample_offset

#######################################################
#   Transformation helper functions - Do not modify!  #
#######################################################


def world2map(pose, gridmap, map_res):
    """Trasnform world coordinates to discrete map coordinates"""
    max_y = np.size(gridmap, 0) - 1
    new_pose = np.zeros_like(pose)
    new_pose[0] = np.round(pose[0] / map_res)
    new_pose[1] = max_y - np.round(pose[1] / map_res)
    return new_pose.astype(int)


def v2t(pose):
    """Transform pose vector pose v = [x, y, θ] into a 2D homogeneous transformation matrix t(3x3)."""
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr


def t2v(tr):
    """Transform 2D homogeneous transformation matrix t(3x3) into pose vector pose v = [x, y, θ] into a ."""
    x = tr[0, 2]
    y = tr[1, 2]
    th = np.arctan2(tr[1, 0], tr[0, 0])
    v = np.array([x, y, th])
    return v


def ranges2points(ranges, angles):
    """Converts ranges and angles to homogeneous points."""
    # 2D points
    points = np.array(
        [np.multiply(ranges, np.cos(angles)), np.multiply(ranges, np.sin(angles))]
    )
    # homogeneous points
    points_hom = np.append(points, np.ones((1, np.size(points, 1))), axis=0)
    return points_hom


def ranges2cells(r_ranges, r_angles, w_pose, gridmap, map_res):
    """Computes the endpoints of raw scans into map coordinates.
    Returns: Numpy array with x,y coordinates for each observation"""

    # ranges to points
    r_points = ranges2points(r_ranges, r_angles)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # world to map
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2, :]
    return m_points


def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose

def wrapToPi(theta):
    while theta < -np.pi:
        theta = theta + 2 * np.pi
    while theta > np.pi:
        theta = theta - 2 * np.pi
    return theta


##############################################
#   Random helper functions - Do not modify!  #
##############################################


def init_uniform(num_particles, img_map, map_res):
    particles = np.zeros((num_particles, 3))
    particles[:, 0] = np.random.rand(num_particles) * np.size(img_map, 1) * map_res
    particles[:, 1] = np.random.rand(num_particles) * np.size(img_map, 0) * map_res
    particles[:, 2] = np.random.rand(num_particles) * 2 * np.pi

    return particles


def get_sample(std):
    # Irwin–Hall -> Using Central Limit Theorem to generate cheap normal distributed samples.
    tot = 0
    for i in range(12):
        tot += np.random.uniform(-std, std)

    return 0.5 * tot


##################################################
#   Debugging helper functions - Do not modify!  #
##################################################

def shape_check(tag,obj,target_shape):

    if target_shape==(): 
        if isinstance(obj,np.generic):
            if obj.shape!=target_shape:
                assert False,(EMSG(call_function()+"\n"+f"{tag} should be a scalar!"))
        elif not isinstance(obj, (int, float, complex, bool)):
            assert False,(EMSG(call_function()+"\n"+f"{tag} should be a scalar!"))

    elif isinstance(obj,np.ndarray):
        if obj.shape!=target_shape:
            assert False,(EMSG(call_function()+"\n"+f"{tag}.shape is {obj.shape} but should be ({target_shape},)!"))

RED = "\033[1;31m"
RESET = "\033[0m"

def EMSG(msg)->str:
    return RED + msg + RESET

def call_function(steps: int = 2):
    frame = inspect.currentframe()
    for _ in range(steps):
        if frame is None:
            break  # reached the top frame

        frame = frame.f_back

    if frame is None:
        return f"No frame found"
    else:
        return f'Wrong outputshape in {frame.f_code.co_name}:'


##############################################
#   Plot helper functions - Do not modify!  #
##############################################


def plot_observation(z):
    points = np.array(
        [np.multiply(z[0, :], np.cos(z[1, :])), np.multiply(z[0, :], np.sin(z[1, :]))]
    )
    plt.scatter(points[0, :], points[1, :], marker="x")
    plt.scatter(0, 0, marker="o", color="red")
    plt.show()


def plot_particles(particles, img_map, map_res, s, legend=None, size=None):
    ax = plt.gca()
    ax.matshow(255 - img_map, cmap="Greys", origin="upper")
    max_y = np.size(img_map, 0) - 1
    xs = np.copy(particles[:, 0]) / map_res
    ys = max_y - np.copy(particles[:, 1]) / map_res
    ax.plot(xs, ys, s, label=legend, markersize=size)
    if legend is not None:
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    ax.set_xlim(0, np.size(img_map, 1))
    ax.set_ylim(0, np.size(img_map, 0))
    ax.xaxis.tick_bottom()


def plot_low_variance_resampler(weights, num_particles, resampled_idx, offset):
    base_colors = matplotlib.colormaps["tab20"]
    colors = [base_colors(i % 10) for i in range(num_particles)]

    w_sum = np.cumsum(weights)

    w_offset = offset
    w_bin = np.linspace(0 + w_offset, 1 + w_offset, num_particles + 1)

    ax = plt.gca()

    rect = Rectangle(
        xy=(0, -1.9),
        width=1 / num_particles + 1,
        height=0.8,
        facecolor="black",
        edgecolor="black",
    )
    ax.add_patch(rect)

    plt.text(
        0 + 0.2 / num_particles, -1.7, "Random\n Offset", color="red", size=8, zorder=1
    )
    plt.text(
        1 + 0.2 / num_particles, -1.7, "Random\n Offset", color="red", size=8, zorder=1
    )

    for m in range(num_particles):
        rect = Rectangle(
            xy=(w_sum[m], m - 0.5),
            width=-weights[m],
            height=1,
            facecolor=colors[m],
            edgecolor="black",
            zorder=10,
        )
        ax.add_patch(rect)

        indices = np.where(resampled_idx == m)[0]

        for i in indices:
            plt.plot([w_bin[i]], [m], "x", color="black")
            plt.plot([w_bin[i]], [-1.5], "x", color="black")

            plt.plot(
                [w_bin[i], w_bin[i]], [-1, m - 0.5], color=colors[m], linestyle="--"
            )
            rect = Rectangle(
                xy=(w_bin[i], -2),
                width=w_bin[i + 1] - w_bin[i],
                height=1,
                facecolor=colors[m],
                edgecolor="black",
            )
            ax.add_patch(rect)

    plt.xlim(-0.05, 1.05 + 1 / num_particles)

    special_positions = [-1.5] + list(range(0, num_particles))
    special_labels = ["Resampled \nParticles  "] + [
        f"ID: {i}" for i in range(num_particles)
    ]

    plt.yticks(special_positions, special_labels)
    plt.show()


def plot_mc_localization(
    GifWriter, init_particles, particles, img_map, map_res, title, legend, s
):
    fig = plt.gcf()
    plt.clf()
    plt.title(title)
    size = get_marker_size(particles.shape[0])
    fig.subplots_adjust(left=0.15, right=0.90, bottom=0.0, top=1.2)
    plot_particles(
        init_particles, img_map, map_res, ".r", legend="Inital particles", size=size
    )
    plot_particles(particles, img_map, map_res, s, legend=legend, size=size)
    plt.grid(color="gray", linestyle="--", linewidth=0.7)
    fig.canvas.draw()
    GifWriter.grab_frame()


def get_marker_size(n_points):
    if n_points <= POINTSIZE_TRESHHOLDS[0][0]:
        return POINTSIZE_TRESHHOLDS[0][1]
    if n_points >= POINTSIZE_TRESHHOLDS[-1][0]:
        return POINTSIZE_TRESHHOLDS[-1][1]

    for i in range(len(POINTSIZE_TRESHHOLDS) - 1):
        x0, y0 = POINTSIZE_TRESHHOLDS[i]
        x1, y1 = POINTSIZE_TRESHHOLDS[i + 1]
        if x0 <= n_points <= x1:
            size = y0 + (n_points - x0) * (y1 - y0) / (x1 - x0)
            return size


def init_plot_mc_localization(num_particles, VECTORIZED):
    fig, ax = plt.subplots(figsize=(3, 5))
    GifWriter = PillowWriter_CUSTOM(fps=5, loop=None, fig=fig)

    if VECTORIZED is False:
        msg = "MCL"
    else:
        msg = "VectorizedMCL"
    file_name = f"{msg}_p={num_particles}.gif"
    return file_name, fig, GifWriter


class PillowWriter_CUSTOM(PillowWriter):
    def __init__(self, fps=10, loop=0, fig=None):
        super().__init__(fps=fps)
        self.loop = loop
        self._frames = []
        self._fig = fig

    def finish(self):
        if len(self._frames) != 0:
            self._frames[0].save(
                self.outfile,
                save_all=True,
                append_images=self._frames[1:],
                duration=int(1000 / self.fps),
                loop=self.loop,
            )


def plot_parallelization_comparison(
    start_particles, total_steps, scale_factor=4,reps_each=1, tsteps=10, return_data=False, surpress_pb=False
):
    noise = [0.1, 0.1, 0.1, 0.1]
    odom = [0.0, 0.1, 1.0]

    data = pickle.load(
        open(
            "/home/lobmaier/TEACHING/exercises-sse/2025/ex06-monte-carlo-localization/solution/dataset_mit_csail.p",
            "rb",
        )
    )
    gridmap = 255 - data["img_map"]
    likelihood_map = data["likelihood_map"]
    odom = data["odom"][:tsteps]
    z = data["z"][:tsteps]
    img_map = data["img_map"]
    map_res = 0.01

    iterations = [start_particles * scale_factor * i for i in range(1, total_steps + 1)]

    data_plot = {"par": [], "nonpar": [], "num_par": []}

    
    for num_particles in iterations:
        if not surpress_pb:
            progress_bar(num_particles, iterations[-1])
        for rep in range(reps_each):     
            particles = init_uniform(num_particles, data["img_map"], map_res)
            t1 = timeit.default_timer()
            mc_localization(
                odom,
                z,
                num_particles,
                particles,
                noise,
                gridmap,
                likelihood_map,
                map_res,
                img_map,
                viz_timestep_intervall=None,
                parallel_mode=False,
                surpress_pb=True
            )
            t2 = timeit.default_timer()
            mc_localization(
                odom,
                z,
                num_particles,
                particles,
                noise,
                gridmap,
                likelihood_map,
                map_res,
                img_map,
                viz_timestep_intervall=None,
                parallel_mode=True,
                surpress_pb=True
            )
            t3 = timeit.default_timer()

            nonpar = t2 - t1
            par = t3 - t2
            data_plot["par"].append(par)
            data_plot["nonpar"].append(nonpar)
            data_plot["num_par"].append(num_particles)
        # print("-"*100)
    if not surpress_pb:
        clear_output(wait=True)
        

    plt.close("all")
    x = data_plot["num_par"]

    x_unique = np.unique(data_plot["num_par"])
    y_means_par = [
        np.mean([y for x, y in zip(data_plot["num_par"], data_plot["par"]) if x == xi])
        for xi in x_unique
    ]
    y_means_nonpar = [
        np.mean(
            [y for x, y in zip(data_plot["num_par"], data_plot["nonpar"]) if x == xi]
        )
        for xi in x_unique
    ]

    # Fit linear trend lines
    par_fit = np.polyfit(x_unique, y_means_par, 1)  # degree 1 = linear
    nonpar_fit = np.polyfit(x_unique, y_means_nonpar, 1)

    # Generate y-values from the fit
    par_trend = np.polyval(par_fit, x)
    nonpar_trend = np.polyval(nonpar_fit, x)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.title("Scaling Comparison")
    plt.scatter(x, data_plot["par"], marker="x", c="orange", label="Parallelized")
    plt.plot(x, par_trend, "--", c="orange")
    plt.scatter(x, data_plot["nonpar"], marker="x", c="blue", label="Non-parallelized")
    plt.plot(x, nonpar_trend, "--", c="blue")
    plt.xlabel("Index")
    plt.ylabel("RunTime (s)")
    plt.legend(loc="upper left")

    ax = plt.gca()
    ax2 = ax.twinx() 
    y_improve = [nonpar/par for par, nonpar in zip(par_trend,nonpar_trend)]
    ax2.plot(x,y_improve, "--", c="green",label="nonpar/par")
    plt.ylabel("Runtime Decrease Factor")
    
    plt.legend(loc="upper right")
    plt.grid(True)

    if return_data:
        return y_improve, data_plot
    else:
        plt.savefig("ScalingOfMCL")
        plt.show()

def progress_bar(iteration, total, bar_length=30):
    percent = iteration / total
    filled_len = int(bar_length * percent)
    bar = '=' * filled_len + '-' * (bar_length - filled_len)
    clear_output(wait=True)  # update in-place
    
    print(f'Progress: |{bar}| {percent*100:.1f}%')