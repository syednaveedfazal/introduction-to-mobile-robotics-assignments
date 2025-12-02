

import numpy as np
import matplotlib.pyplot as plt


def inverse_motion_model(pose_t_1, pose_t):
    ##STUDENT_CODE:
    
    # Unpack the poses
    x1, y1, theta1 = pose_t_1
    x2, y2, theta2 = pose_t

    # Calculate the differences in position
    dx = x2 - x1
    dy = y2 - y1

    # Calculate the translational movement (Euclidean distance)
    trans = np.sqrt(dx**2 + dy**2)

    # Calculate the angle of the translation vector
    # arctan2 handles the quadrants correctly (-pi to pi)
    heading_direction = np.arctan2(dy, dx)

    # Calculate the first rotation (angle to turn towards the target point)
    rot1 = heading_direction - theta1

    # Calculate the second rotation (angle to turn to final heading after reaching target)
    rot2 = theta2 - heading_direction

    ##END_STUDENT_CODE:
    return rot1, trans, rot2


def probability_density(mean, variance):
    """
    Calculates the probability density of a zero-mean normal distribution
    evaluated at value 'mean' with the given 'variance'.
    Formula: (1 / sqrt(2 * pi * var)) * exp(-x^2 / (2 * var))
    """
    variance = max(variance,0.00001) #Avoid division by 0!

    ##STUDENT_CODE:
    # Calculate the standard deviation (sigma)
    sigma = np.sqrt(variance)
    
    # Calculate the coefficient 1 / (sigma * sqrt(2*pi))
    coefficient = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    
    # Calculate the exponential part exp(-x^2 / (2*sigma^2))
    # Note: 'mean' here represents the value x (the error)
    exponent = - (mean ** 2) / (2.0 * variance)
    
    density = coefficient * np.exp(exponent)
    ##END_STUDENT_CODE
    return density

def motion_model(x_t, x_t_1, u_t, alpha, marginalise_p3=False):

    ##STUDENT_CODE:
    # 1. Calculate the measured motion parameters (deltas) from the odometry readings u_t
    # u_t = [pose_odom_t-1, pose_odom_t]
    delta_rot1, delta_trans, delta_rot2 = inverse_motion_model(u_t[0], u_t[1])

    # 2. Calculate the hypothetical motion parameters (hat_deltas) required to move from x_t_1 to x_t
    # These are the 'expected' motion values given the map coordinates
    hat_delta_rot1, hat_delta_trans, hat_delta_rot2 = inverse_motion_model(x_t_1, x_t)

    # 3. Calculate the errors (difference between measured and hypothetical)
    error_rot1 = delta_rot1 - hat_delta_rot1
    error_trans = delta_trans - hat_delta_trans
    error_rot2 = delta_rot2 - hat_delta_rot2

    # Normalize angular errors to range [-pi, pi]
    error_rot1 = (error_rot1 + np.pi) % (2 * np.pi) - np.pi
    error_rot2 = (error_rot2 + np.pi) % (2 * np.pi) - np.pi

    # 4. Calculate the variances (sigma^2) for the motion noise model
    # Based on the noise parameters alpha and the hypothetical motion
    # alpha = [alpha1, alpha2, alpha3, alpha4]
    
    # Variance for first rotation: depends on rotation and translation magnitude
    var_rot1 = alpha[0] * (hat_delta_rot1**2) + alpha[1] * (hat_delta_trans**2)
    
    # Variance for translation: depends on translation and total rotation magnitude
    var_trans = alpha[2] * (hat_delta_trans**2) + alpha[3] * (hat_delta_rot1**2 + hat_delta_rot2**2)
    
    # Variance for second rotation: depends on rotation and translation magnitude
    var_rot2 = alpha[0] * (hat_delta_rot2**2) + alpha[1] * (hat_delta_trans**2)

    # 5. Calculate the probability densities
    # Assuming probability_density(error, variance) is available from the context
    # If variance is extremely small, add a tiny epsilon to avoid division by zero if the function doesn't handle it
    p1 = probability_density(error_rot1, var_rot1)
    p2 = probability_density(error_trans, var_trans)
    p3 = probability_density(error_rot2, var_rot2)

    ##END_STUDENT_CODE 
    if marginalise_p3: 
        return p1 * p2
    else:
        return p1 * p2 * p3


def plot_posterior_belief(x_t_1, u_t, alpha, ret=False, marginalise_p3=False, N_theta = 8):
    size = 150
    gridmap = np.zeros([size, size])
    origin = [np.floor(size/2),np.floor(size/2)]
    res = 0.01

    
    if not marginalise_p3:
        marginalise_p3 = False
        dtheta = 2 * np.pi / N_theta
        

        ##STUDENT_CODE #TODO #3.4 A) Compute gridmap
        
        # Iterate over every cell in the grid
        for r in range(size):
            for c in range(size):
                # Convert grid indices (row, col) to map coordinates (x, y)
                # x corresponds to columns, y corresponds to rows.
                # We assume the map origin is in the center.
                # Usually in image coords: y increases downwards, but map y increases upwards.
                # With origin at center (75,75):
                # Col 0 -> x = -0.75, Col 150 -> x = +0.75
                # Row 0 -> y = +0.75, Row 150 -> y = -0.75
                x_val = (c - origin[0]) * res
                y_val = (origin[1] - r) * res
                
                prob_sum = 0.0
                
                # Integrate over all possible orientations (N_theta steps)
                for k in range(N_theta):
                    # Angle from -pi to pi
                    theta_val = -np.pi + k * dtheta
                    
                    # Create the hypothesized pose xt
                    x_t_hyp = [x_val, y_val, theta_val]
                    
                    # Sum the probability density
                    prob_sum += motion_model(x_t_hyp, x_t_1, u_t, alpha)
                
                # Assign the integrated probability to the grid cell
                gridmap[r, c] = prob_sum

        ##END_STUDENT_CODE 

    
    else:
        marginalise_p3 = True
        ##STUDENT_CODE #TODO #3.4 B) Use Marginalised motion_model marginalise_p3=True and pass to motion_model!
        
        # Iterate over every cell in the grid
        for r in range(size):
            for c in range(size):
                # Convert grid indices to map coordinates
                x_val = (c - origin[0]) * res
                y_val = (origin[1] - r) * res
                
                # Construct hypothesis pose (theta doesn't matter for p1*p2 calculation)
                x_t_hyp = [x_val, y_val, 0.0]
                
                # Calculate marginalized probability
                prob = motion_model(x_t_hyp, x_t_1, u_t, alpha, marginalise_p3=True)
                
                gridmap[r, c] = prob

        ##END_STUDENT_CODE 

    if ret:
        return gridmap
    
    # Normalize
    assert np.sum(gridmap)!=0, "Sum over gridmap is zero! - Error or no implementation!"
    gridmap = gridmap/np.sum(gridmap)
    plt.imshow(1-gridmap, cmap="gray", extent=[-res*(size - origin[0]), res*(size - origin[0]), -res*(size - origin[1]), res*(size - origin[1])])
    plt.show()
def evaluate_sample_odometry(alpha, pose_0, odom, ret=False):
    plot_lims = [(0,6),(2.5,6.5)]

    # get ground truth poses
    gt_poses = [pose_0]
    last_pose = pose_0
    for odom_idx in range(len(odom) - 1):

        ##STUDENT_CODE #TODO: compute xt, yt, theta_t from odometry
        u_t_curr = [odom[odom_idx], odom[odom_idx+1]]
        
        # Unpack odometry readings
        x_bar, y_bar, theta_bar = u_t_curr[0]
        x_bar_prime, y_bar_prime, theta_bar_prime = u_t_curr[1]
        
        # Calculate deltas (Inverse motion model)
        delta_rot1 = np.arctan2(y_bar_prime - y_bar, x_bar_prime - x_bar) - theta_bar
        delta_trans = np.sqrt((x_bar - x_bar_prime)**2 + (y_bar - y_bar_prime)**2)
        delta_rot2 = theta_bar_prime - theta_bar - delta_rot1
        
        # Apply deltas to last_pose (Deterministic/Ground Truth calculation)
        x_prev, y_prev, theta_prev = last_pose
        
        x_t = x_prev + delta_trans * np.cos(theta_prev + delta_rot1)
        y_t = y_prev + delta_trans * np.sin(theta_prev + delta_rot1)
        theta_t = theta_prev + delta_rot1 + delta_rot2


        ##END_STUDENT_CODE
        current_pose = [x_t, y_t, theta_t]
        gt_poses.append(current_pose)
        last_pose = current_pose

    gt_poses = np.array(gt_poses)
   
    # draw samples incrementally
    current_pose = pose_0
    estimated_poses = [pose_0]
    samples_for_plot = []

    num_samples = 1000
    samples = np.zeros((num_samples, 3))
    samples[:, 0] = pose_0[0]
    samples[:, 1] = pose_0[1]

    for odom_idx in range(len(odom) - 1):
        # calculate the new samples

        ##STUDENT_CODE # TODO: update all samples and the current_pose
        u_t_curr = [odom[odom_idx], odom[odom_idx+1]]
        
        # Update every sample individually using the probabilistic motion model
        for i in range(num_samples):
            samples[i] = sample_motion_model(samples[i], u_t_curr, alpha)
            
        # The estimated pose is the mean of all samples
        current_pose = np.mean(samples, axis=0)


        ##END_STUDENT_CODE
        estimated_poses.append(current_pose)
        samples_for_plot.append(np.copy(samples))


    estimated_poses = np.array(estimated_poses)
    if ret:
        return gt_poses, estimated_poses

    print("estimated_poses: ", estimated_poses)

    #plot all samples, simga elipse and estimated and gt poses
    for idx, samples in enumerate(samples_for_plot):
        sc = plt.scatter(samples[:, 0], samples[:, 1],marker='.')
        color = sc.get_facecolor()[0]
        ax = plt.gca()
        confidence_ellipse(samples[:, 0], samples[:, 1], ax, edgecolor=color,n_std=2)
        plt.scatter(estimated_poses[idx+1, 0], estimated_poses[idx+1 ,1], marker='D',color=color,edgecolor='black')
        plt.scatter(gt_poses[idx+1, 0], gt_poses[idx+1, 1], marker='*',color=color,edgecolor='black', )

    plt.scatter(estimated_poses[0, 0], estimated_poses[0 ,1], marker='D',color='black',edgecolor='black',label="Estimated Pose")
    plt.scatter(gt_poses[0, 0], gt_poses[0, 1], marker='*',color='black',edgecolor='black',label="Ground Truth Pose")    
    plt.plot(0,0,label="2-Sigma Ellipse ~95.45%",color="black",lw=0.5)
        
    plt.xlim(plot_lims[0])
    plt.ylim(plot_lims[1])
    plt.legend()
    plt.show()

def plot_sample_motion_model(pose_t_1, u_t, alpha):
    num_samples = 1000
    samples = np.zeros((num_samples, 3))
    samples[:, 0] = pose_t_1[0]
    samples[:, 1] = pose_t_1[1]

    for i in range(num_samples):
        samples[i] = sample_motion_model(samples[i], u_t, alpha)

    plt.figure()
    plt.plot(samples[:,0], samples[:,1], '.')
    plt.axis('equal')
    plt.show()

def get_sample(std): 
    #Irwin–Hall -> Using Central Limit Theorem to generate cheap normal distributed samples. 
    tot = 0
    for i in range(12):
        tot += np.random.uniform(-std,std)
    
    return 0.5*tot


def sample_motion_model(pose_t_1, u_t, alpha):
    ##STUDENT_CODE
    # Unpack previous pose (map frame)
    x, y, theta = pose_t_1
    
    # Unpack odometry readings (odometry frame)
    x_bar, y_bar, theta_bar = u_t[0]
    x_bar_prime, y_bar_prime, theta_bar_prime = u_t[1]

    # 1. Calculate nominal control parameters (deltas)
    delta_rot1 = np.arctan2(y_bar_prime - y_bar, x_bar_prime - x_bar) - theta_bar
    delta_trans = np.sqrt((x_bar - x_bar_prime)**2 + (y_bar - y_bar_prime)**2)
    delta_rot2 = theta_bar_prime - theta_bar - delta_rot1

    # 2. Calculate noise standard deviations
    # Variances (sigma^2)
    var_rot1 = alpha[0] * delta_rot1**2 + alpha[1] * delta_trans**2
    var_trans = alpha[2] * delta_trans**2 + alpha[3] * (delta_rot1**2 + delta_rot2**2)
    var_rot2 = alpha[0] * delta_rot2**2 + alpha[1] * delta_trans**2
    
    # Standard deviations (sigma) - passed to get_sample
    # We perform sqrt here because get_sample likely expects std_dev based on common implementations
    std_rot1 = np.sqrt(var_rot1)
    std_trans = np.sqrt(var_trans)
    std_rot2 = np.sqrt(var_rot2)

    # 3. Add noise to control parameters
    hat_delta_rot1 = delta_rot1 + get_sample(std_rot1)
    hat_delta_trans = delta_trans + get_sample(std_trans)
    hat_delta_rot2 = delta_rot2 + get_sample(std_rot2)

    # 4. Calculate new pose
    x_t = x + hat_delta_trans * np.cos(theta + hat_delta_rot1)
    y_t = y + hat_delta_trans * np.sin(theta + hat_delta_rot1)
    theta_t = theta + hat_delta_rot1 + hat_delta_rot2

    ##END_STUDENT_CODE
    return x_t, y_t, theta_t



from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
#https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)