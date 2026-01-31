#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FastSLAM algorithm for range-bearing landmark observations.
"""

import copy
import inspect
import json
import os
import random
import subprocess
import sys
from math import atan, atan2, cos, pi, sin, sqrt

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Ellipse
from PIL import Image

TODO = None


class Particle(object):
    """Particle for tracking a robot with a particle filter.

    The particle consists of:
    - a robot pose
    - a weight
    - a map consisting of landmarks
    - list of x,y and theta to init robot pose
    """

    class LandmarkEKF(object):
        """EKF representing a landmark"""

        def __init__(self):
            self.observed = False
            self.mu = np.zeros((2)).astype(
                float
            )  # landmark position as vector of length 2
            self.sigma = np.zeros((2, 2))  # covariance as 2x2 matrix

        def __str__(self):
            return "LandmarkEKF(observed  = {0}, mu = {1}, sigma = {2})".format(
                self.observed, self.mu, self.sigma
            )

        def _shape_check(self):
            shape_check("LandmarkEKF mu", self.mu, (2,))
            shape_check("LandmarkEKF sigma", self.sigma, (2, 2))

    def __init__(self, num_landmarks, motion_noise, landmark_noise, robot_init_pos):
        """Creates the particle and initializes location/orientation"""
        self.motion_noise = motion_noise
        self.landmark_noise = np.diag(landmark_noise)

        # initialize robot pose at origin
        self.pose = np.vstack(robot_init_pos)
        # initialize weights uniformly
        self.weight = 1.0
        self.log_weight = 0.0

        # Trajectory of the particle
        self.trajectory = []

        # initialize the landmarks aka the map
        self.landmarks = [self.LandmarkEKF() for _ in range(num_landmarks)]

    def prediction_step(self, odom):
        """Predict the new pose of the robot
        - odom    : odometry reading for the current timestep
        - odom.r1 : rotational component 1 in radians
        - odom.t  : translational component
        - odom.r2 : rotational component 2 in radians
        """

        # Append the old position
        self.trajectory.append(self.pose)

        # Obtain the noise standard deviations
        s_rot1 = self.motion_noise[0]
        s_trans = self.motion_noise[1]
        s_rot2 = self.motion_noise[2]

        ##STUDENT_CODE: #TODO Q1: Estimate the new position of the particle
        # Sample noisy controls
        rot1_hat = odom.r1 + random.gauss(0.0, sqrt(s_rot1))
        trans_hat = odom.t + random.gauss(0.0, sqrt(s_trans))
        rot2_hat = odom.r2 + random.gauss(0.0, sqrt(s_rot2))

        x, y, theta = self.pose.flatten()
        x_new = x + trans_hat * cos(theta + rot1_hat)
        y_new = y + trans_hat * sin(theta + rot1_hat)
        theta_new = wrapToPi(theta + rot1_hat + rot2_hat)

        self.pose = np.array([[x_new], [y_new], [theta_new]])

        ##END_STUDENT_CODE

    def correction_step(self, sensor_measurements):
        """Weight the particles according to the current map of the particle and the landmark observations z.

        - sensor_measurements                : list of sensor measurements for the current timestep
        - sensor_measurements[i].landmark_id : observed landmark ID
        - sensor_measurements[i].z_range     : measured distance to the landmark in meters
        - sensor_measurements[i].z_bearing   : measured angular direction of the landmark in radians
        """

        # Construct the sensor noise matrix Q (2 x 2)
        Q_t = self.landmark_noise
        joint_likelihood = 1
        log_likelihood = 0.0
        prev_weight = self.weight

        for measurement in sensor_measurements:
            landmark = self.landmarks[measurement.landmark_id]

            # Init landmark if not observed: Create
            if not landmark.observed:
                # Obtain Jacobian
                _, H = self.measurement_model(landmark)

                ##STUDENT_CODE: #TODO Q3 A:  Use the inverse measurement model to initialize the EKF for this landmark
                x, y, theta = self.pose.flatten()
                r = measurement.z_range
                b = measurement.z_bearing
                landmark.mu = np.array([x + r * cos(theta + b), y + r * sin(theta + b)])

                # Compute landmark.sigma. Landmarks are assumed static, so there is no process noise!
                _, H = self.measurement_model(landmark)
                H_inv = np.linalg.inv(H)
                landmark.sigma = H_inv @ Q_t @ H_inv.T

                ##END_STUDENT_CODE
                landmark.observed = True

            # EKF Update on landmark
            else:

                # Obtain expected measurement and Jacobian
                expected_z, H = self.measurement_model(landmark)

                ##STUDENT_CODE: #TODO Q3 B: Correction step of the EKF filter
                # Compute the innovation covariance
                S = H @ landmark.sigma @ H.T + Q_t

                # Compute the Kalman gain
                K = landmark.sigma @ H.T @ np.linalg.inv(S)

                # Compute compute the error between the z and expected_z (remember to normalize the angle)
                innovation = np.array(
                    [
                        measurement.z_range - expected_z[0],
                        wrapToPi(measurement.z_bearing - expected_z[1]),
                    ]
                )

                # Update the mean and covariance of the EKF for the landmark
                landmark.mu = landmark.mu + K @ innovation
                landmark.sigma = (np.eye(2) - K @ H) @ landmark.sigma

                ##STUDENT_CODE: #TODO Q3 C: Compute the likelihood of the observation
                # Gaussian likelihood: p(z | mu, Sigma)
                S_inv = np.linalg.inv(S)
                det_S = np.linalg.det(S)
                det_S = max(float(det_S), 1e-12)
                norm_const = 1.0 / (2.0 * pi * sqrt(det_S))
                exponent = -0.5 * float(innovation.T @ S_inv @ innovation)
                likelihood = norm_const * np.exp(exponent)

                # Update joint_likelihood with the likelihood of this observation
                joint_likelihood *= likelihood
                log_likelihood += np.log(max(likelihood, 1e-300))

                ##END_STUDENT_CODE

        # Multiply with the former weight
        self.weight *= joint_likelihood
        self.log_weight = np.log(max(prev_weight, 1e-300)) + log_likelihood

    def measurement_model(self, landmark_ekf):
        """Compute the expected measurement for a landmark and the Jacobian

        - landmark_ekf: EKF representing the landmark

        Returns a tuple (h, H) where
        - h = [expected_range, expected_bearing] is the expected measurement
        - H is the Jacobian.
        """
        # position (x, y) of the observed landmark_ekf
        landmark_x = landmark_ekf.mu[0]
        landmark_y = landmark_ekf.mu[1]

        ##STUDENT_CODE: #TODO Q2: Complete measurement model
        dx = landmark_x - self.pose[0, 0]
        dy = landmark_y - self.pose[1, 0]
        q = dx * dx + dy * dy
        sqrt_q = sqrt(q)

        # Compute expected measurment h = [expected_range, expected_bearing]
        expected_range = sqrt_q
        expected_bearing = wrapToPi(atan2(dy, dx) - self.pose[2, 0])
        h = np.array([expected_range, expected_bearing])

        # Compute the Jacobian H of the measurement z with respect to the landmark position landmark_x and landmark_y
        H = np.array([[dx / sqrt_q, dy / sqrt_q], [-dy / q, dx / q]])

        ##END_STUDENT_CODE
        return h, H

    def _shape_check(self):
        shape_check("Particle weight", self.weight, ())
        shape_check("Particle pose", self.pose, (3, 1))

        h, H = self.measurement_model(self.LandmarkEKF())
        shape_check("Measurement Model h", h, (2,))
        shape_check("Measurement Model H", H, (2, 2))

        for landmark in self.landmarks:
            landmark._shape_check()


def low_variance_resampler(particles):
    """Low_variance_resampler Probabilistic Robotics page 109."""
    N = len(particles)
    resampled_particles = np.zeros_like(particles)
    weights = [particle.weight for particle in particles]

    ##STUDENT_CODE: #TODO: Implement low variance resampling
    r = random.uniform(0.0, 1.0 / N)
    c = weights[0]
    i = 0
    for m in range(N):
        U = r + m * (1.0 / N)
        while U > c:
            i += 1
            c += weights[i]
        resampled_particles[m] = copy.deepcopy(particles[i])

    # Compared to EX06, particles is a list of objects!

    # Apply uniform weights across resampled particles
    for particle in resampled_particles:
        particle.weight = 1.0 / N
        particle.log_weight = np.log(particle.weight)

    ##END_STUDENT_CODE:
    return resampled_particles


#######################################################
#   Transformation helper functions - Do not modify!  #
#######################################################


def wrapToPi(theta):
    while theta < -np.pi:
        theta = theta + 2 * np.pi
    while theta > np.pi:
        theta = theta - 2 * np.pi
    return theta


########################################
#   Helper functions - Do not modify!  #
########################################


class Odometry(object):
    """Represents an odometry command.

    - r1: initial rotation in radians counterclockwise
    - t: translation in meters
    - r2: final rotation in radians counterclockwise
    """

    def __init__(self, r1, t, r2):
        self.r1 = wrapToPi(r1)
        self.t = t
        self.r2 = wrapToPi(r2)

    def __str__(self):
        return "Odometry(r1 = {0} rad, t = {1} m, r2 = {2} rad)".format(
            self.r1, self.t, self.r2
        )


class SensorMeasurement(object):
    """Represents one or more range and bearing sensor measurements.

    - id: ID of the landmark
    - z_range: measured distance to the landmark in meters
    - z_bearing: measured angle towards the landmark in radians
    """

    def __init__(self, landmark_id, z_range, z_bearing):
        self.landmark_id = landmark_id
        self.z_range = z_range
        self.z_bearing = wrapToPi(z_bearing)

    def __str__(self):
        return "SensorMeasurement(landmark_id = {0}, z_range = {1} m, z_bearing = {2} rad)".format(
            self.landmark_id, self.z_range, self.z_bearing
        )


class TimeStep(object):
    """Represents a data point consisting of an odometry command and a list of sensor measurements.

    - odom: Odometry measurement
    - sensor: List of landmark observations of type SensorMeasurement
    """

    def __init__(self, odom, sensor):
        self.odom = odom
        self.sensor = sensor

    def __str__(self):
        return "TimeStep(odom = {0}, sensor = {1})".format(self.odom, self.sensor)


def normalize_weights(particles):
    """Normalize particle weights."""
    log_weights = np.array([p.log_weight for p in particles], dtype=float)
    if log_weights.size == 0 or not np.all(np.isfinite(log_weights)):
        uniform = 1.0 / float(len(particles)) if len(particles) > 0 else 0.0
        for p in particles:
            p.weight = uniform
            p.log_weight = np.log(uniform) if uniform > 0.0 else -np.inf
        return particles

    max_log = np.max(log_weights)
    weights = np.exp(log_weights - max_log)
    sum_weights = float(np.sum(weights))
    if not np.isfinite(sum_weights) or sum_weights <= 0.0:
        uniform = 1.0 / float(len(particles)) if len(particles) > 0 else 0.0
        for p in particles:
            p.weight = uniform
            p.log_weight = np.log(uniform) if uniform > 0.0 else -np.inf
        return particles

    for p, w in zip(particles, weights):
        p.weight = w / sum_weights
        p.log_weight = np.log(max(p.weight, 1e-300))

    return particles


###########################################
#   IO helper functions - Do not modify!  #
###########################################


def read_world_data(filename):
    """Reads the world definition and returns a structure of landmarks.

    filename: path of the file to load
    landmarks: structure containing the parsed information

    Each landmark contains the following information:
    - id : id of the landmark
    - x  : x-coordinate
    - y  : y-coordinate
    - c  : color

    Examples:
    - Obtain x-coordinate of the 5-th landmark
      landmarks(5).x
    """
    landmarks = {}
    with open(filename) as world_file:
        for line in world_file:
            line_s = line.rstrip()  # remove newline character

            line_spl = line_s.split(" ")

            landmarks[int(line_spl[0])] = [
                float(line_spl[1]),
                float(line_spl[2]),
                line_spl[3],
            ]

    return landmarks


def read_sensor_data(filename):
    """Reads the odometry and sensor readings from a file.

    filename: path to the file to parse

    The data is returned as an array of time steps, each time step containing
    odometry data and sensor measurements.

    Usage:
    - access the readings for timestep t:
      data[t]
      this returns a TimeStep object containing the odometry reading and all
      landmark observations, which can be accessed as follows
    - odometry reading at timestep t:
      data[i].odometry
    - sensor reading at timestep i:
      data[i].sensor

    Odometry readings have the following fields:
    - r1 : initial rotation in radians counterclockwise
    - t  : translation in meters
    - r2 : final rotation in radians counterclockwise
    which correspond to the identically labeled variables in the motion
    mode.

    Sensor readings can again be indexed and each of the entries has the
    following fields:
    - landmark_id : id of the observed landmark
    - z_range     : measured range to the landmark in meters
    - z_bearing   : measured angle to the landmark in radians

    Examples:
    - Translational component of the odometry reading at timestep 10
      data[10].odometry.t
    - Measured range to the second landmark observed at timestep 4
      data[4].sensor[1].z_range
    """
    # ADDS RANDOM NOISE TO MEASURMENTS
    p = load_params()
    rot1_error = np.sqrt(p["motion_noise"][0]) * p["noise_odomdata"]
    trans_error = np.sqrt(p["motion_noise"][1]) * p["noise_odomdata"]
    rot2_error = np.sqrt(p["motion_noise"][2]) * p["noise_odomdata"]

    range_error = np.sqrt(p["sensor_noise"][0]) * p["noise_landmarkdata"]
    bearing_error = np.sqrt(p["sensor_noise"][1]) * p["noise_landmarkdata"]

    data = []
    sensor_measurements = []
    first_time = True
    odom = None
    with open(filename) as data_file:
        for line in data_file:
            line_s = line.rstrip()  # remove the new line character
            line_spl = line_s.split(" ")  # split the line
            if line_spl[0] == "ODOMETRY":
                if not first_time:
                    data.append(TimeStep(odom=odom, sensor=sensor_measurements))
                    sensor_measurements = []
                first_time = False
                odom = Odometry(
                    r1=random.gauss(float(line_spl[1]), rot1_error),
                    t=random.gauss(float(line_spl[2]), trans_error),
                    r2=random.gauss(float(line_spl[3]), rot2_error),
                )
            if line_spl[0] == "SENSOR":
                if np.random.rand() < p["landmark_failure"]:
                    print("Failing landmark")
                    continue
                sensor_measurements.append(
                    SensorMeasurement(
                        landmark_id=int(line_spl[1]),
                        z_range=random.gauss(float(line_spl[2]), range_error),
                        z_bearing=random.gauss(float(line_spl[3]), bearing_error),
                    )
                )

    data.append(TimeStep(odom=odom, sensor=sensor_measurements))
    return data


def save_params(data):
    """
    Save additional FastSLAM parameters to a text file in JSON format.
    """
    params = load_params()

    for key, item in data.items():
        params[key] = item

    with open("params.txt", "w") as f:
        json.dump(params, f, indent=4)


def load_params(print_flag=False):
    """
    Load FastSLAM parameters from a JSON text file.
    """
    with open("params.txt", "r") as f:
        params = json.load(f)
    if print_flag:
        print("using params in params.txt:")
        for k, p in params.items():
            print(f"-{k}={p}")
    print("")
    np.random.seed(params["seed"])
    random.seed(params["seed"])
    return params


##################################################
#   Debugging helper functions - Do not modify!  #
##################################################


def shape_check(tag, obj, target_shape):
    if target_shape == ():
        if isinstance(obj, np.generic):
            if obj.shape != target_shape:
                assert False, EMSG(
                    call_function()
                    + "\n"
                    + f"{tag} should be a scalar but is {type(obj)}"
                )
        elif not isinstance(obj, (int, float, complex, bool)):
            assert False, EMSG(
                call_function() + "\n" + f"{tag} should be a scalar but is {type(obj)}!"
            )

    elif isinstance(obj, np.ndarray):
        if obj.shape != target_shape:
            assert False, EMSG(
                call_function()
                + "\n"
                + f"{tag}.shape is {obj.shape} but should be ({target_shape},)!"
            )
    else:
        assert False, EMSG(
            call_function()
            + "\n"
            + f"{tag} should be of shape {target_shape} but is {type(obj)}!"
        )


RED = "\033[1;31m"
RESET = "\033[0m"


def EMSG(msg) -> str:
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
        return f"Wrong outputshape in {frame.f_code.co_name}:"


##############################################
#   Plot helper functions - Do not modify!  #
##############################################


def fast_slam_prediction_correction(particles, data, t):
    """Executes one iteration of the prediction-correction of FastSLAM.

    - t: Frame number of the current frame, starting with 0.

    """
    # Perform the prediction step of the particle filter
    for particle in particles:
        particle.prediction_step(data[t].odom)
    # Perform the correction step of the particle filter
    for particle in particles:
        particle.correction_step(data[t].sensor)

    return particles


def fast_slam_resampling(particles, t, num_iterations_resample, print_flag=True):
    """Executes one iteration of the resampling of FastSLAM.

    - t: Frame number of the current frame, starting with 0.
    - num_iterations_resample: Forced resampling every nth frame.

    """
    # Normalize weights
    particles = normalize_weights(particles)

    if num_iterations_resample == "selective":
        ##STUDENT_CODE: #TODO Q4: Implement Selective Resampling
        n_eff = 1.0 / sum(p.weight**2 for p in particles)
        selective_resample = n_eff < (len(particles) / 2.0)
        ##END_STUDENT_CODE
        if selective_resample:
            if print_flag:
                print("-> Selective Resampling")
            particles = low_variance_resampler(particles)

    elif num_iterations_resample == "off":
        pass

    elif isinstance(num_iterations_resample, int):
        if t % num_iterations_resample == 0:
            if print_flag:
                print("-> Naive Resampling")
            particles = low_variance_resampler(particles)

    return particles


class Plotter(object):
    """Helper class for plotting the current state of the robot and map"""

    def __init__(self, data, particles, landmarks):
        self.data = data
        self.particles = particles
        self.landmarks = landmarks
        self.fig, self.ax = plt.subplots(figsize=(12, 8), dpi=100)

        self.p = load_params()

        # Draw GT Map
        self.draw_gt_map()

        # The following objects are updated during the animation
        # Plot the particles as green dots
        particles_x = [particle.pose[0] for particle in self.particles]
        particles_y = [particle.pose[1] for particle in self.particles]
        self.plot_particles = self.ax.plot(
            particles_x, particles_y, "g.", animated=True, zorder=2
        )[0]

        # draw the best particle as a red circle
        self.plot_best_particle = Ellipse(
            (0.0, 0.0), 0, 0, fill=False, animated=True, color="red", zorder=3
        )
        self.ax.add_patch(self.plot_best_particle)

        # draw the trajectory as estimated by the currently best particle as a red line
        self.plot_trajectory = self.ax.plot(
            [0.0], [0.0], "r-", linewidth=5, animated=True, zorder=1, alpha=0.5
        )[0]

    def draw_gt_map(self):
        """Initializes the figure with the map including grount truth oberservation"""
        # --- Load map ---
        resolution = 0.025  # meters per pixel

        origin = [-16.25, -14, 0.0]  # map origin
        image_file = "gt_map_office.png"

        img = Image.open(image_file)  # grayscale
        map_array = np.array(img, dtype=np.float32)

        # Normalize to 0=free, 1=occupied
        map_array = map_array / 255.0

        # --- Plot map in pixel coordinates ---
        self.ax.imshow(
            map_array,
            cmap="gray",
            origin="upper",
            extent=(
                origin[0],
                origin[0] + map_array.shape[1] * resolution,
                origin[1],
                origin[1] + map_array.shape[0] * resolution,
            ),
        )

        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.title("Map (pixels scaled to world coordinates)")
        plt.grid(False)
        self.ax.set_autoscale_on(False)

        # plot the ground truth landmark positions as black crosses with color circles based on room location
        landmarks_x = [coordinates[0] for _, coordinates in self.landmarks.items()]
        landmarks_y = [coordinates[1] for _, coordinates in self.landmarks.items()]
        landmarks_color = [coordinates[2] for _, coordinates in self.landmarks.items()]

        for x, y, color in zip(landmarks_x, landmarks_y, landmarks_color):
            circle_patch = Circle(
                (x, y), 0.25, edgecolor="black", facecolor=color, alpha=0.5
            )
            self.ax.add_patch(circle_patch)

        self.plot_landmarks = self.ax.plot(
            landmarks_x, landmarks_y, "k+", markersize=10, linewidth=5, animated=True
        )[0]

    def fast_slam_plotter(self, t):
        """Perform filter update for each odometry-observation pair read from the data file."""

        # Run Prediction and Correction step
        self.particles = fast_slam_prediction_correction(self.particles, self.data, t)

        # Generate visualization plots of the current state of the filter
        r = self.plot_state(self.data[t].sensor)

        # Run Resampling step
        self.particles = fast_slam_resampling(
            self.particles, t, self.p["num_iterations_resample"]
        )

        return r

    def plot_state_init(self):
        """Initializes the figure with the elements to be drawn"""
        return [self.plot_landmarks]

    def plot_state(self, sensor_measurements):
        """Visualizes the state of the FastSLAM algorithm.

        The resulting plot displays the following information:
        - map ground truth (black +'s)
        - currently best particle (red)
        - particle set in green
        - current landmark pose estimates (red)
        - visualization of the observations made at this time step (line between Particle and landmark)
        """
        # update particle poses
        particles_x = [particle.pose[0] for particle in self.particles]
        particles_y = [particle.pose[1] for particle in self.particles]
        self.plot_particles.set_data(particles_x, particles_y)

        # determine the currently best particle
        weights = [1 for particle in self.particles]
        index_of_best_particle = 0  # weights.index(max(weights))

        weight_sum = sum(weights)
        mean_pose = sum(1 * particle.pose for particle in self.particles) / weight_sum
        cov_pose = (
            sum(
                1 * np.outer(particle.pose - mean_pose, particle.pose - mean_pose)
                for particle in self.particles
            )
            / weight_sum
        )

        self.plot_best_particle.center = (mean_pose[0].item(), mean_pose[1].item())
        (self.plot_best_particle.width, self.plot_best_particle.height, angle_rad) = (
            self.get_ellipse_params(cov_pose)
        )
        self.plot_best_particle.angle = angle_rad * 180.0 / pi

        # get trajectory points

        trajectory_x_list = [
            sum(1 * particle.trajectory[i][0] for particle in self.particles)
            / weight_sum
            for i in range(0, len(self.particles[0].trajectory))
        ]
        trajectory_y_list = [
            sum(1 * particle.trajectory[i][1] for particle in self.particles)
            / weight_sum
            for i in range(0, len(self.particles[0].trajectory))
        ]
        self.plot_trajectory.set_data(trajectory_x_list, trajectory_y_list)

        plots = [
            self.plot_landmarks,
            self.plot_particles,
            self.plot_best_particle,
            self.plot_trajectory,
        ]

        # draw the estimated landmark locations along with the ellipsoids
        for landmark in self.particles[index_of_best_particle].landmarks:
            if landmark.observed:
                bpx = landmark.mu[0]
                bpy = landmark.mu[1]

                plots.append(
                    self.ax.plot(bpx, bpy, "rx", markersize=5, animated=True)[0]
                )
                [a, b, angle] = self.get_ellipse_params(landmark.sigma)
                angle_degrees = wrapToPi(angle) * 180.0 / pi
                e = Ellipse(
                    xy=(bpx, bpy),
                    width=a,
                    height=b,
                    angle=angle_degrees,
                    fill=True,
                    animated=True,
                )
                self.ax.add_patch(e)
                plots.append(e)

        # draw the observations as lines between the best particle and the observed landmarks
        for measurement in sensor_measurements:
            landmark_x = (
                self.particles[index_of_best_particle]
                .landmarks[measurement.landmark_id]
                .mu[0]
            )
            landmark_y = (
                self.particles[index_of_best_particle]
                .landmarks[measurement.landmark_id]
                .mu[1]
            )
            plots.append(
                plt.plot(
                    (landmark_x, mean_pose[0].item()),
                    (landmark_y, mean_pose[1].item()),
                    "k",
                    linewidth=1,
                    animated=True,
                )[0]
            )

        return plots

    def get_ellipse_params(self, C):
        """Calculates unscaled half axes of the 95% covariance ellipse.

        C: covariance matrix
        alpha: confidence value

        Code from the CAS Robot Navigation Toolbox:
        http://svn.openslam.org/data/svn/cas-rnt/trunk/lib/drawprobellipse.m
        Copyright (C) 2004 CAS-KTH, ASL-EPFL, Kai Arras
        Licensed under the GNU General Public License, version 2
        """

        # Chi square inverse cumulative distribution function for alpha = 0.95
        # and 2 degrees of freedom (see lookup table in statistics textbook)
        CHI_SQUARE_INV_95_2 = 5.99146

        sxx = float(C[0, 0])
        syy = float(C[1, 1])
        sxy = float(C[0, 1])
        a = sqrt(
            0.5 * (sxx + syy + sqrt((sxx - syy) ** 2 + 4.0 * sxy**2))
        )  # always greater
        b = sqrt(
            0.5 * (sxx + syy - sqrt((sxx - syy) ** 2 + 4.0 * sxy**2))
        )  # always smaller

        # Remove imaginary parts in case of neg. definite C
        if not np.isreal(a):
            a = np.real(a)
        if not np.isreal(b):
            b = np.real(b)

        # Scaling in order to reflect specified probability
        a = a * sqrt(CHI_SQUARE_INV_95_2)
        b = b * sqrt(CHI_SQUARE_INV_95_2)

        # Look where the greater half axis belongs to
        if sxx < syy:
            swap = a
            a = b
            b = swap

        # Calculate inclination (numerically stable)
        if sxx != syy:
            angle = 0.5 * atan(2.0 * sxy / (sxx - syy))
        elif sxy == 0.0:
            angle = 0.0  # angle doesn't matter
        elif sxy > 0.0:
            angle = pi / 4.0
        elif sxy < 0.0:
            angle = -pi / 4.0
        return [a, b, angle]


def open_python_console(script_path=None):
    script_path = os.path.abspath("solution.py")
    subprocess.Popen([sys.executable, script_path])


def main():
    # Read world data, i.e. landmarks. The true landmark positions are not given to the FastSlam algorithm
    landmarks_truth = read_world_data("world.data")

    # Read sensor readings, i.e. odometry and range-bearing sensor
    data = read_sensor_data("sensor_data.data")

    p = load_params(True)

    # initialize the particles array
    particles = [
        Particle(
            p["num_landmarks"],
            p["motion_noise"],
            p["sensor_noise"],
            p["robot_init_pos"],
        )
        for _ in range(p["num_particles"])
    ]

    for particle in particles:
        particle.weight = 1.0 / float(p["num_particles"])
        particle.log_weight = np.log(particle.weight)

    # set the axis dimensions
    plotter = Plotter(data, particles, landmarks_truth)

    anim = animation.FuncAnimation(
        plt.gcf(),
        plotter.fast_slam_plotter,
        frames=np.arange(0, len(data)),
        init_func=plotter.plot_state_init,
        interval=0,
        blit=True,
        repeat=False,
    )
    if anim:
        plt.show()


if __name__ == "__main__":
    main()
