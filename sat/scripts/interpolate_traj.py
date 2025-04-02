import json
import yaml
from tqdm import tqdm
import os
import cv2
import imageio
import math
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

import matplotlib.pyplot as plt
import ast
import matplotlib.cm as cm

def load_json(json_path):
    print("Loading json: {}".format(json_path))
    with open(json_path) as f:
        data = json.load(f)
    return data

def dump_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Dumped to: {}".format(json_path))


def parse_yaml(yaml_path):
    print("Parsing yaml: {}".format(yaml_path))
    with open(yaml_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

    
    clips = []
    num_clips = 0
    for token, sample in tqdm(data.items()):
        sample['lidar_pc_token'] = token
        sample['cmd'] = cmd_to_action[sample['cmd']]

        log_name = sample['log_name']
        if log_name not in logs:
            continue
        clips.append(sample)
        num_clips += 1
    out['clips'] = clips
    out['meta']['num_clips'] = num_clips
    return out

def count_actions(data):
    clips = data['clips']
    actions = {}
    for clip in clips:
        action = clip['cmd']
        if action not in actions:
            actions[action] = 0
        actions[action] += 1
    print(actions)
    return actions

def video_to_images(vid_path, frame_rate):
    # Create an output folder based on the video file name
    output_folder = os.path.dirname(vid_path)
    output_folder = os.path.join(output_folder, "frames")
    os.makedirs(output_folder, exist_ok=True)
    image_paths = []

    # Open the video file
    cap = cv2.VideoCapture(vid_path)
    original_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame based on the specified frame rate
        if frame_count % int(original_frame_rate // frame_rate) == 0:
            image_path = os.path.join(output_folder, f"{saved_frame_count:09d}.jpg")
            cv2.imwrite(image_path, frame)
            image_paths.append(image_path)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_frame_count} frames from {vid_path}")

    return image_paths


def visualize_trajectory_bev(traj, traj2=None, out_path='bev.png'):

    plt.figure(figsize=(5, 4))

    def plot_traj(traj, color='orange', zorder=1, markersize=8):
        if isinstance(traj, str):
            filepath = traj
            with open(filepath, 'r') as f:
                lines = f.readlines()

            # Parse only x and y (ignore heading)
            trajectory = [ast.literal_eval(line.strip())[:2] for line in lines[1:] if line.strip().startswith('[')]
        
        elif isinstance(traj, list):
            trajectory = traj

        trajectory = [[0, 0, 0]] + trajectory

        # Remap: x (forward) -> y-axis, y (lateral) -> x-axis
        forward = [pt[0] for pt in trajectory]
        lateral = [-1 * pt[1] for pt in trajectory]
        
        num_points = len(trajectory)

        # plt.figure(figsize=(5, 4))
        
        # plt.plot(lateral, forward, marker='o', color='blue', linewidth=2, alpha=0.8, label="Trajectory")
        if color == 'orange':
            # colors = cm.Oranges(np.linspace(0, 1, num_points))  # Reverse to get darkening
            colors = cm.plasma(np.linspace(0, 1, num_points))  # Reverse to get darkening

        elif color == 'blue':
            colors = cm.Blues(np.linspace(0, 1, num_points))
        
        for i in range(num_points):
            plt.plot(lateral[i], forward[i], marker='o', color=colors[i], linewidth=2, alpha=0.8, markersize=markersize, zorder=zorder)

    
    plot_traj(traj, markersize=6)
    if traj2 is not None:
        plot_traj(traj2, color='blue', zorder=2, markersize=10)

    plt.xlim(-15, 15)
    
    
    plt.ylim(0, 50)
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_aspect(0.8)
    
    # plt.xlabel("Left / Right (meters)")
    # plt.ylabel("Forward (meters)")
    
    plt.xlabel("Left / Right (meters)")
    plt.ylabel("Forward (meters)")
    # plt.title("Bird's-Eye View of Ego Trajectory")
    
    plt.grid(True)
    
    plt.savefig(out_path)


def euclidean_distance(p1, p2):
    """Compute Euclidean distance between two 2D points (ignoring heading)."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def detect_static_region(traj, distance_threshold=0.1, min_static_points=3):
    """
    Detect the starting index of a static region.
    
    A static region is defined as at least `min_static_points` consecutive waypoints 
    where the distance between each consecutive pair is below `distance_threshold`.
    
    Returns the index where the static behavior begins.
    If no static region is detected, returns None.
    """
    n = len(traj)
    for i in range(1, n - min_static_points + 1):
        # Check the previous min_static_points-1 differences from index i-1 to i+min_static_points-2
        static = True
        for j in range(i, i + min_static_points - 1):
            if euclidean_distance(traj[j-1], traj[j]) >= distance_threshold:
                static = False
                break
        if static:
            return i - 1  # k is the last dynamic index (start of static region)
    return None


def compute_arc_length(traj_2d):
    """
    Compute the cumulative arc-length (distance) along the 2D trajectory.
    Returns an array of the same length as traj_2d.
    """
    s = [0.0]
    for i in range(1, len(traj_2d)):
        ds = euclidean_distance(traj_2d[i-1], traj_2d[i])
        s.append(s[-1] + ds)
    return np.array(s)

def extrapolate_trajectory(traj, distance_threshold=0.1, min_static_points=3):
    """
    Given a trajectory (list of [x, y, heading]) that may include a near-static region
    (due to collision) at the end, this function replaces the static tail with an extrapolated
    trajectory. The extrapolation is based on fitting cubic splines (with respect to arc-length)
    to the dynamic (non-collision) portion. The extra arc-length for the static tail is computed
    based on the last observed dynamic increment. The output has the same number of waypoints.
    
    Returns:
      new_traj: the extrapolated trajectory (or original if no static region is detected)
      need_extrapolation: a boolean flag (True if static region was detected and extrapolation applied)
    """
    N = len(traj)
    # Detect static region (returns index of last dynamic waypoint)
    static_idx = detect_static_region(traj, distance_threshold, min_static_points)
    if static_idx is None or static_idx >= N - 1:
        # No static region or no tail to replace.
        return traj, False

    # Use dynamic portion up to and including the last dynamic waypoint.
    dynamic_traj = traj[:static_idx+1]
    if len(dynamic_traj) < 3:
        return traj, False  # Not enough points for advanced extrapolation.


    # * ONLY USE First 4 waypoints for dynamics
    ONLY_USE_FOUR = True
    if ONLY_USE_FOUR:
        dynamic_traj = dynamic_traj[:4]  # * Only using the first three waypoints for extrapolation
    
    # Use only the first two dimensions (x,y)
    dynamic_2d = [[pt[0], pt[1]] for pt in dynamic_traj]
    s_dynamic = compute_arc_length(dynamic_2d)
    
    # Fit cubic splines for x(s) and y(s)
    # cs_x = CubicSpline(s_dynamic, [pt[0] for pt in dynamic_2d], extrapolate=True)
    # cs_y = CubicSpline(s_dynamic, [pt[1] for pt in dynamic_2d], extrapolate=True)

    cs_x = interp1d(s_dynamic, [pt[0] for pt in dynamic_2d], kind='quadratic', fill_value="extrapolate")
    cs_y = interp1d(s_dynamic, [pt[1] for pt in dynamic_2d], kind='quadratic', fill_value="extrapolate")
    
    
    # Determine the arc-length spacing for the full trajectory.
    # For the dynamic portion, we already have s_dynamic.
    # For the remaining (static) part, we assume that each new step covers the same extra distance
    # as the last dynamic increment.
    last_step = s_dynamic[-1] - s_dynamic[-2]
    extra_steps = N - len(dynamic_traj)
    
    # Create a full arc-length parameter: first dynamic points, then extrapolated values.
    s_full = list(s_dynamic)
    for i in range(extra_steps):
        s_full.append(s_full[-1] + last_step)
    s_full = np.array(s_full)
    
    # Evaluate the spline at the full arc-length values.
    x_full = cs_x(s_full)
    y_full = cs_y(s_full)
    
    # Reconstruct the trajectory: for heading, you might compute from the derivative if desired.
    # Here, we simply keep heading at 0 for simplicity.
    new_traj = [[float(x_full[i]), float(y_full[i]), 0.0] for i in range(1, N)]

    original_traj = traj[1:]
    # print(f"original traj: {original_traj}")

    # print(f"extrapolated traj: {new_traj}")

    # import pdb; pdb.set_trace()
    
    # assert original_traj[:2] == new_traj[:2q]
    
    return new_traj, True

# def interpolate_trajectory(traj, distance_threshold=0.1, min_static_points=3):
#     """
#     Given a trajectory (list of [x, y, heading]) that may include a near-static region at the end,
#     detect the static part and generate an interpolated trajectory (using cubic spline interpolation)
#     that extends the dynamics of the initial dynamic waypoints.
    
#     The number of output waypoints will match the original trajectory.
    
#     Interpolation is performed only if a static region is detected.
#     """
#     N = len(traj)
#     # Detect start of static region using the threshold
#     static_start_idx = detect_static_region(traj, distance_threshold, min_static_points)
    
#     # If no static region is detected, return the original trajectory
#     if static_start_idx is None:
#         return traj
    
#     # Use waypoints up to static_start_idx as the dynamic portion for fitting.
#     dynamic_traj = traj[:static_start_idx]
    
#     # If there are not enough points to fit a spline, return the original trajectory.
#     if len(dynamic_traj) < 3:
#         return traj
    
#     # Parameterize the dynamic trajectory using their indices.
#     t_dynamic = np.arange(len(dynamic_traj))
#     x_dynamic = np.array([pt[0] for pt in dynamic_traj])
#     y_dynamic = np.array([pt[1] for pt in dynamic_traj])
    
#     # Fit cubic splines to the x and y coordinates.
#     cs_x = CubicSpline(t_dynamic, x_dynamic, extrapolate=True)
#     cs_y = CubicSpline(t_dynamic, y_dynamic, extrapolate=True)
    
#     # Create new parameter values for the entire trajectory length.
#     t_new = np.linspace(0, t_dynamic[-1], N)
    
#     # Evaluate the spline at the new parameter values.
#     x_new = cs_x(t_new)
#     y_new = cs_y(t_new)
    
#     # For heading, since it's always 0 in this scenario, we keep it constant.
#     interpolated_traj = [[float(x_new[i]), float(y_new[i]), 0.0] for i in range(N)]
    
#     return interpolated_traj

def extrapolate_traj_for_dataset(json_path):

    data = load_json(json_path)
    clips = data['clips']

    new_clips = []
    for ind_clip, clip in enumerate(tqdm(clips)):
        traj = clip['traj_fut']
        traj = [[0., 0., 0.]] + traj        
        
        is_collision = clip['collision'] == 1.0

        if is_collision:
            new_traj, flag_static_detected= extrapolate_trajectory(traj, distance_threshold=1., min_static_points=3)

            if flag_static_detected:
                clip['extrapolated_traj_fut'] = new_traj

                # * Visualize both traj and interpolated traj
                VISUALIZE = False
                if VISUALIZE:
                    visualize_trajectory_bev(traj, new_traj, out_path=f"/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/carla/v2_reward/tmp/traj_{ind_clip}.png")

        new_clips.append(clip)

    data['clips'] = new_clips

    out_path = json_path.replace(".json", "_interpolate_traj.json")

    dump_json(data, out_path)


# data_json = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/carla/v2_reward/val_carla_0227_24k_append_reward_v2.json'
data_json = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/carla/v2_reward/carla_0227_24k_append_reward_v2_train.json'

extrapolate_traj_for_dataset(data_json)
# video_to_images('/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/tmp/OOD.mp4', frame_rate=10)