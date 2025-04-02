import matplotlib.pyplot as plt
import numpy as np
import ast
import matplotlib.cm as cm

def visualize_trajectory_bev(traj, out_path='bev.png'):

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

    plt.figure(figsize=(5, 4))
    
    # plt.plot(lateral, forward, marker='o', color='blue', linewidth=2, alpha=0.8, label="Trajectory")
    colors = cm.Oranges(np.linspace(0, 1, num_points))  # Reverse to get darkening
    
    for i in range(num_points):
        plt.plot(lateral[i], forward[i], marker='o', color=colors[i], linewidth=2, alpha=0.8, markersize=8)
 
    # Draw arrows between waypoints
    for i in range(len(trajectory) - 1):
        start_lat, start_fwd = lateral[i], forward[i]
        end_lat, end_fwd = lateral[i + 1], forward[i + 1]
        
        mid_lat = (start_lat + end_lat) / 2
        mid_fwd = (start_fwd + end_fwd) / 2
        
        dx = end_lat - start_lat
        dy = end_fwd - start_fwd
        # plt.arrow(start_lat, start_fwd, dx, dy, head_width=1, head_length=1, fc='green', ec='green', length_includes_head=True)
        # plt.arrow(mid_lat, mid_fwd, dx, dy, head_width=1, head_length=0.5, fc='green', ec='green', length_includes_head=True)
        

    # plt.xlim(-12.5, 12.5)
    # plt.xlim(-12.5, 12.5)
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
    # plt.legend()
    # plt.show()
    # plt.savefig('trajectory2.png')
    # out_path = filepath.replace('text.txt', 'bev.png')


    plt.savefig(out_path)


# traj_txt = '/Users/shawn_yang/MySpace/Projects/GenADv3/Demos/carla_improves_action_control/youtube_no_carla_8/text.txt'
traj_txt = '/Users/shawn_yang/MySpace/Projects/GenADv3/Demos/carla_improves_action_control/youtube_with_carla_338/text.txt'

# Example usage
visualize_trajectory_bev(traj_txt)