import matplotlib.pyplot as plt
import numpy as np
import ast
import matplotlib.cm as cm

def visualize_trajectory_bev_single(traj, out_path='bev.png'):

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
    
    # plt.xlabel("Left / Right (meters)")
    # plt.ylabel("Forward (meters)")
    # plt.title("Bird's-Eye View of Ego Trajectory")
    
    plt.grid(True)
    # plt.legend()
    # plt.show()
    # plt.savefig('trajectory2.png')
    # out_path = filepath.replace('text.txt', 'bev.png')


    plt.savefig(out_path)



def visualize_trajectory_bev_2(traj, traj2=None, out_path='bev.png'):

    plt.figure(figsize=(5, 4))
    plt.rcParams['legend.fontsize'] = 15

    def plot_traj(traj, color='orange', zorder=1, markersize=8, label="GT Traj."):
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
            colors = cm.Oranges(np.linspace(0, 1, num_points))  # Reverse to get darkening
            # colors = cm.plasma(np.linspace(0, 1, num_points))  # Reverse to get darkening

        elif color == 'blue':
            colors = cm.Blues(np.linspace(0, 1, num_points))

        elif color == 'green':
            colors = cm.Greens(np.linspace(0, 1, num_points))

        elif color == 'purples':
            colors = cm.Purples(np.linspace(0, 1, num_points))

        elif color == 'reds':
            colors = cm.Reds(np.linspace(0, 1, num_points))

        if label=="GT Traj":
            alpha = 0.5
            # alpha = 1

            # alpha = 0.6
        else:
            alpha = 0.5
            # alpha = 1
            # alpha = 0.6

            # alpha = 0.8
        
        for i in range(num_points):
            if i == len(trajectory) - 1:
              # plt.plot(lateral[i], forward[i], marker='o', color=colors[i], linewidth=2, alpha=0.6, markersize=markersize, zorder=zorder, label=label)
              # plt.plot(lateral[i], forward[i], marker='o', color=colors[i], alpha=0.6, markersize=markersize, zorder=zorder, label=label)
              plt.plot(lateral[i], forward[i], marker='o', color=colors[i], alpha=alpha, markersize=markersize, zorder=zorder, label=label)
              
                
            else:
              # plt.plot(lateral[i], forward[i], marker='o', color=colors[i], linewidth=2, alpha=0.6, markersize=markersize, zorder=zorder)
              # plt.plot(lateral[i], forward[i], marker='o', color=colors[i], alpha=0.6, markersize=markersize, zorder=zorder)
              plt.plot(lateral[i], forward[i], marker='o', color=colors[i], alpha=alpha, markersize=markersize, zorder=zorder)


    
    plot_traj(traj, color='green', markersize=8, label="GT Traj")
    # plot_traj(traj, color='purples', markersize=8, label="GT Traj")
    # plot_traj(traj, color='reds', markersize=8, label="GT Traj")

    if traj2 is not None:
        plot_traj(traj2, color='purples', zorder=1, markersize=8, label="Pred Traj")
        # plot_traj(traj2, color='reds', zorder=1, markersize=8, label="Pred Traj")


    # plt.xlim(-15, 15)
    plt.xlim(-7.5, 7.5)
    # plt.xlim(-10, 10)
    
    # plt.ylim(0, 50)
    plt.ylim(0, 25)
    # plt.ylim(0, 20)

    # plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_aspect(0.8)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    # plt.xlabel("Left / Right (meters)")
    # plt.ylabel("Forward (meters)")
    
    # plt.xlabel("Left / Right (meters)")
    # plt.ylabel("Forward (meters)")
    # plt.title("Bird's-Eye View of Ego Trajectory")
    
    plt.grid(True)
    # plt.legend(loc='upper right')

    # plt.legend(fontsize=100)
    leg = plt.legend(loc='upper right')  # Create the legend
    for handle in leg.legendHandles:
        handle.set_linestyle('none')
    
    plt.savefig(out_path)


# traj_txt = '/Users/shawn_yang/MySpace/Projects/GenADv3/Demos/carla_improves_action_control/youtube_no_carla_8/text.txt'
# traj_txt = '/Users/shawn_yang/MySpace/Projects/GenADv3/Demos/carla_improves_action_control/youtube_with_carla_338/text.txt'


# Example usage
# visualize_trajectory_bev_single(traj_txt)
    
# token

gt_traj = [
        [
          0.29627945765140845,
          -0.00043809496754478267,
          7.660478541504645e-05
        ],
        [
          0.7781084144726867,
          -0.0005298968399733303,
          -0.00014058455391730004
        ],
        [
          1.3924248510057102,
          -0.004121026786565432,
          -0.0010248790298019728
        ],
        [
          2.208172715588381,
          -0.009924115908664677,
          -0.002348204850599167
        ],
        [
          3.2862216276559164,
          -0.016256960553632162,
          -0.0039804548616539925
        ],
        [
          4.648977681002271,
          -0.026329219904006936,
          -0.005941028197253838
        ],
        [
          6.267787860839283,
          -0.03247022845577936,
          -0.0022124362304349887
        ],
        [
          8.132756589263005,
          -0.033519445260329184,
          0.00012736844390515145
        ],
        [
          10.220202219365438,
          -0.032156231250044125,
          0.007047851257307779
        ],
        [
          12.542838509281877,
          -0.024240157185055372,
          0.010376648494013985
        ]
      ]

pred_traj = [
        [
          0.2839398980140686,
          0.00029530562460422516,
          0.00015318508667405695
        ],
        [
          0.6869722604751587,
          -0.0020558759570121765,
          -0.003110572462901473
        ],
        [
          1.2008954286575317,
          -0.001501921797171235,
          -0.007968024350702763
        ],
        [
          1.860300898551941,
          -0.0028484612703323364,
          -0.0028569407295435667
        ],
        [
          2.649775505065918,
          -0.009266933426260948,
          -0.004160849843174219
        ],
        [
          3.600506544113159,
          -0.013523752801120281,
          -0.01068265549838543
        ],
        [
          4.68917179107666,
          -0.02256081812083721,
          -0.008806421421468258
        ],
        [
          5.910465717315674,
          -0.048460882157087326,
          -0.012458103708922863
        ]
      ]

# gt_traj = [
#         [
#           0.8796292646449247,
#           0.018564575345866387,
#           0.04211778498237394
#         ],
#         [
#           1.4862697272806564,
#           0.054036082518851884,
#           0.0642389007221702
#         ],
#         [
#           1.7932979843025196,
#           0.0808823485543015,
#           0.07444559798252648
#         ],
#         [
#           1.8429278796277269,
#           0.09705313931167374,
#           0.07522158825345038
#         ],
#         [
#           1.816318172416481,
#           0.09777108981252856,
#           0.07548488113856289
#         ],
#         [
#           1.7979972829809265,
#           0.09701157998873312,
#           0.07557621701186569
#         ],
#         [
#           1.7877185740602053,
#           0.09663029307030255,
#           0.07558184920635115
#         ],
#         [
#           1.781896807960093,
#           0.09642265777546463,
#           0.0755258899935747
#         ],
#         [
#           1.7780079344354294,
#           0.09637514903841372,
#           0.07558996225937077
#         ],
#         [
#           1.7753469149282275,
#           0.09633986188775348,
#           0.0755872372037163
#         ]
#       ]
# pred_traj = [
#         [
#           0.9219192266464233,
#           0.025059053674340248,
#           0.05001698434352875
#         ],
#         [
#           1.81771981716156,
#           0.09430920332670212,
#           0.09744518250226974
#         ],
#         [
#           2.6164677143096924,
#           0.18533554673194885,
#           0.12127994000911713
#         ],
#         [
#           3.2815141677856445,
#           0.2826230227947235,
#           0.16066280007362366
#         ],
#         [
#           3.7279884815216064,
#           0.3496461808681488,
#           0.1915311962366104
#         ],
#         [
#           3.967740297317505,
#           0.39511680603027344,
#           0.19876687228679657
#         ],
#         [
#           4.046157360076904,
#           0.40678641200065613,
#           0.19647735357284546
#         ],
#         [
#           4.056614398956299,
#           0.4205471873283386,
#           0.1926029473543167
#         ]
#       ]

visualize_trajectory_bev_2(gt_traj, pred_traj, out_path='/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/scripts/bev.png')