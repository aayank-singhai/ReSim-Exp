import json

# Automatically generate a list of 49 frame filenames
frames = [f"frame_{str(i).zfill(4)}.jpg" for i in range(45, 94)]

data = {
  "meta": {
    # CHANGE THIS to your actual absolute path where the video_01 folder lives
    "data_root": "/home/aic7kor/Desktop/Bosch/ReSim/dataset/images/1ce86e13-03cf-4cb5-99b2-a65400650117.camera_front_wide_120fov" 
  },
  "clips": [
    {
      "img_seq": frames,
      "cmd": "Moving_Forward",
      "traj_fut": [
         [0.0, 0.0, 0.0], 
         [1.0, 0.0, 0.0], 
         [2.0, 0.0, 0.0], 
         [3.0, 0.0, 0.0], 
         [4.0, 0.0, 0.0], 
         [5.0, 0.0, 0.0], 
         [6.0, 0.0, 0.0], 
         [7.0, 0.0, 0.0]
      ],
       "token": "nvidia-future-straight"
    }
  ]
}

# [0.0, 0.0, 0.0], 
#          [1.0, 0.2, 0.1], 
#          [2.0, 0.5, 0.2], 
#          [3.0, 1.0, 0.3], 
#          [4.0, 1.5, 0.4], 
#          [5.0, 2.0, 0.5], 
#          [6.0, 2.5, 0.6], 
#          [7.0, 3.0, 0.7]

# Write to the JSON file expected by the config
with open('nvidia_data.json', 'w') as f:
    json.dump(data, f, indent=2)
    
print("Successfully generated nvidia_data.json with 49 frames!")