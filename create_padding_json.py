import json

# 1. The 3 real context frames
frames = ["frame_0116.jpg", "frame_0117.jpg", "frame_0118.jpg"]

# 2. Pad the remaining 46 frames with duplicates of frame 118 
# (The model will completely overwrite these with its generated future)
frames += ["frame_0118.jpg"] * 46

data = {
  "meta": {
    "data_root": "/home/aic7kor/Desktop/Bosch/ReSim/dataset/images/0abe118e-aa79-41f6-a719-f2df8abaf1ea.camera_front_wide_120fov"
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

with open('nvidia_data.json', 'w') as f:
    json.dump(data, f, indent=2)
    
print("Successfully generated padded nvidia_data.json for future prediction!")