import cv2
import numpy as np

def compute_optical_flow_score(video_path, flow_fps=2, flow_map_size=16.0, flow_resize=True):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Parameters for Farnebäck optical flow
    fb_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    
    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video")
        return None
    
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Storage for flow maps
    flow_maps = []

    # Process video frames at 2fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / flow_fps)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **fb_params)
        
        # Downscale flow maps
        if flow_resize:
            height, width = flow.shape[:2]
            scale_factor = flow_map_size / min(height, width)
            flow_resized = cv2.resize(flow, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_LINEAR)
        else:
            flow_resized = flow
        
        # Store the flow map
        flow_maps.append(flow_resized)
        
        prev_gray = gray
    
    cap.release()
    
    # Compute the global motion score
    if flow_maps:
        flow_maps_array = np.array(flow_maps)
        flow_magnitude = np.sqrt(np.square(flow_maps_array[..., 0]) + np.square(flow_maps_array[..., 1]))
        global_motion_score = np.mean(flow_magnitude)
    else:
        global_motion_score = 0.0

    return global_motion_score

# Example usage:
# video_path = 'path_to_your_video.mp4'
# video_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/local_save/nus/gt_0_720x1280.mp4'  # * Score: 0.255496084690094
# video_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/local_save/nus/gt_10000_720x1280.mp4'  # * Score: 6.000
# video_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/local_save/youtube/30hz/gt_YOUTUBE_30hz_35000_720x1280.mp4'  # * Score: 10.428961753845215
# video_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/local_save/youtube/30hz/gt_YOUTUBE_30hz_37000_720x1280.mp4'  # * Score: 9.35198974609375 (Not accurate?, nearly not moving)

video_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/tmp_data/93481c2504474ca0b21d4abbab2cf3cf (online-video-cutter.com).mp4'  # * Score: 1.6543112993240356
score = compute_optical_flow_score(video_path, flow_resize=False)
print(f"Optical Flow Score: {score}")