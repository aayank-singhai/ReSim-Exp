import cv2
import numpy as np
import json
import os
import imageio
import random

def load_json(json_path):
    print("Loading json: {}".format(json_path))
    with open(json_path) as f:
        data = json.load(f)
    return data

def dump_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Dumped to: {}".format(json_path))

def img_path_list_to_video(img_path_list, out_path='test.mp4', fps=10):
    writer = imageio.get_writer(out_path, fps=fps)
    for img_path in img_path_list:
        img = imageio.imread(img_path)
        writer.append_data(img)
    writer.close()

def get_frame_list(start_frame, end_frame):
    # eg:
    # start_frame: "000000000.jpg"
    # end_frame: "000000039.jpg"
    # return: ["000000000.jpg", "000000001.jpg", ..., "000000039.jpg"]
    ext = start_frame.split('.')[-1]
    len_num = len(start_frame.split('.')[0])
    start_frame = int(start_frame.split('.')[0])
    end_frame = int(end_frame.split('.')[0])
    frame_list = []
    for i in range(start_frame, end_frame + 1):
        frame_list.append(str(i).zfill(len_num) + f'.{ext}')
    return frame_list


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


def compute_optical_flow_score_from_images(img_path_list, video_fps=10, flow_fps=2, flow_map_size=16.0, flow_resize=True):
    # downsample the frequency for efficiency
    interval = video_fps // flow_fps
    img_path_list = img_path_list[::interval]
    
    # Parameters for Farnebäck optical flow
    fb_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    
    # Read the first image
    prev_frame = cv2.imread(img_path_list[0])
    if prev_frame is None:
        print("Error reading first image")
        return None
    
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Storage for flow maps
    flow_maps = []

    for img_path in img_path_list[1:]:
        frame = cv2.imread(img_path)
        if frame is None:
            print("Error reading image: {}".format(img_path))
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


def test_video():
    video_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/tmp_data/93481c2504474ca0b21d4abbab2cf3cf (online-video-cutter.com).mp4'  # * Score: 1.6543112993240356
    score = compute_optical_flow_score(video_path, flow_resize=False)
    print(f"Optical Flow Score: {score}")


def test_img_path_list():
    # save_folder = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/tmp_data/test_optical_flow'
    # save_folder = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/tmp_data/test_optical_flow_fps5'
    save_folder = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/tmp_data/test_random_from_all'

    # json_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/debug.json'
    json_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/YouTube_svd_clip-len-49_interval-10_5M.json'
    debug_info = load_json(json_path)
    data_root = debug_info['meta']['data_root']
    clip_infos = debug_info['clips']

    FLOW_FPS = 2
    # * fps 2 and 5 yield similar results, use 2 for efficiency
    
    # !! Only use 10
    # head = 10
    # clip_infos = clip_infos[::head]
    selected_index = [
        random.randint(0, len(clip_infos) - 1) for _ in range(50)
    ]

    # for ind in range(len(clip_infos)):
    for ind in selected_index:
        clip = clip_infos[ind]
        folder_name, first_frame, end_frame = clip['folder_name'], clip['first_frame'], clip['end_frame']
        img_list = get_frame_list(first_frame, end_frame)
        img_list = [
            os.path.join(data_root, folder_name, n) for n in img_list
        ]

        optical_flow_score = compute_optical_flow_score_from_images(img_list, video_fps=10, flow_fps=FLOW_FPS)

        video_path = os.path.join(save_folder, f"flow_{ind}_score_{optical_flow_score:.2f}.mp4")
        img_path_list_to_video(img_list, out_path=video_path, fps=10)

        print("flow score", optical_flow_score)


if __name__ == '__main__':
    test_img_path_list()