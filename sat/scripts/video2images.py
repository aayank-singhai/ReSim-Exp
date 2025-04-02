import os
import cv2



def video2images(video_path):
    # Check if video_path is a valid path
    # using cv2.VideoCapture(video_path)
    if not os.path.exists(video_path):
        print("Invalid video path")
        return
        
    # Create a directory to store images
    image_dir = video_path.split('.')[0]
    os.makedirs(image_dir, exist_ok=True)
    
    # Read the video
    cap = cv2.VideoCapture(video_path)
    
    # Read the video frame by frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{image_dir}/frame_{frame_count}.jpg", frame)
        frame_count += 1
    cap.release()
    print(f"Total frames: {frame_count}")
    
    return frame_count

video = '/Users/shawn_yang/MySpace/Projects/GenADv3/Demos/carla_improves_action_control/youtube_no_carla_338/Sample_folder-338_2570_000000.mp4'

video2images(video)