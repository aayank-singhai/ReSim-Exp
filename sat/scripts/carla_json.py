
import os
import json
from tqdm import tqdm
import imageio
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import math

def load_json(json_path):
    print("Loading json: {}".format(json_path))
    with open(json_path) as f:
        data = json.load(f)
    return data

def dump_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Dumped to: {}".format(json_path))

def exist_file(file_path):
    return os.path.exists(file_path)

def checked_img_loading(file_path):
    if not exist_file(file_path):
        return False, file_path
    try:
        img = cv2.imread(file_path)
        return True, None
    except:
        return False, file_path

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

def img_path_list_to_video(img_path_list, out_path='test.mp4', fps=10):
    writer = imageio.get_writer(out_path, fps=fps)
    for img_path in img_path_list:
        img = imageio.imread(img_path)
        writer.append_data(img)
    writer.close()

def load_image(path, scale=1.0):
    """
    Helper function to load an image and optionally downscale it.
    """
    img = cv2.imread(path)
    if img is not None and scale != 1.0:
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_size)
    return img

def merge_json_dir(json_dir, merged_json_path, end_identifier):
    # json_files = [
    #     f for f in os.listdir(json_dir) if f.startswith('waymo') and f.endswith(end_identifier)
    # ]

    json_files = [
        f for f in os.listdir(json_dir) if 'filtered' in f and f.endswith(end_identifier)
    ]
    
    print("Total number of json files: ", len(json_files))

    def get_ind(filename):
        ind = filename.split('_')[-1].replace(end_identifier, '')
        return int(ind)
    
    json_files.sort(key=get_ind)
    print("After sorted, the first 5 files are: ", json_files[:5])

    # Initialize an empty list to store the data from each JSON file
    total_cnt = 0

    first_json = os.path.join(json_dir, json_files[0])
    first_json = load_json(first_json)
    meta = first_json['meta']
    # info_annos = first_json['info_annos']
    merged_clips = []
    
    # Read each JSON file and append its data to the merged_data list
    for json_file in tqdm(json_files):
        json_path = os.path.join(json_dir, json_file)
        data = load_json(json_path)
        merged_clips.extend(data["clips"])
        total_cnt += len(data["clips"])

    # merged = dict()
    # merged['meta'] = {
    #     'frame_rate': frame_rate,
    #     'total_counts': total_cnt
    # }
    # merged['annotations'] = merged_data

    merged = dict(
        meta=meta,
        # info_annos=info_annos,
        clips=merged_clips
    )

    # merged['meta']['num_clips'] = total_cnt
    merged['meta']['num_clips_filtered'] = total_cnt

    # assert total_cnt == meta['num_clips'], f"{total_cnt} != {meta['num_clips']}"


    # merged_json_path = os.path.join('/cpfs01/user/yangjiazhi/workspace/Ask-Anything/video_chat/youtube_process', 'full_merged.json')
    # merged_json_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/YouTube_merged.json'
    dump_json(merged, merged_json_path)
    print(f'Merged JSON file saved to: {merged_json_path}')
    print(f'Total counts: {total_cnt}')


def remove_first_frame(json_path):
    data = load_json(json_path)
    clips = data['clips']
    for clip in clips:
        # clip['first_frame'] = clip['first_frame'].split('.')[0][9:] + '.jpg'
        clip["img_seq"] = clip["img_seq"][1:]

    data['clips'] = clips
    json_path = json_path.replace('.json', '_correct_v2.json')
    dump_json(data, json_path)


def detect_broken_images(json_path):
    data = load_json(json_path)
    root = data['meta']['data_root']
    clips = data['clips']
    broken_clips = []
    for clip in tqdm(clips):
        img_seq = clip['img_seq']
        for img in img_seq:
            # if not exist_file(img):
            #     broken_clips.append(clip)
            #     break
            try:
                img = load_image(os.path.join(root, img))
            except:
                print(f"Error loading image: {img}")
                broken_clips.append(os.path.join(root, img))
    print(f"Broken clips: {len(broken_clips)}")
    
    # save broken clips as json file
    broken_json_path = json_path.replace('.json', '_broken.json')
    dump_json(broken_clips, broken_json_path)
    return broken_clips

def detect_broken_images_multiprocess(json_path):
    # using multiprocess to speed up
    data = load_json(json_path)
    root = data['meta']['data_root']
    clips = data['clips']
    broken_clips = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for clip in clips:
            img_seq = clip['img_seq']
            for img in img_seq:
                futures.append(executor.submit(checked_img_loading, os.path.join(root, img)))
        for future in tqdm(as_completed(futures), total=len(futures), desc="Checking images"):
            is_checked, path = future.result()
            try:
                if not is_checked:
                    # broken
                    broken_clips.append(path)
                    print(f"Broken clip: {path}")
            except Exception as e:
                print(f"Error checking image: {e}")
    print(f"Broken clips: {len(broken_clips)}")

    # save broken clips as json file
    broken_json_path = json_path.replace('.json', '_broken.json')
    dump_json(broken_clips, broken_json_path)
    return broken_clips

def add_token_to_clips(json_path):
    data = load_json(json_path)
    clips = data['clips']
    for ind, clip in enumerate(clips):
        clip['token'] = ind
    dump_json(data, json_path.replace('.json', f'_token.json'))

def replace_navsim_traj_with_cala(navsim_json, carla_json):

    carla_data = load_json(carla_json)
    navsim_data =  load_json(navsim_json)

    carla_clips = carla_data['clips']
    navsim_clips = navsim_data['clips']

    # * Select top 200
    N = 200
    carla_clips = carla_clips[: N] 
    navsim_clips = navsim_clips[: N]

    new_navsim_clips = []
    for carla_clip, navsim_clip in zip(carla_clips, navsim_clips):
        if carla_clip['score_penalty'] < 0.8:
            navsim_clip['traj_fut'] = carla_clip['traj_fut']
            
            navsim_clip['score_penalty'] = carla_clip['score_penalty']

            new_navsim_clips.append(navsim_clip)

    navsim_data['clips'] = new_navsim_clips

    out_navsim_json = navsim_json.replace('.json', '_carla-traj.json')
    dump_json(navsim_data, out_navsim_json)


navsim_json = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/carla/navsim_test.json'
carla_json = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/carla/demo_0213_correct_v2.json'
replace_navsim_traj_with_cala(navsim_json, carla_json)
import pdb; pdb.set_trace()

# detect_broken_images("/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/carla/demo_0213_correct_v2.json")
detect_broken_images_multiprocess("/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/carla/demo_0213_correct_v2.json")
# add_token_to_clips("/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/carla/demo_0213_correct_v2.json")
import pdb; pdb.set_trace()


# Use PySceneDetect to detect transition frames in a video clip
import scenedetect
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode
# def detect_scenes(image_sequence):
#     # Create a scene manager
#     scene_manager = SceneManager()
    
#     # Add the ContentDetector (which detects changes in content)
#     scene_manager.add_detector(ContentDetector(threshold=30.0))
    
#     # Convert image sequence to a list of frames
#     scene_list = []
#     prev_frame = None
#     frame_number = 0
    
#     for image_path in image_sequence:
#         frame = cv2.imread(image_path)
#         if frame is None:
#             continue
#         frame_number += 1
#         if prev_frame is not None:
#             scene_manager.process_frame(frame, FrameTimecode(frame_number, 30))
#         prev_frame = frame
    
#     # Get list of detected scenes
#     scene_list = scene_manager.get_scene_list()
    
#     # Convert scene list to frame numbers
#     scene_frames = [(start.get_frames(), end.get_frames()) for start, end in scene_list]
    
#     # Determine if the sequence is consistent (no transition detected)
#     is_consistent = len(scene_frames) <= 1
    
#     return scene_frames, is_consistent
import cv2
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode


def detect_scenes(image_sequence, threshold=30.0):
    # Initialize ContentDetector
    detector = ContentDetector(threshold=threshold)
    
    prev_frame = None
    frame_number = 0
    scene_changes = []


    for image_path in image_sequence:
        frame = load_image(image_path, scale=0.25)

        if frame is None:
            print(f"Warning: Could not load image {image_path}")
            continue  # Skip this image

        frame_number += 1
        # timecode = FrameTimecode(frame_number, fps)

        if prev_frame is not None:
            # Ensure frame is in the correct format before processing
            try:
                if detector.process_frame(frame_number, frame):
                    scene_changes.append(frame_number)
                    return False  # Transition detected
            
            except cv2.error as e:
                print(f"OpenCV error processing {image_path}: {e}")
                continue

        prev_frame = frame

    # Determine if the sequence is consistent (no transition detected)
    # is_consistent = len(scene_changes) == 0

    return True


# Helper function to process a single clip.
def process_clip(clip, meta, out_root, clip_index):
    first_frame = clip['first_frame']
    end_frame = clip['end_frame']
    folder_name = clip['folder_name']
    data_root = meta['data_root']
    
    # Build the list of image paths for the clip.
    frame_list = get_frame_list(first_frame, end_frame)
    frame_list = [os.path.join(data_root, folder_name, frame) for frame in frame_list]
    
    is_consistent = detect_scenes(frame_list, threshold=100.0)   # * Main running
    # is_consistent = detect_scenes(frame_list, threshold=50.0)  # * NOT GOOD, lots of false positive


    if not is_consistent:
        print(f'Clip {clip_index} has transition!')
        # Convert to video if transition is detected.
        video_out_path = os.path.join(out_root, f'{clip_index}.mp4')
        img_path_list_to_video(frame_list, video_out_path)
        return None
    else:
        # Return clip index or the clip metadata if the clip is consistent.
        return clip_index


def traverse_youtube_json(json_path, out_root, n_subset=None, subset_ind=None):
    if n_subset is not None and subset_ind is not None:
        out_root = out_root + f'_sub_{subset_ind}'
    os.makedirs(out_root, exist_ok=True)
    infos = load_json(json_path)
    meta  = infos['meta']
    clips = infos['clips']

    if n_subset is not None and subset_ind is not None:
        print(f"Using subset: {subset_ind}/{n_subset}")
        length_per_subset = math.ceil(len(clips) / n_subset)
        start_ind = subset_ind * length_per_subset
        end_ind = (subset_ind + 1) * length_per_subset
        clips = clips[start_ind:end_ind]
        print(f"Subset length: {len(clips)}")

    # DEBUG = True
    # if DEBUG:
    #     clips = clips[:25]

    consistent_clips_index = []
    
    # for i, clip in enumerate(tqdm(clips)):
    #     first_frame = clip['first_frame']
    #     end_frame   = clip['end_frame']
    #     folder_name = clip['folder_name']
    #     data_root   = meta['data_root']
    #     frame_list = get_frame_list(first_frame, end_frame)
    #     frame_list = [os.path.join(data_root, folder_name, frame) for frame in frame_list]
    #     is_consistent = detect_scenes(frame_list)
    #     if not is_consistent:
    #         print(f'Clip {i} has transition!')
            # img_path_list_to_video(frame_list, os.path.join(out_root, f'{i}.mp4'))
    # with ThreadPoolExecutor(max_workers=8) as executor:
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for i, clip in enumerate(clips):
            futures.append(executor.submit(process_clip, clip, meta, out_root, i))
        
        # Optionally, wait for all to complete and handle exceptions.
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing clips"):
            try:
                result = future.result()
                # If result is not None, the clip is consistent.
                if result is not None:
                    consistent_clips_index.append(result)
                    # print("Consistent clips:", consistent_clips_index)
            except Exception as e:
                print(f"Error processing a clip: {e}")

    # sort index?
    print("Sorting consistent clips index...")
    consistent_clips_index.sort()
    print("Finished sorting consistent clips index.")
    consistent_clips = [clips[i] for i in consistent_clips_index]
    infos['clips'] = consistent_clips  # * Filtered clips
    out_json_path = json_path.replace('.json', f'_filtered_{subset_ind}.json')
    dump_json(infos, out_json_path)


# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_subset', type=int, default=None)
parser.add_argument('--subset_ind', type=int, default=None)
args = parser.parse_args()

# json_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/YouTube_svd_val_1080p_clip-len-49_interval-10_77k_flow.json'  # ! DEBUG
# json_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/YouTube_svd_clip-len-49_interval-10_5M_flow_round2.json'
json_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/scene_cut/YouTube_svd_clip-len-49_interval-10_5M_flow_round2.json'  # * MAIN

# out_root = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/tmp/scenecut'
out_root = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/tmp/scenecut3'
# out_root = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/tmp/scenecut_val'  # ! DEBUG

traverse_youtube_json(json_path, out_root, n_subset = args.n_subset, subset_ind=args.subset_ind)