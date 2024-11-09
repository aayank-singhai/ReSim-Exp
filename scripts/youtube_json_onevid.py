import json
import os
from tqdm import tqdm
import imageio
import concurrent.futures
import argparse

# cd xxx/debug
from opencv_optical_flow import compute_optical_flow_score_from_images

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

  # writer = imageio.get_writer('test.mp4', fps=fps)
    # for img in img_list:
    #     writer.append_data(img)
    # writer.close()

# imageio
def img_path_list_to_video(img_path_list, out_path='test.mp4', fps=10):
    writer = imageio.get_writer(out_path, fps=fps)
    for img_path in img_path_list:
        img = imageio.imread(img_path)
        writer.append_data(img)
    writer.close()

def debug_visualize_one_clip(json_path=None, infos=None, out_root="", select_ind=-1):
    # SELECT_INDEX = 10
    # SELECT_INDEX = -1
    # SELECT_INDEX = -1
    # json_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/YouTube_svd.json'
    # json_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/YouTube_svd_clip-len-49_interval-10.json'
    if infos is None:
        infos = load_json(json_path)
    data_root = infos['meta']['data_root']
    clips = infos['clips']
    vis_clip = clips[select_ind]
    flow_dir = vis_clip.get('flow_direction', None)

    folder_name = vis_clip['folder_name']
    vis_clip = get_frame_list(vis_clip['first_frame'], vis_clip['end_frame'])
    vis_clip = [
        os.path.join(data_root, folder_name, name) for name in vis_clip
    ]
    if flow_dir:
        out_path = os.path.join(out_root, f"test_{select_ind}_{flow_dir}.mp4")
    else:
        out_path = os.path.join(out_root, f"test_{select_ind}.mp4")
        
    img_path_list_to_video(vis_clip, out_path=out_path, fps=10)
    return vis_clip

# base = 100000
# debug_visualize_one_clip(base * 10)
# debug_visualize_one_clip(base * 100) 
# # debug_visualize_one_clip(base + 2048)
# # debug_visualize_one_clip(base +10000)
# import pdb; pdb.set_trace()

def check_file_existence(cur_ind, format_length, ext_str, DATA_ROOT, folder_name):
    cur_name = str(cur_ind).zfill(format_length) + '.' + ext_str
    cur_path = os.path.join(DATA_ROOT, folder_name, cur_name)
    if not exist_file(cur_path):
        print("Not exist: {}".format(cur_path))
        return False
    return True

def process_clips(start_ind, clip_length, format_length, ext_str, DATA_ROOT, folder_name):
    successful_clip = True
    indices = range(start_ind, start_ind + clip_length)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(
            check_file_existence,
            indices,
            [format_length] * clip_length,
            [ext_str] * clip_length,
            [DATA_ROOT] * clip_length,
            [folder_name] * clip_length
        ))

    if not all(results):
        successful_clip = False

    return successful_clip

def init_raw_youtube_json(img_folder, save_json_folder):
    DATA_ROOT = '/cpfs01/shared/opendrivelab/GenAD_Datasets/YouTube/'
    infos = []

    # img_folder: /cpfs01/shared/opendrivelab/GenAD_Datasets/YouTube/V1/1080p_val_images
    for author in tqdm(os.listdir(img_folder)):
        # ['KenoVelicanstveni', 'Pete_Drives_USA', 'Driving_Experience']
        author_folder = os.path.join(img_folder, author)
        for video in os.listdir(author_folder):
            video_folder = os.path.join(author_folder, video)

            images = os.listdir(video_folder)
            images = sorted(images, key=lambda x: int(x.split('.')[0]))
            
            for image_path in images:
                sample = {
                    'folder_name': video_folder.replace(DATA_ROOT, ''),
                    'first_frame': image_path
                }

                infos.append(sample)

    dump_json(infos, os.path.join(save_json_folder, 'YouTube_val_raw.json'))

# init_raw_youtube_json(img_folder='/cpfs01/shared/opendrivelab/GenAD_Datasets/YouTube/V1/1080p_val_images/', 
#                       save_json_folder='/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json')

# print("Successful!")
# import pdb; pdb.set_trace()

def create_youtube_json(clip_length=49, is_train=True, is_debug=False, interval=10):
    # interval = 10  # * 1s
    # new_val: use 1080p val set (newly downloaded)

    DATA_ROOT = '/cpfs01/shared/opendrivelab/GenAD_Datasets/YouTube/'

    if is_train:
        json_path = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/gaoshenyuan/YouTube_svd.json'  # * Train
    else:
        # json_path = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/gaoshenyuan/YouTube_svd_val.json'  # * Val
        json_path = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/yangjiazhi/DVGen/data_json/YouTube_svd_val_1080p.json'  # * New Val 1080p
    infos = load_json(json_path)

    if is_debug:
        infos = infos[:1000]

    out_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json'
    
    if is_debug:
        out_path = os.path.join(out_path, 'debug.json')
    else:
        out_path = os.path.join(out_path, os.path.basename(json_path))
        # if new_val:
        #     out_path = out_path.replace("val.json", "val_1080p.json")
    
    clip_infos = dict()
    clip_infos['meta'] = {
        'data_root': DATA_ROOT,
        'clip_length': clip_length,
        'num_interval_frames': interval,
        'fps_clip': 10,
    }
    clip_infos['clips'] = []
    n_incomplete = 0

    for frame_ind, frame in tqdm(enumerate(infos[::interval])):
        folder_name = frame['folder_name']
        # if new_val:
        #     folder_name = folder_name.replace("val_images", "1080p_val_images")

        frame_name = frame['first_frame']
        start_str, ext_str = frame_name.split('.')
        format_length = len(start_str)
        start_ind = int(start_str)

        # * Filtered out incomplete clips
        successful_clip = True

        start_frame_ind = frame_ind * interval
        end_frame_ind = start_frame_ind + clip_length - 1
        if end_frame_ind >= len(infos):
            successful_clip = False
            break
        else:
            end_frame = infos[end_frame_ind]
            end_folder_name = end_frame['folder_name']
            # if new_val:
                # end_folder_name = end_folder_name.replace("val_images", "1080p_val_images")

            end_frame_name = end_frame['first_frame']
            end_str = end_frame_name.split('.')[0]
            end_ind = int(end_str)
            if end_folder_name != folder_name or end_ind != start_ind + clip_length - 1:
                n_incomplete += 1
                successful_clip = False

        if successful_clip:
            end_name = str(start_ind + clip_length - 1).zfill(format_length) + '.' + ext_str
            clip_info = {
                'folder_name': folder_name,
                'first_frame': frame_name,
                'end_frame': end_name,
            }
            clip_infos['clips'].append(clip_info)

    print("len(clip_infos): {}".format(len(clip_infos['clips'])))

    clip_infos['meta']['num_clips'] = len(clip_infos['clips'])
    clip_infos['meta']['num_incomplete'] = n_incomplete

    dump_json(clip_infos, out_path)


# clip_length: 49 or 101
# create_youtube_json(clip_length=101)
create_youtube_json(clip_length=49, is_train=False, is_debug=False, interval=10, new_val=True)
print("successful!")
import pdb; pdb.set_trace()


def merge_json_dir(json_dir, merged_json_path, end_identifier):
    json_files = [
        f for f in os.listdir(json_dir) if f.startswith('YouTube') and f.endswith(end_identifier)
    ]
    
    print("Total number of json files: ", len(json_files))

    def get_ind(filename):
        # full_020_desc_blip2.json
        # return int(filename.split('_')[1])
        ind = filename.replace(end_identifier,'').split('_')[-1]
        return int(ind)
    
    json_files.sort(key=get_ind)
    print("After sorted, the first 5 files are: ", json_files[:5])

    # Initialize an empty list to store the data from each JSON file
    total_cnt = 0

    first_json = os.path.join(json_dir, json_files[0])
    first_json = load_json(first_json)
    meta = first_json['meta']
    info_annos = first_json['info_annos']
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
        info_annos=info_annos,
        clips=merged_clips
    )

    assert total_cnt == meta['num_clips'], f"{total_cnt} != {meta['num_clips']}"
    # merged_json_path = os.path.join('/cpfs01/user/yangjiazhi/workspace/Ask-Anything/video_chat/youtube_process', 'full_merged.json')
    # merged_json_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/YouTube_merged.json'
    dump_json(merged, merged_json_path)
    print(f'Merged JSON file saved to: {merged_json_path}')
    print(f'Total counts: {total_cnt}')

# TODO: 统计数据集各个动作的分布
def traverse_youtube_json(json_path, out_root, random=False):
    os.makedirs(out_root, exist_ok=True)

    infos = load_json(json_path)
    clips = infos['clips']
    n_total = len(clips)

    stats = {
        'Static': 0,
        'Moving_Forward': 0,
        'Turning_Left': 0,
        'Turning_Right': 0,
        'Highly_Static': 0,
    }

    i = 0
    if random:
        import random
        n_sample = 10000
        random_indices = random.sample(range(len(clips)), n_sample)
        for ind in tqdm(random_indices):
            clip = clips[ind]
            if clip['flow_direction'] == "Static":
                debug_visualize_one_clip(infos=infos, out_root=out_root, select_ind=ind)
                i += 1
                if i > 100:
                    break
    else:
        for ind, clip in enumerate(tqdm(clips)):
            stats[clip['flow_direction']] += 1
        print(stats)
        for k, v in stats.items():
            print(k, f"ratio: {v / n_total * 100 :.3f}")
            # if clip['flow_direction'] in ['Turning_Left']:
            #     debug_visualize_one_clip(infos=infos, out_root=out_root, select_ind=ind)


            # # if clip['flow_direction'] == "Static":
            # flow_x, flow_y = clip['flow_score_xy']
            # threshold = 0.01  # * Highly Static, should be removed for training.
            # # threshold = 0.0005  # * Highly Static, should be removed for training.
            # # threshold = 1e-5

            # # threshold = 1e-5 
            # if abs(flow_x) < threshold and abs(flow_y) < threshold:
            #     debug_visualize_one_clip(infos=infos, out_root=out_root, select_ind=ind)
            #     i += 1
            #     if i > 1000:
            #         break


def round2_rectify_direction(json_path, n_split=None, split_ind=None):

    if n_split is not None and split_ind is not None:
        json_path = json_path + f'{split_ind}_of_{n_split}.json'

    infos = load_json(json_path)

    data_root = infos['meta']['data_root']
    clips = infos['clips']

    n_highly_static = 0
    # n_highway = 0
    n_others = 0
    
    for clip_ind, clip in enumerate(tqdm(clips)):
        flow_x, flow_y = clip['flow_score_xy']
        flow_direction = clip['flow_direction']
        high_static_thresh = 0.01  
        # * Highly Static, should be removed for training. Maybe including irrelevant ads.
        
        if abs(flow_x) < high_static_thresh and abs(flow_y) < high_static_thresh:

            # * Rectify the direction
            clips[clip_ind]['flow_direction'] = 'Highly_Static'
            clips[clip_ind]['flow_direction_round1'] = 'Static'
            n_highly_static += 1
        
        elif flow_direction == 'Static':
            # Decide wether it's highway driving
            # recompute flow score with FPS 10
            NUM_FRAMES=25  # fewer frames for speed, less accuracy, return to 25
            FLOW_FPS=10
            folder_name, first_frame, end_frame = clip['folder_name'], clip['first_frame'], clip['end_frame']
            img_list = get_frame_list(first_frame, end_frame)
            img_list = [
                os.path.join(data_root, folder_name, n) for n in img_list
            ]
            flow_magnitude_score, direction, flow_x_score, flow_y_score = compute_optical_flow_score_from_images(
                img_list, 
                video_fps=10, 
                num_frames=NUM_FRAMES, 
                flow_fps=FLOW_FPS, 
                flow_resize=False, 
                img_resize=True, 
                bottom_half=False, 
                bottom_center=True, # * Only use the bottom center part of the flow
            )
            
            # * After checking, seems that using Moving forward is a better choice.
            # * No class of 'Highway' for accuracy.

            if direction != 'Static':
                # if direction == 'Moving_Forward':
                #     # * Highway Driving
                #     direction = 'Highway'  
                #     n_highway += 1
                # else:
                n_others += 1
                clips[clip_ind]['flow_magnitude_score'] = round(float(flow_magnitude_score), 4)
                clips[clip_ind]['flow_score_xy'] = [round(float(flow_x_score), 4), round(float(flow_y_score), 4)]
                clips[clip_ind]['flow_direction'] = direction
                clips[clip_ind]['flow_direction_round1'] = 'Static'

    infos['clips'] = clips

    out_json_path = json_path.replace(".json", "_round2.json")
    dump_json(infos, out_json_path)
    print("Highly Static: ", n_highly_static)
    # print("Highway: ", n_highway)
    print("Others: ", n_others)

if __name__ == '__main__':
    # youtube_json = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/YouTube_svd_clip-len-49_interval-10_5M_flow.json'

    # youtube_json_prefix = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/splits/YouTube_svd_clip-len-49_interval-10_5M_flow_split_'

    # !!! DEBUG
    youtube_json = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/YouTube_svd_clip-len-49_interval-10_5M_flow_round2.json'
    # youtube_json = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/splits/YouTube_svd_clip-len-49_interval-10_5M_flow_split_0_of_20.json'

    RANDOM = False
    out_root = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/traverse_youtube/ROUND2_Statics'
    traverse_youtube_json(youtube_json, out_root=out_root, random=RANDOM)

    # merge_json_dir('/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/splits', '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/YouTube_svd_clip-len-49_interval-10_5M_flow_round2.json', end_identifier='_of_20_round2.json')

    # parser = argparse.ArgumentParser()
    # parser.add_argument("n_split", type=int)
    # parser.add_argument("split_ind", type=int)
    # args = parser.parse_args()
    # round2_rectify_direction(youtube_json_prefix, n_split=args.n_split, split_ind=args.split_ind)
    # print("Successful")