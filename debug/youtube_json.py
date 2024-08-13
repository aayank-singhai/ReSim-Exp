import json
import os
from tqdm import tqdm
import imageio
import concurrent.futures

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

def debug_visualize_one_clip(select_ind=-1):
    # SELECT_INDEX = 10
    # SELECT_INDEX = -1
    # SELECT_INDEX = -1
    # json_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/YouTube_svd.json'
    json_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/YouTube_svd_clip-len-49_interval-10.json'
    infos = load_json(json_path)
    data_root = infos['meta']['data_root']
    clips = infos['clips']
    vis_clip = clips[select_ind]
    folder_name = vis_clip['folder_name']
    vis_clip = get_frame_list(vis_clip['first_frame'], vis_clip['end_frame'])
    vis_clip = [
        os.path.join(data_root, folder_name, name) for name in vis_clip
    ]
    img_path_list_to_video(vis_clip, out_path=f"test_{select_ind}.mp4", fps=10)
    # import pdb; pdb.set_trace()
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


DEBUG = False  # * DEBUG This Script

DATA_ROOT = '/cpfs01/shared/opendrivelab/GenAD_Datasets/YouTube/'
json_path = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/gaoshenyuan/YouTube_svd.json'
# val_json = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/gaoshenyuan/YouTube_svd_val.json'
infos = load_json(json_path)

if DEBUG:
    infos = infos[:1000]

out_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json'
if DEBUG:
    out_path = os.path.join(out_path, 'debug.json')
else:
    out_path = os.path.join(out_path, os.path.basename(json_path))
# clip_length = 49  # 4.9s
clip_length = 101  #  10s
interval = 10  # * 1s
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
    frame_name = frame['first_frame']
    start_str, ext_str = frame_name.split('.')
    format_length = len(start_str)
    start_ind = int(start_str)

    # * Filtered out incomplete clips
    successful_clip = True

    start_frame_ind = frame_ind * interval
    end_frame_ind = start_frame_ind + clip_length - 1
    if end_frame_ind >= len(infos):
        # print("excepted_end_ind >= len(infos): {}".format(end_frame_ind))
        successful_clip = False
        break
    else:
        end_frame = infos[end_frame_ind]
        end_folder_name = end_frame['folder_name']
        end_frame_name = end_frame['first_frame']
        end_str = end_frame_name.split('.')[0]
        end_ind = int(end_str)
        if end_folder_name != folder_name or end_ind != start_ind + clip_length - 1:
            n_incomplete += 1
            # print("end_ind != start_ind + clip_length - 1: {}".format(end_ind))
            successful_clip = False
            # import pdb; pdb.set_trace()

    # for cur_ind in range(start_ind, start_ind + clip_length):
    #     cur_name = str(cur_ind).zfill(format_length) + '.' + ext_str
    #     cur_path = os.path.join(DATA_ROOT, folder_name, cur_name)
    #     if not exist_file(cur_path):
    #         print("Not exist: {}".format(cur_path))
    #         successful_clip = False
    #         break
    # successful_clip = process_clips(start_ind, clip_length, format_length, ext_str, DATA_ROOT, folder_name)
    
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
import pdb; pdb.set_trace()