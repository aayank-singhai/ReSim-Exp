
# TODO: Split train-val

from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import os

nusc = NuScenes(version='v1.0-trainval', dataroot='/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/nuscenes', verbose=True)
dataset = nusc


sweeps = dict()  # scene_token -> sweep_clip
samples = dataset.sample
samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))

sensor = 'CAM_FRONT'
# num_frames = 49

scene_token = samples[0]["scene_token"]

def get_sensor_from_sensor_token(sensor_token):
    return dataset.get('sample_data', sensor_token)

def get_sample_token_from_sensor_token(sensor_token):
    return dataset.get('sample_data', sensor_token)['sample_token']

def get_scene_token_from_sensor_token(sensor_token):
    # sample_token = dataset.get('sample_data', sensor_token)['sample_token']
    sample_token = get_sample_token_from_sensor_token(sensor_token)
    return dataset.get('sample', sample_token)['scene_token']

# * Use prev, not next
for sample_ind, sample in enumerate(samples):
    
    sweep_clip = []
    sample_token = sample['token']
    scene_token = sample['scene_token']

    if scene_token not in sweeps:
        sweeps[scene_token] = dict()
        sweeps[scene_token]['sweep'] = []
        sweeps[scene_token]['desc'] = dataset.get('scene', scene_token)['description']
    
    cur_cam_token = dataset.get('sample_data', sample['data'][sensor])['token']
    cur_sample_token = get_sample_token_from_sensor_token(cur_cam_token)

    indi = 0
    while cur_sample_token == sample_token:
        cur_sensor = get_sensor_from_sensor_token(cur_cam_token)
        indi += 1
        # sweep_clip.append(
        #     cur_sensor['filename']
        # )
        sweep_clip.insert(
            0,
            cur_sensor['filename']
        )

        cur_cam_token = cur_sensor['prev']
        if cur_cam_token == '':
            break
        cur_sample_token = get_sample_token_from_sensor_token(cur_cam_token)
    
    # sweeps.append(sweep_clip)
    sweeps[scene_token]['sweep'].extend(sweep_clip)

def img_list_to_video_with_imageio(img_list, out_path, fps=12):
    import imageio
    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)  # * macro_block_size=1, compatibility
    for img in img_list:
        writer.append_data(imageio.imread(img))
    writer.close()

img_root = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/nuscenes'
label_root = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/debug/nus_sweeps/labels'
videos_root = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/debug/nus_sweeps/videos'
for scene_token in tqdm(sweeps.keys()):
    # save description to txt  # in labels/{scene_token}.txt
    # save sweep_clip to mp4  # in videos/{scene_token}.mp4
    desc = sweeps[scene_token]['desc']
    sweep_clip = sweeps[scene_token]['sweep']
    sweep_clip = [os.path.join(img_root, img) for img in sweep_clip]

    with open(os.path.join(label_root, f"{scene_token}.txt"), 'w') as f:
        f.write(desc)

    img_list_to_video_with_imageio(sweep_clip, os.path.join(videos_root, f"{scene_token}.mp4"), fps=12)