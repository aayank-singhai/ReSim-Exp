
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
    
    sweeps[scene_token]['sweep'].extend(sweep_clip)

def img_list_to_video_with_imageio(img_list, out_path, fps=12):
    import imageio
    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)  # * macro_block_size=1, compatibility
    for img in img_list:
        writer.append_data(imageio.imread(img))
    writer.close()

def get_available_scenes(nusc):
    """
    Get available scenes from the input nuScenes class.
    Given the raw data, get the information of available scenes for further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the available scenes.
    """

    available_scenes = list()
    print(f"Total scene num: {len(nusc.scene)}")
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from Lyft dataset is absolute path
                lidar_path = lidar_path.split(f"{os.getcwd()}/")[-1]
                # relative path
            if not os.path.exists(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print(f"Exist scene num: {len(available_scenes)}")
    return available_scenes

def split_scenes(nus_version, nusc):
    # split scenes into train, val subset
    from nuscenes.utils import splits
    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert nus_version in available_vers
    if nus_version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif nus_version == "v1.0-test":
        train_scenes = list()
        val_scenes = splits.test
    elif nus_version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("Unknown")

    # filter existing scenes
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]

    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))

    train_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in val_scenes
    ])

    test = "test" in nus_version
    if test:
        print(f"Test scene: {len(val_scenes)}")
    else:
        print(f"Train scene: {len(train_scenes)}")
        print(f"Val scene: {len(val_scenes)}")
    return train_scenes, val_scenes

img_root = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/nuscenes'
# label_root = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/debug/nus_sweeps/labels'
# videos_root = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/debug/nus_sweeps/videos'

train_scenes, val_scenes = split_scenes("v1.0-trainval", dataset)
label_root = {
    'train': '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/nus_sweeps2/train/labels',
    'val': '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/nus_sweeps2/val/labels',
}
videos_root = {
    'train': '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/nus_sweeps2/train/videos',
    'val': '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/nus_sweeps2/val/videos',
}

n_cont = 0
for scene_token in tqdm(sweeps.keys()):
    # save description to txt  # in labels/{scene_token}.txt
    # save sweep_clip to mp4  # in videos/{scene_token}.mp4
    desc = sweeps[scene_token]['desc']
    sweep_clip = sweeps[scene_token]['sweep']
    sweep_clip = [os.path.join(img_root, img) for img in sweep_clip]
    
    if scene_token in train_scenes:
        lb_root = label_root['train']
        vid_root = videos_root['train']
    elif scene_token in val_scenes:
        lb_root = label_root['val']
        vid_root = videos_root['val']
    else:
        print("Continue:", n_cont)
        n_cont += 1
        continue

    with open(os.path.join(lb_root, f"{scene_token}.txt"), 'w') as f:
        f.write(desc)

    img_list_to_video_with_imageio(sweep_clip, os.path.join(vid_root, f"{scene_token}.mp4"), fps=12)