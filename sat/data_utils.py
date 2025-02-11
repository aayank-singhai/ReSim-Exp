import io
import os
import torchvision.transforms as TT
import numpy as np
import torch
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import imageio
import json

cmd_to_action = {
    0: "Turning_Left",
    1: "Moving_Forward",
    2: "Turning_Right",
}

def load_json(json_path, verbose=True):
    if verbose:
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


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    # arr: T C H W
    # image_size: H W
    
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    elif reshape_mode == "lower":
        # Crop out the upper part
        top, left = delta_h, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr


def pad_last_frame(tensor, num_frames):
    # T, H, W, C
    if tensor.shape[0] < num_frames:
        last_frame = tensor[-int(num_frames - tensor.shape[1]) :]
        padded_tensor = torch.cat([tensor, last_frame], dim=0)
        return padded_tensor
    else:
        return tensor[:num_frames]
