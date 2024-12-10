import os
import json

file = "10hz_YouTube_val_sub.json"
path = "/cpfs01/shared/opendrivelab/GenAD_Datasets/REVISED_ANNOS/dense_anno/"
NUM_FRAMES = 30
KEY_FRMAES = 6
SAVE_PATH = "/cpfs01/shared/opendrivelab/GenAD_Datasets/REVISED_ANNOS/dense_anno/for_eval/youtube"


data = json.load(open(os.path.join(path, file), "r"))

# collect = []
# clip = data[0]["frames"]
# folder = data[0].get("folder", None)
# if len(clip) > NUM_FRAMES:
#     for i in range(NUM_FRAMES, len(clip)):
#         keys = [os.path.join(folder, clip[i-NUM_FRAMES:i][j]) \
#                          for j in range(0, NUM_FRAMES, NUM_FRAMES // KEY_FRMAES)]
#         collect.append(keys)
#     clip = clip[-NUM_FRAMES:]

# for datum in data:
#     if (datum["frames"][-2] != clip[-1]) or (datum.get("folder", None) != folder):
#         assert NUM_FRAMES == len(clip), "NUM_FRAMES: {} clip len: {}".format(NUM_FRAMES, len(clip))
#         keys = [os.path.join(folder, clip[i]) for i in range(0, NUM_FRAMES, NUM_FRAMES // KEY_FRMAES)]
#         # keys = [clip[i] for i in range(0, NUM_FRAMES, NUM_FRAMES // KEY_FRMAES)]
#         collect.append(keys)
#         clip = datum["frames"]
#         folder = datum.get("folder", None)
#         if len(clip) > NUM_FRAMES:
#             for i in range(NUM_FRAMES, len(clip)):
#                 keys = [os.path.join(folder, clip[i-NUM_FRAMES:i][j]) \
#                          for j in range(0, NUM_FRAMES, NUM_FRAMES // KEY_FRMAES)]
#                 # keys = [clip[i-NUM_FRAMES:i][j] for j in range(0, NUM_FRAMES, NUM_FRAMES // KEY_FRMAES)]
#                 collect.append(keys)
#             clip = clip[-NUM_FRAMES:]
#     elif len(clip) < NUM_FRAMES:
#         clip.append(datum["frames"][-1])
#     else:
#         assert NUM_FRAMES == len(clip), "NUM_FRAMES: {} clip len: {}".format(NUM_FRAMES, len(clip))
#         keys = [os.path.join(folder, clip[i]) for i in range(0, NUM_FRAMES, NUM_FRAMES // KEY_FRMAES)]
#         # keys = [clip[i] for i in range(0, NUM_FRAMES, NUM_FRAMES // KEY_FRMAES)]
#         collect.append(keys)
#         clip = clip[-NUM_FRAMES:] + [datum["frames"][-1]]
        
# os.makedirs(SAVE_PATH, exist_ok=True)
# with open(os.path.join(SAVE_PATH, "{}_v_{}.json".format(file.split(".")[0].split("hz_")[-1].lower(), KEY_FRMAES)), "w") as f:
#     json.dump(collect, f, indent=4)

# clips = []
frames = set()
for datum in data:
    # if (len(clips) == 0) or (clips[-1][-7] != datum["frames"][0]):
    #     clips.append(datum["frames"])
    # else:
    #     clips[-1].extend(datum["frames"][1:])

    for frm in datum["frames"]:
        frames.add(os.path.join(datum.get("folder", ""), frm))

# with open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/GenAD_Proj/eval/dataset_source/nuscenes_val_v_clip.json", "w") as f:
#     json.dump(clips, f, indent=4)

with open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/GenAD_Proj/eval/dataset_source/youtube/youtube_val_sub_i.json", "w") as f:
    frames = list(frames)
    frames.sort()
    json.dump(frames, f, indent=4)