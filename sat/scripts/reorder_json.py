import json

def load_json(json_path):
    print("Loading json: {}".format(json_path))
    with open(json_path) as f:
        data = json.load(f)
    return data

def dump_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Dumped to: {}".format(json_path))


ltf_json = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/navsim/offline_rl/testset_init_latent_transformer/out_2025.02.27.09.16.35_eval_latent_transformer_merged.json'
transfuser_json = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/navsim/offline_rl/testset_init_transfuser/out_2025.02.27.09.47.59_eval_transfuser_merged.json'

ltf = load_json(ltf_json)
transfuser = load_json(transfuser_json)


# order transfuser json according to the sequential order (key: lidar_pc_token) of ltf samples
ltf_clips = ltf["clips"]
transfuser_clips = transfuser["clips"]

ltf_clips_dict = {clip["lidar_pc_token"]: clip for clip in ltf_clips}
transfuser_clips_dict = {clip["lidar_pc_token"]: clip for clip in transfuser_clips}

# * Find shared key
shared_keys = set(ltf_clips_dict.keys()).intersection(set(transfuser_clips_dict.keys()))

print("Number of shared keys: {}".format(len(shared_keys)))  # 4805
import pdb; pdb.set_trace()

# * Only keep shared key for transfuser and ltf
ltf_clips = []
transfuser_clips = []

for key in shared_keys:
    ltf_clips.append(ltf_clips_dict[key])
    transfuser_clips.append(transfuser_clips_dict[key])

# * Check the first 10 keys of transfuser and ltf
for i in range(10):
    print("Lidar pc token {}: {}".format(i, transfuser_clips[i]["lidar_pc_token"]))
    print("Lidar pc token {}: {}".format(i, ltf_clips[i]["lidar_pc_token"]))
    assert transfuser_clips[i]["lidar_pc_token"] == ltf_clips[i]["lidar_pc_token"]

ltf["clips"] = ltf_clips
transfuser["clips"] = transfuser_clips

dump_json(ltf, ltf_json.replace('.json', '_shared.json'))
dump_json(transfuser, transfuser_json.replace('.json', '_shared.json'))



# # reorder transfuser clips according to the order of ltf clips
# transfuser_clips = []
# for clip in ltf_clips:
#     lidar_pc_token = clip["lidar_pc_token"]
#     if lidar_pc_token in transfuser_clips_dict:
#         transfuser_clips.append(transfuser_clips_dict[lidar_pc_token])
#     else:
#         print("Lidar pc token {} not found in transfuser clips".format(lidar_pc_token))

# transfuser["clips"] = transfuser_clips

# # * Check the first 10 keys of transfuser and ltf
# for i in range(10):
#     print("Lidar pc token {}: {}".format(i, transfuser["clips"][i]["lidar_pc_token"]))
#     print("Lidar pc token {}: {}".format(i, ltf["clips"][i]["lidar_pc_token"]))
#     assert transfuser["clips"][i]["lidar_pc_token"] == ltf["clips"][i]["lidar_pc_token"]

# dump_json(transfuser, transfuser_json.replace('.json', '_reordered_to_ltf.json'))