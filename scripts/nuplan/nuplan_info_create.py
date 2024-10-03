import numpy as np
import os
from pyquaternion import Quaternion
import pickle
import cv2
import tqdm

from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.image import Image
from nuplan.database.nuplan_db_orm.lidar import Lidar
from nuplan.database.nuplan_db_orm.category import Category
from nuplan.database.nuplan_db_orm.ego_pose import EgoPose
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.log import Log
from nuplan.database.nuplan_db_orm.scene import Scene
from nuplan.database.nuplan_db_orm.scenario_tag import ScenarioTag
from nuplan.database.nuplan_db_orm.traffic_light_status import TrafficLightStatus
from nuplan.database.nuplan_db_orm.track import Track
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_images_from_lidar_tokens,
    get_cameras,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import discover_log_dbs
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario, CameraChannel, LidarChannel
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
from nuplan.database.nuplan_db.nuplan_db_utils import get_lidarpc_sensor_data
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_lidarpc_tokens_with_scenario_tag_from_db,
)
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_lidarpc_tokens_with_scenario_tag_from_db,
    get_sensor_data_token_timestamp_from_db,
    get_sensor_token_map_name_from_db,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    DEFAULT_SCENARIO_NAME,
    ScenarioExtractionInfo,
)

from tutorials.utils.tutorial_utils import get_scenario_type_token_map, get_default_scenario_from_token
import mmcv
import argparse
import json

NUPLAN_DATA_ROOT = "/cpfs01/shared/opendrivelab/opendrivelab_hdd/nuplan/dataset/nuplan-v1.1"
NUPLAN_DB_FILES = f"{NUPLAN_DATA_ROOT}/splits/trainval_sensor"
# NUPLAN_DB_FILES = f"{NUPLAN_DATA_ROOT}/splits/mini"

NUPLAN_SENSOR_ROOT = f"{NUPLAN_DATA_ROOT}/sensor_blobs"
NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"
NUPLAN_MAPS_ROOT = f"/cpfs01/shared/opendrivelab/opendrivelab_hdd/nuplan/dataset/maps"

# Modified
def get_default_scenario_extraction(
    scenario_duration: float = 15.0,
    extraction_offset: float = -2.0,
    subsample_ratio: float = 0.5,
) -> ScenarioExtractionInfo:
    """
    Get default scenario extraction instructions used in visualization.
    :param scenario_duration: [s] Duration of scenario.
    :param extraction_offset: [s] Offset of scenario (e.g. -2 means start scenario 2s before it starts).
    :param subsample_ratio: Scenario resolution.
    :return: Scenario extraction info object.
    """
    return ScenarioExtractionInfo(DEFAULT_SCENARIO_NAME, scenario_duration, extraction_offset, subsample_ratio)


def get_default_scenario_from_token_CUSTOM(
    data_root: str, log_file_full_path: str, token: str, map_root: str, map_version: str,
    scenario_duration: float = 15.0,
    extraction_offset: float = -2.0,
    subsample_ratio: float = 0.5,
) -> NuPlanScenario:
    """
    Build a scenario with default parameters for visualization.
    :param data_root: The root directory to use for looking for db files.
    :param log_file_full_path: The full path to the log db file to use.
    :param token: Lidar pc token to be used as anchor for the scenario.
    :param map_root: The root directory to use for looking for maps.
    :param map_version: The map version to use.
    :return: Instantiated scenario object.
    """
    timestamp = get_sensor_data_token_timestamp_from_db(log_file_full_path, get_lidarpc_sensor_data(), token)
    map_name = get_sensor_token_map_name_from_db(log_file_full_path, get_lidarpc_sensor_data(), token)
    return NuPlanScenario(
        data_root=data_root,
        log_file_load_path=log_file_full_path,
        initial_lidar_token=token,
        initial_lidar_timestamp=timestamp,
        scenario_type=DEFAULT_SCENARIO_NAME,
        map_root=map_root,
        map_version=map_version,
        map_name=map_name,
        scenario_extraction_info=get_default_scenario_extraction(scenario_duration, extraction_offset, subsample_ratio),
        ego_vehicle_parameters=get_pacifica_parameters(),
    )


# TODO: OpenAI Refine
def convert_scenario_info_to_caption(scenario_info):
    # First character is capitalized
    # And the last character is a period
    out = scenario_info.split('_')
    out = ' '.join(out)
    out = out[0].upper() + out[1:]
    if out[-1] != '.':
        out += '.'
    return out


def dump_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Dumped to: {}".format(json_path))

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def split_list_to_k_folds(input_list, k):
    assert k >= 1
    fold_size = len(input_list) // k
    folds = []
    for i in range(k):
        if i == k - 1:
            folds.append(input_list[i * fold_size:])
            # The final fold may be larger
        else:
            folds.append(input_list[i * fold_size: (i + 1) * fold_size])
    return folds


def create_nuplan_info(nuplandb_wrapper, out_dir='./', split=None, n_folds=10, train_val_split='train'):
    train_logs = load_json('/cpfs01/user/yangjiazhi/workspace/DiffuSim/nuplan_outputs/train_logs.json')
    val_logs = load_json('/cpfs01/user/yangjiazhi/workspace/DiffuSim/nuplan_outputs/val_logs.json')

    if train_val_split == 'train':
        nuplan_logs = train_logs
    elif train_val_split == 'val':
        nuplan_logs = val_logs

    log_token2pcs = {}
    log_token2log_db = {}
    lidarpc_token2lidarpc = {}

    all_log_dbs = nuplandb_wrapper.log_dbs
    if split is not None:
        assert isinstance(split, int) and split < n_folds
        all_log_dbs = split_list_to_k_folds(all_log_dbs, n_folds)[split]
    
    print(f"Processing split {split} of {n_folds} folds.")

    for log in tqdm.tqdm(all_log_dbs):
        log_token2pcs[log.log.token] = []
        log_token2log_db[log.log.token] = log.log
        for lidar_pc in log.lidar_pc:
            log_token2pcs[log.log.token].append(lidar_pc)
            lidarpc_token2lidarpc[lidar_pc.token] = lidar_pc
    log_sensors = os.listdir(NUPLAN_SENSOR_ROOT)

    log_idx = 0
    scenario_idx = 0

    all_infos = []
    for log_token in tqdm.tqdm(log_token2log_db, desc='outer loop'):
        log_name = log_token2log_db[log_token].logfile
        log_file = os.path.join(NUPLAN_DB_FILES, log_name + '.db')
        log_db = nuplandb_wrapper.get_log_db(log_name)

        if log_name not in nuplan_logs:
            continue

        if log_name not in log_sensors:
            continue

        log_idx += 1

        # Get mapping from lidar_pc to scenario_type
        scenario_type_lidar_tokens_map = get_scenario_type_token_map([log_file])
        print(f"Number of different scenario types: {len(scenario_type_lidar_tokens_map)}")
        print(scenario_type_lidar_tokens_map.keys())
        
        if len(scenario_type_lidar_tokens_map) == 0:
            # No scenario in this log
            continue

        for scenario_type in tqdm.tqdm(scenario_type_lidar_tokens_map.keys(), desc='scene type loop'):
            print(f"Processing scenario type {scenario_type}: {len(scenario_type_lidar_tokens_map[scenario_type])} clips")

            added_lidarpc_tokens = set()

            logs_with_lidarpc_tokens = scenario_type_lidar_tokens_map[scenario_type]
            # if DEBUG:
            #     logs_with_lidarpc_tokens = logs_with_lidarpc_tokens[:30]
            
            for log_file_full_path, lidarpc_token in logs_with_lidarpc_tokens:
                if lidarpc_token in added_lidarpc_tokens:
                    continue

                subsample_ratio = 1.0
                cur_scenario = get_default_scenario_from_token_CUSTOM(NUPLAN_DATA_ROOT, log_file_full_path, lidarpc_token, NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION,
                                                                      scenario_duration=16,  # Duration: 16s
                                                                      extraction_offset=-2, 
                                                                      subsample_ratio=subsample_ratio)
                all_lidarpc_tokens_in_scenario = cur_scenario.get_scenario_tokens()
                
                # subsample_ratio: 0.5 -> 10 hz, 15s, 150 lidar_pc_tokens(frames) per scenario
                # subsample_ratio: 1.0 -> 20 hz, 15s, 300 lidar_pc_tokens(frames) per scenario

                if any([tk in added_lidarpc_tokens for tk in all_lidarpc_tokens_in_scenario]):
                    continue
                
                added_lidarpc_tokens.update(all_lidarpc_tokens_in_scenario)
                
                if subsample_ratio == 1.0:
                    all_lidarpc_tokens_in_scenario = all_lidarpc_tokens_in_scenario[::2]  # Convert 20hz -> 10hz

                all_lidarpc_tokens_in_scenario = all_lidarpc_tokens_in_scenario[::5]  # !! 10hz -> 2hz
                         
                all_lidarpc_in_scenario = [lidarpc_token2lidarpc[lidarpc_token] for lidarpc_token in all_lidarpc_tokens_in_scenario]
                sorted_all_lidarpc_in_scenario = sorted(all_lidarpc_in_scenario, key=lambda x: x.timestamp)
                assert sorted_all_lidarpc_in_scenario == all_lidarpc_in_scenario

                frame_idx = 0
                for cur_lidarpc in all_lidarpc_in_scenario:
                    scene_token = cur_lidarpc.scene_token
                    scene = cur_lidarpc.scene

                    log_str = '%04d' % log_idx
                    scene_name = 'log-' + log_str + '-' + scene.name

                    time_stamp = cur_lidarpc.timestamp
                    info = {
                        'lidar_pc_token': cur_lidarpc.token,
                        'lidar_token': cur_lidarpc.lidar_token,
                        'scene_token': scene_token,  # temporal related info
                        'scene_name': scene_name,  # additional info
                        'timestamp': time_stamp,  # temporal related info
                        'scenario_index': scenario_idx,  # Each scenario is a video clip
                        'frame_index': frame_idx,
                        'scenario_info': convert_scenario_info_to_caption(scenario_type),
                    }

                    # WIP: Add more 3D infos
                    can_bus = [cur_lidarpc.ego_pose.x, cur_lidarpc.ego_pose.y, cur_lidarpc.ego_pose.z, 
                       cur_lidarpc.ego_pose.qw, cur_lidarpc.ego_pose.qx, cur_lidarpc.ego_pose.qy, cur_lidarpc.ego_pose.qz,
                       cur_lidarpc.ego_pose.acceleration_x, cur_lidarpc.ego_pose.acceleration_y, cur_lidarpc.ego_pose.acceleration_z, 
                       cur_lidarpc.ego_pose.vx, cur_lidarpc.ego_pose.vy, cur_lidarpc.ego_pose.vz, 
                       cur_lidarpc.ego_pose.angular_rate_x, cur_lidarpc.ego_pose.angular_rate_y, cur_lidarpc.ego_pose.angular_rate_z]
                    can_bus.extend([0., 0.])
                    can_bus = np.array(can_bus)

                    lidar = log_db.session.query(Lidar) \
                    .filter(Lidar.token == cur_lidarpc.lidar_token) \
                    .all()

                    three_d_info = {
                        'can_bus': can_bus,
                        'lidar2ego_translation': lidar[0].translation_np,
                        'lidar2ego_rotation': [lidar[0].rotation.w, lidar[0].rotation.x, lidar[0].rotation.y, lidar[0].rotation.z],
                        'ego2global_translation': can_bus[:3],
                        'ego2global_rotation': can_bus[3:7],
                    }

                    l2e_r = three_d_info['lidar2ego_rotation']
                    l2e_t = three_d_info['lidar2ego_translation']
                    e2g_r = three_d_info['ego2global_rotation']
                    e2g_t = three_d_info['ego2global_translation']
                    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

                    # add lidar2global: map point coord in lidar to point coord in the global
                    l2e = np.eye(4)
                    l2e[:3, :3] = l2e_r_mat
                    l2e[:3, -1] = l2e_t
                    e2g = np.eye(4)
                    e2g[:3, :3] = e2g_r_mat
                    e2g[:3, -1] = e2g_t
                    lidar2global  = np.dot(e2g, l2e)
                    three_d_info['ego2global'] = e2g
                    three_d_info['lidar2ego'] = l2e
                    three_d_info['lidar2global'] = lidar2global  # additional info

                    WITH_3D_INFO = True
                    if WITH_3D_INFO:
                        info.update(three_d_info)

                    retrieved_images = get_images_from_lidar_tokens(
                        log_file, [cur_lidarpc.token], [str(channel.value) for channel in CameraChannel]
                    )
                    cams = {}
                    
                    for img in retrieved_images:
                        channel = img.channel
                        filename = img.filename_jpg
                        timestamp = img.timestamp

                        filepath = os.path.join(NUPLAN_SENSOR_ROOT, filename)
                        if not os.path.exists(filepath):
                            frame_str = f'{log_name}, {log_token}, {cur_lidarpc.token}'
                            tqdm.tqdm.write(f'camera file missing: {frame_str}')
                            continue
                        cams[channel] = dict(
                            data_path = filename,
                        )

                    if 'CAM_F0' not in cams:
                        continue
                    
                    info['front_img'] = cams['CAM_F0']['data_path']

                    ## ndarray to list
                    for key in info.keys():
                        if isinstance(info[key], np.ndarray):
                            info[key] = info[key].tolist()

                    all_infos.append(info)
                    frame_idx += 1

                scenario_idx += 1
        
    dump_json(all_infos, os.path.join(out_dir, f'nuplan_info_{split}_2hz.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, default=0)
    args = parser.parse_args()
    deal_split = args.split

    out_dir = '/cpfs01/user/yangjiazhi/workspace/DiffuSim/nuplan_outputs/FULL_OUTPUT_2'
    os.makedirs(out_dir, exist_ok=True)

    nuplandb_wrapper = NuPlanDBWrapper(
        data_root=NUPLAN_DATA_ROOT,
        map_root=NUPLAN_MAPS_ROOT,
        db_files=NUPLAN_DB_FILES,
        map_version=NUPLAN_MAP_VERSION,
    )

    # DEBUG = True

    N_FOLDS = 20
    create_nuplan_info(nuplandb_wrapper, out_dir=out_dir, split=deal_split, n_folds=N_FOLDS, train_val_split='train')