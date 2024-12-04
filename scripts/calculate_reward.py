import json
import yaml
from tqdm import tqdm
import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_json(json_path):
    print("Loading json: {}".format(json_path))
    with open(json_path) as f:
        data = json.load(f)
    return data

def dump_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Dumped to: {}".format(json_path))


# * Consider the evaluation metric of uniad?
def ade_normalized(x, y, horizon=8):
    x = np.array(x)[:horizon]
    y = np.array(y)[:horizon]

    dispalcement = np.linalg.norm(x - y, axis=1)
    normalization = np.linalg.norm(x, axis=1)
    delta = 1e-6

    return np.mean(dispalcement)

    # return np.mean(dispalcement) / (np.mean(normalization) + delta)


# * Draw the curve of pdm_score and reward wrt. data points
# def draw_plot(res):
#     pdm_scores = [clip['pdm_score'] for clip in res]
#     reward     = [clip['reward'] for clip in res]
    
def draw_plot(res, filename='reward'):
    pdm_scores = [clip['pdm_score'] for clip in res]
    gt_traj_reward     = [clip['gt_traj_reward'] for clip in res]
    pred_traj_reward   = [clip['pred_traj_reward'] for clip in res]
    data = pd.DataFrame({
        'GT_Traj_Reward': gt_traj_reward,
        'Pred_Traj_Reward': pred_traj_reward,
        'Pred_Traj_PDMScore': pdm_scores
    })

    plt.figure(figsize=(10, 5))
    
    plt.plot(gt_traj_reward, label='GT_Traj_Reward', color='orange')
    plt.plot(pred_traj_reward, label='Pred_Traj_Reward', color='green')
    plt.plot(pdm_scores, label='Pred_Traj_PDMScore', color='blue')
    
    
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.title('PDM Score and Reward vs Data Points')
    plt.legend()
    plt.savefig(f'/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/navsim_eval/reward/{filename}.png')

    # 计算相关性矩阵
    correlation_matrix = data.corr()
    print("Correlation matrix:")
    print(correlation_matrix)

    # 可视化相关性矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig(f'/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/navsim_eval/reward/{filename}_corr.png')
    plt.show()

    # 绘制散点图矩阵
    # 绘制散点图矩阵
    # sns.pairplot(data, diag_kind='kde', markers='o')
    # plt.suptitle('Scatterplot Matrix', y=1.02)
    # plt.savefig(f'/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/navsim_eval/reward/{filename}_analysis.png')
    # plt.show()
    
    # plt.show()

if __name__ == "__main__":
    policy_and_traj_json = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/navsim_eval/debug2/infer_nuplan5_lora_not-contained_all_tokens_resume-from-256_not-apply-traj_planning-11-01-14-30_out_traj_eval_video_idm_planner_trans.json'

    gt_traj_controlled_wm_idm_json = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/navsim_eval/debug2/reward_infer_nuplan5_lora_resume-from-256_wm_gt-traj-12-04-11-23_out_traj_eval_idm_with_gen_gt_traj.json'
    policy_traj_controlled_wm_idm_json = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/navsim_eval/debug2/reward_infer_nuplan5_lora_resume-from-256_wm_pred-traj-12-04-11-27_out_traj_eval_idm_with_gen_pred_traj.json'

    policy_and_traj_res = load_json(policy_and_traj_json)
    policy_and_traj_res = {
        clip['token']: clip for clip in policy_and_traj_res['clips']
    }

    gt_traj_controlled_wm_idm_res = load_json(gt_traj_controlled_wm_idm_json)
    gt_traj_controlled_wm_idm_res = {
        clip['token']: clip for clip in gt_traj_controlled_wm_idm_res['clips']
    }

    policy_traj_controlled_wm_idm_res = load_json(policy_traj_controlled_wm_idm_json)
    policy_traj_controlled_wm_idm_res = {
        clip['token']: clip for clip in policy_traj_controlled_wm_idm_res['clips']
    }
    
    tokens_gt_idm = gt_traj_controlled_wm_idm_res.keys()
    tokens_policy_idm = policy_traj_controlled_wm_idm_res.keys()
    # intersection
    tokens_to_eval = set(tokens_gt_idm).intersection(set(tokens_policy_idm))

    all_eval_res = []
    for token in tokens_to_eval:
        policy_and_traj_clip = policy_and_traj_res[token]
        gt_traj_idm = gt_traj_controlled_wm_idm_res[token]['pred_traj_fut']
        policy_traj_idm = policy_traj_controlled_wm_idm_res[token]['pred_traj_fut']
        
        policy_and_traj_clip['gt_traj_fut_idm'] = gt_traj_idm
        policy_and_traj_clip['pred_traj_fut_idm'] = policy_traj_idm
        
        Offset = 1
        policy_and_traj_clip['gt_traj_reward'] = -1 * ade_normalized(policy_and_traj_clip['gt_traj_fut'], gt_traj_idm) + Offset
        policy_and_traj_clip['pred_traj_reward'] = -1 * ade_normalized(policy_and_traj_clip['pred_traj_fut'], policy_traj_idm) + Offset

        policy_and_traj_clip['pdm_score'] = policy_and_traj_clip['scores']['pdm_score']

        # * Remove unused keys
        pop_keys = ['img_seq_his', 'img_seq_fut', 'cmd', 'img_seq_gen_2hz']
        for pop_key in pop_keys:
            policy_and_traj_clip.pop(pop_key)

        all_eval_res.append(policy_and_traj_clip)

    draw_plot(all_eval_res, filename='all_traj')
    draw_plot([res for res in all_eval_res if res['pdm_score'] < 0.5], filename='bad_pred_traj')
    draw_plot([res for res in all_eval_res if res['pdm_score'] >= 0.5], filename='good_pred_traj')


    dump_json(all_eval_res, '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/navsim_eval/reward/reward.json')

    import pdb; pdb.set_trace()