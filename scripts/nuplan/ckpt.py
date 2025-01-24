import torch

def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)
    if "module" in ckpt.keys():
        ckpt = ckpt["module"]
    # check all paramters whose name include "register_tokens"
    # import pdb; pdb.set_trace()
    for key in ckpt.keys():
        if "register_tokens" in key:
            print(key,ckpt[key])
    return ckpt

load_ckpt("/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/tmp_ckpts/abl_loss_short_dcl_1e-1_full_gpu16_register_zero-01-18-21-12/5000/mp_rank_00_model_states.pt")