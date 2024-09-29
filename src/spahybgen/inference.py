# Inherent from [VGN](https://github.com/ethz-asl/vgn)

import numpy as np
from scipy import ndimage
import torch
from pathlib import Path
from spahybgen.networks import load_network

from spahybgen.grasptip import *
from spahybgen.utils.utils_trans_np import Rotation
import spahybgen.utils.utils_trans_np as ut_tranp


def predict(grid_vol, net, device):
    # move input to the GPU
    grid_vol = torch.from_numpy(grid_vol.astype(np.float32)).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        qual_vol, rot_vol, wren_vol = net(grid_vol)

    # move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    wren_vol = wren_vol.cpu().squeeze().numpy()
    return qual_vol, rot_vol, wren_vol


def process(
    qual_vol,
    rot_vol,
    wren_vol,
    gaussian_filter_sigma=1.0
):

    # smooth quality volume with a Gaussian
    if gaussian_filter_sigma > 0:
        qual_vol = ndimage.gaussian_filter(
            qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
        )
        wren_vol = ndimage.gaussian_filter(
            wren_vol, sigma=gaussian_filter_sigma, mode="nearest"
        )
    else: pass
    return qual_vol, rot_vol, wren_vol


def select(qual_vol, rot_vol, wren_vol, qual_threshold=0.90, wren_threshold=0.90):
    # threshold on grasp quality
    qual_vol[qual_vol < qual_threshold] = 0.0
    wren_vol[wren_vol < wren_threshold] = 0.0

    # # construct grasps
    tip_poses, tip_scores = [], []
    for index in np.argwhere(qual_vol > 0):
        tip_pose, tip_score = select_index(qual_vol, rot_vol, index)
        tip_poses.append(tip_pose)
        tip_scores.append(tip_score)
    
    wren_posis = np.argwhere(wren_vol > 0)
    wren_scores = wren_vol[wren_posis[:,0], wren_posis[:,1], wren_posis[:,2]]
    return tip_poses, tip_scores, wren_posis, wren_scores


def fetch_topK(qual_vol, rot_vol, wren_vol, qual_numbers=1000, wren_numbers=1000):
    # # construct grasps
    tip_poses, tip_scores = [], []
    good_list = np.array(np.unravel_index(np.argsort(-qual_vol, axis=None), qual_vol.shape)).T[:qual_numbers]
    for index in good_list:
        tip_pose, tip_score = select_index(qual_vol, rot_vol, index)
        tip_poses.append(tip_pose)
        tip_scores.append(tip_score)
    
    wren_posis = np.array(np.unravel_index(np.argsort(-wren_vol, axis=None), wren_vol.shape)).T[:wren_numbers]
    wren_scores = wren_vol[wren_posis[:,0], wren_posis[:,1], wren_posis[:,2]]
    return tip_poses, tip_scores, wren_posis, wren_scores


def select_index(qual_vol, rot_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    rotats = np.expand_dims(rot_vol[:, i, j, k], 0)
    if rot_vol.shape[0] == 4:  ori = Rotation.from_quat(rotats)
    elif rot_vol.shape[0] == 3: ori = Rotation.from_quat(ut_tranp.so32quat(rotats))
    elif rot_vol.shape[0] == 6: ori = Rotation.from_quat(ut_tranp.R6d2quat(rotats))
    else: raise RuntimeError("Unknown index size for pred orientation.")
    pos = np.array([i, j, k], dtype=np.float64)
    return (ori[0], pos), score



class InferenceBase:
    def __init__(self, model_path, voxel_disc, ori_type) -> None:
        self.voxel_disc = voxel_disc

        ## Inintialize Inference Network
        ntargs = {"voxel_discreteness": voxel_disc, "orientation": ori_type, "augment": False}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(Path(model_path), self.device, ntargs)

    def inference(self, gird_vol, gaussian_filter_sigma=0):
        if(len(gird_vol.shape) == 3): gird_vol = np.expand_dims(gird_vol, axis=0)
        ## grasp generation: inference
        qual_vol, rot_vol, wren_vol = predict(gird_vol, self.net, self.device)
        qual_vol_pro, rot_vol_pro, wren_vol_pro = process(qual_vol, rot_vol, wren_vol, gaussian_filter_sigma)
        prediction = np.vstack([gird_vol, np.expand_dims(qual_vol_pro, axis=0), rot_vol_pro, 
                                np.expand_dims(wren_vol_pro, axis=0)])
        return prediction
    
    