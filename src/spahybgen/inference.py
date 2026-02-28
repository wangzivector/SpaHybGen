# Inherent from [VGN](https://github.com/ethz-asl/vgn)

from typing import Tuple
import numpy as np
from scipy import ndimage, spatial
import torch
from pathlib import Path
from spahybgen.networks import load_network

from spahybgen.grasptip import *
from spahybgen.utils.utils_trans_np import Rotation
import spahybgen.utils.utils_trans_np as ut_tranp


def predict(
    grid_vol: np.ndarray, net: torch.nn.Module, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict the quality, rotation and wrench volumes from the input grid volume using network

    Args:
        grid_vol: The input grid volume, should be in shape of (n, D, H, W)
        net: The loaded network for inference
        device: The device to run the inference

    Returns:
        out:
        - qual_vol: The predicted quality volume, in shape of (D, H, W)
        - rot_vol: The predicted rotation volume, in shape of (C, D, H, W),
            where C is the channel number for rotation representation (e.g., 4 for quaternion)
        - wren_vol: The predicted wrench volume, in shape of (D, H, W)
    """
    # move input to the GPU
    grid_vol_t = torch.from_numpy(grid_vol.astype(np.float32)).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        qual_vol, rot_vol, wren_vol = net(grid_vol_t)

    # move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    wren_vol = wren_vol.cpu().squeeze().numpy()
    return qual_vol, rot_vol, wren_vol


def process(
    qual_vol: np.ndarray, rot_vol: np.ndarray, wren_vol: np.ndarray, gaussian_filter_sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth estimated feature volumes with a Gaussian

    Args:
        qual_vol: The predicted quality volume, in shape of (D, H, W)
        rot_vol: The predicted rotation volume, in shape of (C, D, H, W),
            where C is the channel number for rotation representation (e.g., 4 for quaternion)
        wren_vol: The predicted wrench volume, in shape of (D, H, W)
        gaussian_filter_sigma: sigma value for ndimage.gaussian_filter()

    Returns:
        filtered qual_vol, rot_vol, wren_vol
    """
    if gaussian_filter_sigma > 0:
        qual_vol = ndimage.gaussian_filter(qual_vol, sigma=gaussian_filter_sigma, mode="nearest")
        wren_vol = ndimage.gaussian_filter(wren_vol, sigma=gaussian_filter_sigma, mode="nearest")
    else:
        pass
    return qual_vol, rot_vol, wren_vol


def select(
    qual_vol: np.ndarray,
    rot_vol: np.ndarray,
    wren_vol: np.ndarray,
    qual_threshold: float = 0.90,
    wren_threshold: float = 0.90,
) -> Tuple[list, list, np.ndarray, np.ndarray]:
    """Select the contact poses and wrench positions from the predicted volumes based on the thresholds

    Args:
        qual_vol: The predicted quality volume, in shape of (D, H, W)
        rot_vol: The predicted rotation volume, in shape of (C, D, H, W),
            where C is the channel number for rotation representation (e.g., 4 for quaternion)
        wren_vol: The predicted wrench volume, in shape of (D, H, W)
        qual_threshold: The threshold for selecting contact poses based on quality
        wren_threshold: The threshold for selecting wrench positions based on wrench score

    Returns:
        out:
        - tip_poses: list of selected contact poses, each pose is a tuple of (Rotation, np.ndarray)
        - tip_scores: list of quality scores corresponding to the selected contact poses
        - wren_posis: array of selected wrench positions, (N, 3), N is the number of selected wrench positions
        - wren_scores: array of wrench scores corresponding to the selected wrench positions, in shape of (N,)
    """
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
    wren_scores = wren_vol[wren_posis[:, 0], wren_posis[:, 1], wren_posis[:, 2]]
    return tip_poses, tip_scores, wren_posis, wren_scores


def fetch_topK(
    qual_vol: np.ndarray,
    rot_vol: np.ndarray,
    wren_vol: np.ndarray,
    qual_numbers: int = 1000,
    wren_numbers: int = 1000,
) -> Tuple[list, list, np.ndarray, np.ndarray]:
    """Select the top-K contact poses and wrench positions from the predicted volumes based on the scores

    Args:
        qual_vol: The predicted quality volume, in shape of (D, H, W)
        rot_vol: The predicted rotation volume, in shape of (C, D, H, W),
            where C is the channel number for rotation representation (e.g., 4 for quaternion)
        wren_vol: The predicted wrench volume, in shape of (D, H, W)
        qual_numbers: The number of top contact poses to select based on quality
        wren_numbers: The number of top wrench positions to select based on wrench score

    Returns:
        out:
        - tip_poses: list of selected contact poses, each pose is a tuple of (Rotation, np.ndarray)
        - tip_scores: list of quality scores corresponding to the selected contact poses
        - wren_posis: array of selected wrench positions, (N, 3), N
        - wren_scores: array of wrench scores corresponding to the selected wrench positions, in shape of (N,)
    """
    # # construct grasps
    tip_poses, tip_scores = [], []
    good_list = np.array(np.unravel_index(np.argsort(-qual_vol, axis=None), qual_vol.shape)).T[:qual_numbers]
    for index in good_list:
        tip_pose, tip_score = select_index(qual_vol, rot_vol, index)
        tip_poses.append(tip_pose)
        tip_scores.append(tip_score)

    wren_posis = np.array(np.unravel_index(np.argsort(-wren_vol, axis=None), wren_vol.shape)).T[:wren_numbers]
    wren_scores = wren_vol[wren_posis[:, 0], wren_posis[:, 1], wren_posis[:, 2]]
    return tip_poses, tip_scores, wren_posis, wren_scores


def select_index(
    qual_vol: np.ndarray, rot_vol: np.ndarray, index: np.ndarray
) -> Tuple[Tuple[spatial.transform._rotation.Rotation, np.ndarray], np.ndarray]:
    """Select the contact pose and score from the predicted volumes based on the index

    Args:
        qual_vol: The predicted quality volume, in shape of (D, H, W)
        rot_vol: The predicted rotation volume, in shape of (C, D, H, W),
            where C is the channel number for rotation representation (e.g., 4 for quaternion)
        index: The index of the selected contact pose, in shape of (3,) in (D, H, W)

    Returns:
        out:
        - tip_pose: The selected contact pose, a tuple of (Rotation, np.ndarray)
        - tip_score: The quality score corresponding to the selected contact pose
    """
    i, j, k = index
    score = qual_vol[i, j, k]
    rotats = np.expand_dims(rot_vol[:, i, j, k], 0)
    if rot_vol.shape[0] == 4:
        ori = Rotation.from_quat(rotats)
    elif rot_vol.shape[0] == 3:
        ori = Rotation.from_quat(ut_tranp.so32quat(rotats))
    elif rot_vol.shape[0] == 6:
        ori = Rotation.from_quat(ut_tranp.R6d2quat(rotats))
    else:
        raise RuntimeError("Unknown index size for pred orientation.")
    pos = np.array([i, j, k], dtype=np.float64)
    return (ori[0], pos), score


class InferenceBase:
    """Inference class for predicting contact poses and wrench positions
    from the input grid volume using a trained network loaded locally"""

    def __init__(self, model_path: str, voxel_disc: int, ori_type: str) -> None:
        """Init fun. for Inference

        Args:
            model_path: path to model weights
            voxel_disc: voxel counts, 80
            ori_type: quat or r6d
        """
        self.voxel_disc = voxel_disc

        ## Inintialize Inference Network
        ntargs = {"voxel_discreteness": voxel_disc, "orientation": ori_type, "augment": False}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(Path(model_path), self.device, ntargs)

    def inference(self, gird_vol: np.ndarray, gaussian_filter_sigma: float = 0) -> np.ndarray:
        """Predict the contact poses and wrench positions from the input grid volume using the loaded network

        Args:
            gird_vol: The input grid volume, should be in shape of (n, D, H, W) or (D, H, W)
            gaussian_filter_sigma: The sigma for Gaussian smoothing of the predicted quality and wrench volumes

        Returns:
            prediction: The predicted volumes, in shape of (C, D, H, W),
                where C is the channel number for input grid, quality, rotation and wrench volumes
        """
        if len(gird_vol.shape) == 3:
            gird_vol = np.expand_dims(gird_vol, axis=0)
        ## grasp generation: inference
        qual_vol, rot_vol, wren_vol = predict(gird_vol, self.net, self.device)
        qual_vol_pro, rot_vol_pro, wren_vol_pro = process(qual_vol, rot_vol, wren_vol, gaussian_filter_sigma)
        prediction = np.vstack(
            [
                gird_vol,
                np.expand_dims(qual_vol_pro, axis=0),
                rot_vol_pro,
                np.expand_dims(wren_vol_pro, axis=0),
            ]
        )
        return prediction
