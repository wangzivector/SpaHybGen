import numpy as np
from spahybgen import inference as Inference
import torch
from spahybgen.networks import load_network
from pathlib import Path
import os

if __name__ == "__main__":

    model = Path("assets/trained_models/spahybgen_unet_64_voxel.pt")
    grid_path = os.path.join("assets/observations/scene_010_ann_0124_voxel.npz")
    save_result_path = os.path.join("assets/", 'inference_result.npy')

    grid_volume = np.load(grid_path)["grid"]
    if len(grid_volume.shape) == 3: grid_volume = np.expand_dims(grid_volume, axis=0)
    if len(grid_volume.shape) == 4: grid_volume = np.expand_dims(grid_volume[0], axis=0)

    grid_volume_size, grid_discreteness = 0.4, 80
    voxel_size = grid_volume_size / grid_discreteness
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    ntargs = {"voxel_discreteness": grid_discreteness, "orientation": 'quat', "augment": False}

    net = load_network(model, device, ntargs)
    qual_vol, rot_vol, wren_vol = Inference.predict(grid_volume, net, device)
    qual_vol_pro, rot_vol_pro, wren_vol_pro = Inference.process(qual_vol, rot_vol, wren_vol, gaussian_filter_sigma=0)
    result_prediction = np.vstack([grid_volume, np.expand_dims(qual_vol_pro, axis=0), rot_vol_pro, np.expand_dims(wren_vol_pro, axis=0)])

    np.save(save_result_path, result_prediction)
    print("inference result saved: {}".format(save_result_path))