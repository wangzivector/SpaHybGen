# This pipeline is a simple combination of contact_inference_test.py and grasp_optimization_test.py

import numpy as np
from spahybgen import inference as Inference
import torch
from spahybgen.networks import load_network
from pathlib import Path
import os
from datetime import datetime
from spahybgen.pipeline.grasp_optimization import GraspOptimization
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    ## Contact Inference phase
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
    print("loading network: {}".format(model))
    net = load_network(model, device, ntargs)
    qual_vol, rot_vol, wren_vol = Inference.predict(grid_volume, net, device)
    qual_vol_pro, rot_vol_pro, wren_vol_pro = Inference.process(qual_vol, rot_vol, wren_vol, gaussian_filter_sigma=0)
    inferred_result = np.vstack([grid_volume, np.expand_dims(qual_vol_pro, axis=0), rot_vol_pro, np.expand_dims(wren_vol_pro, axis=0)])

    ## Grasp Optimization phase
    robot_name, hand_scale, init_rand_scale = 'robotiq2f', 1.0, 0.2
    tb_writer = None
    logs_basedir = 'data/grasp_otimization'
    time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    tb_dir = os.path.join(logs_basedir, time_stamp)
    tb_writer = SummaryWriter(tb_dir)
    
    result_filedir = 'data/'
    visulize_mode='ONLINE'

    batch_size = 48
    max_iter = 200
    lr = 1e-3
    cam_appr_vector = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
    penetration_mode='contact_penetration'
    pent_check_split = 0.8
    focal_ratio = 0.1
    grid_type='voxel'
    input_orientation_type='quat'
    running_name='trial'
    tip_downsample = 0.7
    wrench_downsample = 0.9

    grasp_generation = GraspOptimization(
        robot_name=robot_name, 
        hand_scale=hand_scale,
        batch=batch_size, 
        init_rand_scale=init_rand_scale,
        focal_ratio = focal_ratio,
        learning_rate=lr, 
        tip_downsample=tip_downsample,
        wrench_downsample=wrench_downsample,
        penetration_mode=penetration_mode, 
        grid_type=grid_type,
        input_orientation_type=input_orientation_type,
    )

    q_trajectory, losses_dict = grasp_generation.run_optimization(
        scene_infer_map = inferred_result, 
        target_appr_matrix = cam_appr_vector,
        pent_check_split = pent_check_split,
        max_iter=max_iter, 
        tb_writer=tb_writer, 
        running_name=running_name, 
    )

    grasp_generation.visualize_optimization(visulize_mode=visulize_mode, filedir=result_filedir,
        q_trajectory=q_trajectory, losses_dict=losses_dict, trial_id=robot_name
    )