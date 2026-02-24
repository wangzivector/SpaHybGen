# This pipeline is a simple combination of contact_inference_test.py and grasp_optimization_test.py

import numpy as np
from spahybgen import inference as Inference
import torch
from spahybgen.networks import load_network
from pathlib import Path
from datetime import datetime
from spahybgen.pipeline.grasp_optimization import GraspOptimization
from torch.utils.tensorboard.writer import SummaryWriter
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", type=str, required=False, default="robotiq2f")
    parser.add_argument("--max_iter", type=int, required=False, default=120)
    parser.add_argument("--batch_size", type=int, required=False, default=64)
    args = parser.parse_args()

    ## parameters of Optimization Module
    robot_name = args.hand
    batch_size = args.batch_size
    max_iter = args.max_iter
    penetration_mode = "contact_penetration"

    ## files for Contact Inference
    if not Path("./assets").is_dir():
        raise FileExistsError("Please first go to root of spahybgen.")
    model = Path("assets/trained_models/spahybgen_unet_64_voxel.pt")
    grid_path = Path("assets/observations/scene_010_ann_0124_voxel.npz")
    save_result_path = Path("assets/inference_result.npy")

    ## Contact Inference phase
    grid_volume = np.load(grid_path)["grid"]
    if len(grid_volume.shape) == 3:
        grid_volume = np.expand_dims(grid_volume, axis=0)
    if len(grid_volume.shape) == 4:
        grid_volume = np.expand_dims(grid_volume[0], axis=0)

    print("loading network: {}".format(model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_volume_size = 0.4
    grid_discreteness = 80
    ntargs = {"voxel_discreteness": grid_discreteness, "orientation": "quat", "augment": False}
    voxel_size = grid_volume_size / grid_discreteness
    net = load_network(model, device, ntargs)
    qual_vol, rot_vol, wren_vol = Inference.predict(grid_volume, net, device)
    qual_vol_pro, rot_vol_pro, wren_vol_pro = Inference.process(
        qual_vol, rot_vol, wren_vol, gaussian_filter_sigma=0
    )
    inferred_result = np.vstack(
        [grid_volume, np.expand_dims(qual_vol_pro, axis=0), rot_vol_pro, np.expand_dims(wren_vol_pro, axis=0)]
    )

    ## Grasp Optimization phase
    visulize_mode = "ONLINE"
    running_name = "trial"
    result_filedir = "data"

    logs_basedir = Path(result_filedir, "grasp_otimization")
    time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    tb_dir = Path(logs_basedir, time_stamp)
    tb_writer = SummaryWriter(tb_dir)
    print(f"SummaryWriter log to: {tb_dir}")

    grasp_generation = GraspOptimization(
        robot_name=robot_name,
        batch=batch_size,
        penetration_mode=penetration_mode,
    )

    q_trajectory, losses_dict = grasp_generation.run_optimization(
        scene_infer_map=inferred_result,
        max_iter=max_iter,
        tb_writer=tb_writer,
        running_name=running_name,
    )

    grasp_generation.visualize_optimization(
        visulize_mode=visulize_mode,
        filedir=result_filedir,
        q_trajectory=q_trajectory,
        losses_dict=losses_dict,
        trial_id=robot_name,
    )
