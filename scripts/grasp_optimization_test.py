from datetime import datetime
import numpy as np
from spahybgen.pipeline.grasp_optimization import GraspOptimization
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", type=str, required=False, default="robotiq2f")
    parser.add_argument("--max_iter", type=int, required=False, default=120)
    parser.add_argument("--batch_size", type=int, required=False, default=64)
    args = parser.parse_args()
    robot_name = args.hand

    if not Path("./assets").is_dir():
        raise FileExistsError("Please first go to root of spahybgen.")
    inferred_result = np.load("assets/inference_result.npy")

    visulize_mode = "ONLINE"
    penetration_mode = "contact_penetration"
    batch_size = args.batch_size
    max_iter = args.max_iter

    tb_writer = None
    running_name = "trial"
    result_filedir = "data/"
    logs_basedir = result_filedir + "grasp_otimization"
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
        scene_infer_map_np=inferred_result,
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
