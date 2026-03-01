import numpy as np
from spahybgen.pipeline.param_server import GraspParameter
from spahybgen.pipeline.grasp_optimization import GraspOptimization
import torch


def test_load_parameter_ros():
    GP = GraspParameter("./config/grasp_generation.yaml")


def test_grasp_optimization():
    hand_list = [
        "robotiq2f",
        "finray2f",
        "finray3f",
        "finray4f",
        "softpneu3f",
        "robotiq3f",
        "leaphand",
        "brunelhand",
    ]
    batch_size = 64
    max_iter = 120
    tb_writer = None
    running_name = "trial"
    penetration_mode = "contact_penetration"

    inferred_result = np.load("assets/inference_result.npy")  # download as guided in README.md
    # inferred_result = np.random.rand(7, 80, 80, 80) # or a ramdon feature to test

    for robot_name in hand_list:
        grasp_generation = GraspOptimization(
            robot_name=robot_name, batch=batch_size, penetration_mode=penetration_mode
        )

        q_trajectory, losses_dict = grasp_generation.run_optimization(
            scene_infer_map_np=inferred_result,
            max_iter=max_iter,
            tb_writer=tb_writer,
            running_name=running_name,
        )
        hand_dofs = 3 + 6 + len(grasp_generation.handmodel.joints_q_lower[0])
        assert q_trajectory.shape == torch.Size([batch_size, max_iter + 1, hand_dofs])
        assert losses_dict["sort_ids"].shape == (batch_size,)
