from datetime import datetime
import numpy as np
from spahybgen.pipeline.grasp_optimization import GraspOptimization


if __name__ == '__main__':
    import os
    from torch.utils.tensorboard import SummaryWriter

    robot_name, hand_scale, init_rand_scale = 'finray2f', 1.0, 0.2
    robot_name, hand_scale, init_rand_scale = 'brunel_hand', 1.0, 0.2
    robot_name, hand_scale, init_rand_scale = 'antipodal_hand', 1.0, 0.2
    robot_name, hand_scale, init_rand_scale = 'finray4f', 1.0, 0.2
    robot_name, hand_scale, init_rand_scale = 'leaphand', 1.0, 0.2
    robot_name, hand_scale, init_rand_scale = 'robotiq3f', 1.0, 0.2
    robot_name, hand_scale, init_rand_scale = 'softpneu3f', 1.0, 0.2 # ModelO
    robot_name, hand_scale, init_rand_scale = 'robotiq2f', 1.0, 0.2

    tb_writer = None
    logs_basedir = 'data/graspoti'
    time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    tb_dir = os.path.join(logs_basedir, time_stamp)
    tb_writer = SummaryWriter(tb_dir)
    
    inferred_result = np.load('assets/std_inference_result_from_clutter.npy')
    result_filedir = 'assets/'
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