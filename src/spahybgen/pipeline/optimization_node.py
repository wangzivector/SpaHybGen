
import time
import numpy as np
import spahybgen.pipeline.grasp_optimization as Optimization
import spahybgen.utils.utils_trans_np as t_np
import rospy
from std_msgs.msg import Float32MultiArray

class GraspOptNode(object):
    def __init__(self, ori_type, infer_topic, voxel_disc, voxel_size, opti_rate, hand_name, 
                 opti_batches, opti_max_iter, hand_scale, tip_downsample, wrench_downsample, 
                 penetration_mode, grid_type, pent_check_split, save_optimization, focal_ratio) -> None:
        ## Get params
        self.ori_type = ori_type
        self.voxel_disc = voxel_disc
        self.opti_rate = opti_rate
        self.pent_check_split = pent_check_split
        self.hand_name = hand_name
        self.opti_batches = opti_batches
        self.opti_max_iter = opti_max_iter
        self.hand_scale = hand_scale
        self.tip_downsample = tip_downsample
        self.penetration_mode = penetration_mode
        self.grid_type = grid_type
        self.save_optimization = save_optimization
        self.wrench_downsample = wrench_downsample
        self.focal_ratio = focal_ratio
        self.voxel_size = voxel_size
        self.tb_writer = None
        self.inference = None

        ## Gird Callback
        rospy.Subscriber(infer_topic, Float32MultiArray, self.infer_info_cb)

        ## Inintialize Optimization
        self.grasp_optimization = Optimization.GraspOptimization(
            robot_name=self.hand_name, 
            hand_scale=self.hand_scale,
            batch=self.opti_batches, 
            tip_downsample=self.tip_downsample,
            wrench_downsample=self.wrench_downsample,
            focal_ratio = self.focal_ratio,
            learning_rate=self.opti_rate, 
            penetration_mode=self.penetration_mode, 
            grid_type=self.grid_type,
            input_orientation_type=self.ori_type,
            voxel_size=self.voxel_size,
            voxel_num=self.voxel_disc,
        )


    def infer_info_cb(self, msg):
        self.inference = np.array(msg.data).astype(np.float32).reshape(-1, self.voxel_disc, self.voxel_disc, self.voxel_disc)
        rospy.loginfo("\n[Optimization]: Updated inference pred msg: {}\n".format(self.inference.shape))

    def clear_inference_buff(self):
        self.inference = None

    def have_inference_buff(self):
        return self.inference is not None

    def conduct_grasp_optimization(self, inference):
        tic = time.time()
        cam_appr_vector = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
        q_trajectory, loss_dict = self.grasp_optimization.run_optimization(
            scene_infer_map = inference, 
            target_appr_matrix = cam_appr_vector,
            pent_check_split = self.pent_check_split,
            max_iter=self.opti_max_iter, 
            tb_writer=self.tb_writer, 
            running_name=self.hand_name,
        )
        rospy.loginfo("[Inference]: RunOptim: {:.03f}s".format(time.time() - tic))
        return q_trajectory, loss_dict


    @staticmethod
    def final_grasp_configuration(q_trajectory, loss_dict):
        indx_good = loss_dict['sort_ids'][0]
        gq_np = q_trajectory[indx_good]

        ## q2configuration
        pose_transl = gq_np[:, :3]
        pose_rotvet = t_np.Rotation.from_matrix(t_np.R6d2R9d(gq_np[:, 3:9])).as_rotvec()
        pose = np.hstack([pose_transl, pose_rotvet])
        joints = gq_np[:, 9:]
        # return pose, joints
        return pose[-1], joints[-1]
   

    def optimization_pipeline(self, logdir=None, trial_id=0):
        q_trajectory, loss_dict = self.conduct_grasp_optimization(self.inference)
        self.grasp_optimization.visualize_optimization(
            visulize_mode=self.save_optimization, losses_dict=loss_dict, 
            q_trajectory = q_trajectory, filedir=logdir, trial_id=trial_id
            )
        pose, joints = self.final_grasp_configuration(q_trajectory, loss_dict)
        self.inference = None
        return pose, joints




class GraspOptNodeInstance(object):
    def __init__(self, ori_type, infer_topic, voxel_disc, voxel_size, opti_rate, hand_name, 
                 opti_batches, opti_max_iter, hand_scale, tip_downsample, wrench_downsample, 
                 penetration_mode, grid_type, pent_check_split, save_optimization, focal_ratio) -> None:
        ## Get params
        self.ori_type = ori_type
        self.voxel_disc = voxel_disc
        self.opti_rate = opti_rate
        self.pent_check_split = pent_check_split
        self.hand_name = hand_name
        self.opti_batches = opti_batches
        self.opti_max_iter = opti_max_iter
        self.hand_scale = hand_scale
        self.tip_downsample = tip_downsample
        self.penetration_mode = penetration_mode
        self.grid_type = grid_type
        self.save_optimization = save_optimization
        self.wrench_downsample = wrench_downsample
        self.focal_ratio = focal_ratio
        self.voxel_size = voxel_size
        self.tb_writer = None
        self.inference = None

        ## Gird Callback
        rospy.Subscriber(infer_topic, Float32MultiArray, self.infer_info_cb)

    def infer_info_cb(self, msg):
        self.inference = np.array(msg.data).astype(np.float32).reshape(-1, self.voxel_disc, self.voxel_disc, self.voxel_disc)
        rospy.loginfo("\n[Optimization]: Updated inference pred msg: {}\n".format(self.inference.shape))

    def clear_inference_buff(self):
        self.inference = None

    def have_inference_buff(self):
        return self.inference is not None

    def conduct_grasp_optimization(self, grasp_optimization, inference):
        tic = time.time()
        cam_appr_vector = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
        # Temp for multi

        q_trajectory, loss_dict = grasp_optimization.run_optimization(
            scene_infer_map = inference, 
            target_appr_matrix = cam_appr_vector,
            pent_check_split = self.pent_check_split,
            max_iter=self.opti_max_iter, 
            tb_writer=self.tb_writer, 
            running_name=self.hand_name,
        )
        rospy.loginfo("[Inference]: RunOptim: {:.03f}s".format(time.time() - tic))
        return q_trajectory, loss_dict


    @staticmethod
    def final_grasp_configuration(q_trajectory, loss_dict, id):
        indx_good = loss_dict['sort_ids'][id]
        gq_np = q_trajectory[indx_good]

        ## q2configuration
        pose_transl = gq_np[:, :3]
        pose_rotvet = t_np.Rotation.from_matrix(t_np.R6d2R9d(gq_np[:, 3:9])).as_rotvec()
        pose = np.hstack([pose_transl, pose_rotvet])
        joints = gq_np[:, 9:]
        # return pose, joints
        return pose[-1], joints[-1]
   

    def optimization_pipeline(self, logdir=None, trial_id=0, weights=None):
        grasp_optimization = Optimization.GraspOptimization(
            robot_name=self.hand_name, 
            hand_scale=self.hand_scale,
            batch=self.opti_batches, 
            tip_downsample=self.tip_downsample,
            wrench_downsample=self.wrench_downsample,
            focal_ratio = self.focal_ratio,
            learning_rate=self.opti_rate, 
            penetration_mode=self.penetration_mode, 
            grid_type=self.grid_type,
            input_orientation_type=self.ori_type,
            voxel_size=self.voxel_size,
            voxel_num=self.voxel_disc,
        )
        grasp_optimization.set_optimization_weights(weights)
        q_trajectory, loss_dict = self.conduct_grasp_optimization(grasp_optimization, self.inference)
        grasp_optimization.visualize_optimization(
            visulize_mode=self.save_optimization, losses_dict=loss_dict, 
            q_trajectory = q_trajectory, filedir=logdir, trial_id=trial_id
            )
        del grasp_optimization
        import torch
        torch.cuda.empty_cache()

        # pose, joints = self.final_grasp_configuration(q_trajectory, loss_dict)
        self.inference = None
        return q_trajectory, loss_dict
