# Inherent from [GenDexGrasp](https://github.com/tengyu-liu/GenDexGrasp)
from typing import List, Optional, Tuple
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R
from spahybgen.handmodel import HandModel
import spahybgen.utils.utils_trans_torch as ut_trans_torch


class SpactialOptimization:
    def __init__(
        self,
        robot_name: str,
        hand_scale: float = 1.0,
        batch: int = 32,
        init_rand_scale: float = 0.2,
        tip_downsample: float = 0.7,
        wrench_downsample: float = 0.9,
        focal_ratio: float = 0.2,
        learning_rate: float = 1e-3,
        penetration_mode: str = "surface_penetration",
        grid_type: str = "voxel",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        input_orientation_type: str = "quat",
        voxel_size: float = 0.4 / 80,
        voxel_num: int = 80,
        npts_hand_pnt: int = 128,
        normalize_scale: bool = True,
    ):
        """
        Initialize the SpactialOptimization class.

        Args:
            robot_name (str): Name of the robot hand model.
            hand_scale (float): Scaling factor for the hand model.
            batch (int): Batch size for optimization.
            init_rand_scale (float): Initial random scale for joint initialization.
            tip_downsample (float): Downsampling ratio for tip points.
            wrench_downsample (float): Downsampling ratio for wrench points.
            focal_ratio (float): Ratio for selecting focal points.
            learning_rate (float): Learning rate for optimizer.
            penetration_mode (str): Mode for penetration loss calculation.
            grid_type (str): Type of grid used (e.g., 'voxel').
            device (str): Device for computation ('cuda' or 'cpu').
            input_orientation_type (str): Type of input orientation ('quat' or 'R6d').
            voxel_size (float): Size of each voxel.
            voxel_num (int): Number of voxels.
            npts_hand_pnt (int): Number of hand surface points.
            normalize_scale (bool): Whether to normalize loss scales.
        """
        self.grid_type = grid_type
        self.device = torch.device(device)
        self.robot_name = robot_name
        self.hand_scale = hand_scale
        self.batch = batch
        self.init_random_scale = init_rand_scale
        self.learning_rate = learning_rate
        self.voxel_size = voxel_size
        self.voxel_num = voxel_num
        self.penetration_mode = penetration_mode
        self.input_orientation_type = input_orientation_type
        self.normalize_scale = normalize_scale

        self.npts_hand_surface_points = npts_hand_pnt

        self.losses = None

        self.scene_grid = None
        self.wrench_score_grad = None

        self.focal_ratio = focal_ratio
        self.tip_downsample = tip_downsample
        self.wrench_downsample = wrench_downsample
        self.batch_scene_surface_points = None
        self.batch_scene_surface_grads = None

        self.target_appr_batch = None

        if not self.normalize_scale:  # constant scales for debug only
            self.scale_RH = 1.0
            self.scale_WH = 10.0
            self.scale_FC = 0.01
            self.scale_QH = 1.0
            self.scale_CK = 10.0
            self.scale_AB = 0.1
            self.scale_PN = 100.0

        self.clutter_times = 1

        ## Initial Hand Model
        self.handmodel = HandModel.load_hand_from_json(
            self.robot_name, self.batch, self.device, hand_scale=self.hand_scale
        )
        self.q_joint_lower = self.handmodel.joints_q_lower.detach()
        self.q_joint_upper = self.handmodel.joints_q_upper.detach()
        self.npts_cnt_pts = self.handmodel.contact_points_num
        print(
            "[SpactialOptimization]: Hand Model Initialized - {}. Num of Contact Point: {}".format(
                self.robot_name, self.npts_cnt_pts
            )
        )

    def reset_scene(self, scene_infer_map: torch.Tensor, cam_appr_matrix: list) -> None:
        """
        Reset the scene information and initialize the optimization variables.

        Args:
            scene_infer_map : The inferred scene map, including grid, tip scores, orientations, and wrench scores.
            cam_appr_matrix : The camera approaching matrix for initializing hand orientation.
        """
        self.loss_normalizer = MeanScaleNormalizer()
        self.global_step = 0
        self.scene_information_process(scene_infer_map)

        self.q_current = torch.zeros(
            self.batch, self.handmodel.orie_bit + len(self.handmodel.joints), device=self.device
        )

        ## Initialization of Hand Orientation
        hand_approaching_matrix = self.handmodel.hand_approaching_matrix
        hand_approaching_matrix = np.array(hand_approaching_matrix)
        # initial to cam orientation, multiply to hand original orientation
        target_approaching_matrix = np.array(cam_appr_matrix)
        self.target_appr_batch = torch.Tensor(
            np.tile(np.expand_dims(target_approaching_matrix @ np.array([0, 0, 1]), axis=0), (self.batch, 1))
        ).to(self.device)

        rotation_execute_vector = R.from_matrix(
            np.linalg.solve(target_approaching_matrix, hand_approaching_matrix).T
        ).as_rotvec()

        batch_hand_rot = np.tile(np.expand_dims(rotation_execute_vector, axis=0), (self.batch, 1))
        batch_hand_rot += (np.pi / 6) * np.random.randn(self.batch, 3)  # small randomness to all axis
        # randomness to approaching axis
        approach_rand = np.tile(np.expand_dims(target_approaching_matrix[:, 2], axis=0), (self.batch, 1)) * (
            np.random.randn(self.batch, 1) * (np.pi)
        )

        # Orientation Types: quat, R6d -> pointing vector
        if self.handmodel.orientation_type == "R6d":
            batch_rotation = torch.tensor(
                (R.from_rotvec(batch_hand_rot) * R.from_rotvec(approach_rand)).as_matrix(), device=self.device
            ).float()
            batch_rotation_init = batch_rotation.reshape(self.batch, 9)[:, :6]
        elif self.handmodel.orientation_type == "quat":
            batch_rotation_init = torch.tensor(
                (R.from_rotvec(batch_hand_rot) * R.from_rotvec(approach_rand)).as_quat(), device=self.device
            ).float()
        self.q_current[:, self.handmodel.tras_bit : self.handmodel.orie_bit] = batch_rotation_init

        ## Initialization of Joint Space # joint values intial with randness
        self.q_current[:, self.handmodel.orie_bit :] = (
            self.init_random_scale
            * torch.rand_like(self.q_current[:, self.handmodel.orie_bit :])
            * (self.q_joint_upper - self.q_joint_lower)
            + self.handmodel.initial_joints
        )

        ## Initialization of Hand Translation
        # Randomly Select Wrench Center
        batch_indices = np.random.choice(
            self.wrench_focal_points.shape[0],
            size=self.batch,
            replace=True,
            p=self.wrench_focal_points_weight.cpu().numpy(),
        )
        batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=self.device)
        batch_good_location = self.wrench_focal_points[batch_indices]
        # Contact Point Center
        contact_average = self.handmodel.get_contact_points_average(q=self.q_current).mean(dim=1)
        # Offset from Approach Direction
        target_approaching_shift = (
            np.matmul(target_approaching_matrix, np.array([[0, 0, 1]]).T)
            * self.handmodel.hand_position_offset
        )
        batch_approaches = (
            torch.tensor(target_approaching_shift, dtype=torch.float32, device=self.device)
            .squeeze(-1)
            .unsqueeze(0)
            .repeat(self.batch, 1)
        )
        batch_good_location = batch_good_location - contact_average - batch_approaches
        self.q_current[:, : self.handmodel.tras_bit] = batch_good_location

        self.handmodel.update_kinematics(self.q_current)
        self.q_current.requires_grad = True
        self.optimizer = torch.optim.Adam([self.q_current], lr=self.learning_rate)

    def scene_information_process(self, scene_infer_map: torch.Tensor) -> None:
        """
        Pre-processes the scene inference map and extracts relevant scene information
        for optimization, such as grid, tip scores, orientations, and wrench scores.

        Args:
            scene_infer_map (torch.Tensor): The inferred scene map containing grid, tip scores,
                                            orientations, and wrench scores.
        """
        scene_infer_map = scene_infer_map.to(self.device)
        self.scene_infer_map = scene_infer_map
        self.scene_grid = scene_infer_map[0]

        self.tip_score_full = scene_infer_map[1]
        self.tip_orient_full = scene_infer_map[2:-1]
        self.scene_wrench_full = scene_infer_map[-1]

        self.refine_contact_features_from_raw()
        self.compute_score_scene_gradients()

    def refine_contact_features_from_raw(self) -> None:
        """
        Refine the contact features from the raw scene inference map,
        including selecting focal points for tips and wrenches,
        and updating the wrench focal points with clustering to avoid over-concentration.
        """

        ## Obtain Focal Points
        index_good_tips = torch.argwhere(
            self.tip_score_full > (self.tip_score_full.max() * self.tip_downsample)
        )
        index_good_tips_focal = index_good_tips[
            torch.tensor(
                np.random.choice(
                    index_good_tips.shape[0],
                    size=int(index_good_tips.shape[0] * self.focal_ratio),
                    replace=False,
                ),
                dtype=torch.long,
                device=index_good_tips.device,
            )
        ]
        tip_focal_points = index_good_tips_focal * self.voxel_size  # [sp, 3]
        # Batch to [batch, cp, sp, 3]: [sp, 3] -> [batch, sp, 3] -> [batch, 1, sp, 3] -> [batch, cp, sp, 3]
        self.batch_tip_focal_points = (
            tip_focal_points.unsqueeze(0)
            .repeat(self.batch, 1, 1)
            .unsqueeze(1)
            .repeat(1, self.npts_cnt_pts, 1, 1)
        )
        self.npts_tip_focal_points = self.batch_tip_focal_points.shape[2]
        index_good_wrenches = torch.argwhere(
            self.scene_wrench_full > (self.scene_wrench_full.max() * self.wrench_downsample)
        )
        self.index_good_wrenches_focal = index_good_wrenches[
            torch.tensor(
                np.random.choice(
                    index_good_wrenches.shape[0],
                    size=int(index_good_wrenches.shape[0] * self.focal_ratio),
                    replace=False,
                ),
                dtype=torch.long,
                device=index_good_wrenches.device,
            )
        ]
        wrench_good_points_initial = self.index_good_wrenches_focal * self.voxel_size  # [wp, 3]
        self.npts_wrench_good_points = len(wrench_good_points_initial)

        ## Update the Wrench Focal Point: discrete eps=0.012, min_samples=4
        ## Below dynamically search cluttering parameters to gain balanced focal wrenches; self.clutter_times=1 -> default
        centroid_attempt, centroid_num, best_id = 0, 0, 0
        eps_instance, min_samples_instance = 0.012, 4
        filter_out_thres = 10  # filtering wrench outlines
        while (centroid_attempt < self.clutter_times) and (centroid_num < self.clutter_times):
            object_cloud_labels = torch.from_numpy(
                DBSCAN(eps=eps_instance, min_samples=min_samples_instance).fit_predict(
                    wrench_good_points_initial.clone().cpu().numpy()
                )
            ).to(self.device)
            object_cen_id = torch.unique(object_cloud_labels)
            oci_count = torch.tensor(
                [(object_cloud_labels == oci).sum() for oci in object_cen_id[object_cen_id != -1]]
            )
            if oci_count.max() > filter_out_thres:
                object_cen_id = [
                    oci
                    for oci in object_cen_id
                    if (((object_cloud_labels == oci).sum() > filter_out_thres) and (oci != -1))
                ]
            eps_instance, min_samples_instance = eps_instance, min_samples_instance + 1
            centroid_attempt, centroid_num = centroid_attempt + 1, len(object_cen_id)
            if best_id < centroid_num:
                best_id, object_cloud_labels_new, object_cen_id_new = (
                    centroid_num,
                    object_cloud_labels,
                    object_cen_id,
                )
        # print(f'Centroid clustering: attempt_times:{centroid_attempt}, centroid_num:{centroid_num}')
        self.wrench_focal_points = torch.vstack(
            [
                wrench_good_points_initial[object_cloud_labels_new == id_x].mean(dim=0)
                for id_x in object_cen_id_new
            ]
        ).to(self.device)
        self.batch_wrench_focal_points = self.wrench_focal_points.unsqueeze(0).repeat(
            (self.batch, 1, 1)
        )  # [batch, wp, 3]
        self.npts_wrench_focal_points = self.batch_wrench_focal_points.shape[1]

        ##  Wrench weighted by nums of clutters (Default)
        self.wrench_focal_points_weight = torch.Tensor(
            [(object_cloud_labels_new == id_x).sum() for id_x in object_cen_id_new]
        )

        self.wrench_focal_points_weight = torch.nn.functional.normalize(
            self.wrench_focal_points_weight, p=1, dim=0
        )

    def compute_score_scene_gradients(self) -> None:
        """
        Compute the gradients of the tip scores and wrench scores in the scene inference map for optimization.
        This method calculates the spatial gradients of the tip scores and wrench scores using finite differences,
        which can be used to guide the optimization process towards better contact points and orientations.
        """
        ## Calculate the Gradient of the Tip's Scores
        self.tip_score_grad = torch.stack(
            (
                torch.roll(self.tip_score_full, shifts=-1, dims=0)
                - torch.roll(self.tip_score_full, shifts=1, dims=0),
                torch.roll(self.tip_score_full, shifts=-1, dims=1)
                - torch.roll(self.tip_score_full, shifts=1, dims=1),
                torch.roll(self.tip_score_full, shifts=-1, dims=2)
                - torch.roll(self.tip_score_full, shifts=1, dims=2),
            ),
            dim=3,
        )

        ## Calculate the Gradient of the Wrench's Scores
        self.wrench_score_grad = torch.stack(
            (
                torch.roll(self.scene_wrench_full, shifts=-1, dims=0)
                - torch.roll(self.scene_wrench_full, shifts=1, dims=0),
                torch.roll(self.scene_wrench_full, shifts=-1, dims=1)
                - torch.roll(self.scene_wrench_full, shifts=1, dims=1),
                torch.roll(self.scene_wrench_full, shifts=-1, dims=2)
                - torch.roll(self.scene_wrench_full, shifts=1, dims=2),
            ),
            dim=3,
        )

        # Calculate Normal with TSDF or Voxel
        scene_grid = self.scene_infer_map[0]
        grad_x = torch.roll(scene_grid, shifts=-1, dims=0) - torch.roll(scene_grid, shifts=1, dims=0)
        grad_y = torch.roll(scene_grid, shifts=-1, dims=1) - torch.roll(scene_grid, shifts=1, dims=1)
        grad_z = torch.roll(scene_grid, shifts=-1, dims=2) - torch.roll(scene_grid, shifts=1, dims=2)

        ## For Surface-based Penetration
        # Scene surface and normal with grad.
        scene_surface_grads_full = torch.abs(grad_x) + torch.abs(grad_y) + torch.abs(grad_z)
        scene_surface_grads_full[:, :, [0, -1]] = 0  # remove the top and bottom layers
        index_surface = torch.argwhere(scene_surface_grads_full > 0)

        # print('Optimization: self.index_surface.shape[0]:{}'.format(index_surface.shape[0]))
        scene_surface_limit = 8000
        if index_surface.shape[0] > scene_surface_limit:
            index_surface = index_surface[
                np.random.choice(index_surface.shape[0], size=scene_surface_limit, replace=False), :
            ]
        self.npts_scene_surface_points = index_surface.shape[0]
        # print('Optimization: self.npts_scene_surface_points:{}'.format(self.npts_scene_surface_points))
        scene_surface_points = index_surface * self.voxel_size
        scene_surface_points = scene_surface_points.reshape(
            1, 1, self.npts_scene_surface_points, 3
        )  # [sp ,3] -> [1, 1, sp ,3]
        self.batch_scene_surface_points_surface = scene_surface_points.repeat(
            self.batch, self.npts_hand_surface_points, 1, 1
        )  # [batch, hp, sp ,3]
        self.batch_scene_surface_points_candidate = scene_surface_points.repeat(
            self.batch, self.npts_cnt_pts, 1, 1
        )  # [batch, cp, sp ,3]

        ss_grad_x = grad_x[index_surface[:, 0], index_surface[:, 1], index_surface[:, 2]]
        ss_grad_y = grad_y[index_surface[:, 0], index_surface[:, 1], index_surface[:, 2]]
        ss_grad_z = grad_z[index_surface[:, 0], index_surface[:, 1], index_surface[:, 2]]
        scene_surface_grads = torch.stack([ss_grad_x, ss_grad_y, ss_grad_z], dim=1)
        scene_surface_grads = scene_surface_grads.reshape(
            1, 1, self.npts_scene_surface_points, 3
        )  # [sp ,3] -> [1, 1, sp ,3]
        self.batch_scene_surface_grads_surface = scene_surface_grads.repeat(
            self.batch, self.npts_hand_surface_points, 1, 1
        )  # [batch, hp, sp ,3]
        self.batch_scene_surface_grads_candidate = scene_surface_grads.repeat(
            self.batch, self.npts_cnt_pts, 1, 1
        )  # [batch, cp, sp ,3]

    def get_batch_handmodel_surface_points(self, repeat_num: int) -> torch.Tensor:
        """
        Generates a batch of hand model surface points with repeated samples.

        Args:
            repeat_num (int): The number of times to repeat the hand surface points along the sample dimension.

        Returns:
            torch.Tensor: A tensor of shape [batch, npts_hand_surface_points, repeat_num, 3]
            containing the repeated hand model surface points for each batch.
        """
        hand_surface_points = self.handmodel.get_surface_points(
            downsample_size=self.npts_hand_surface_points
        )  # [batch, hp ,3]
        hand_surface_points = hand_surface_points.reshape(
            self.batch, 1, self.npts_hand_surface_points, 3
        )  # [batch, 1, hp ,3]
        batch_hand_surface_points = hand_surface_points.repeat(1, repeat_num, 1, 1).transpose(
            1, 2
        )  # [batch, hp, sp ,3]
        return batch_hand_surface_points

    def get_batch_handmodel_contact_points_and_normals(self, repeat_num: int) -> tuple:
        """
        Generate batched contact points and normals for the hand model.

        This method samples contact points and normals from the hand model, reshapes them,
        and repeats them along a new dimension to create batched data suitable for
        parallel processing or optimization.

        Args:
            repeat_num (int): The number of times to repeat the contact points and normals along the repeat dimension.

        Returns:
            tuple: A tuple containing:
                - batch_contact_points (torch.Tensor): Tensor of shape [batch, cp, repeat_num, 3]
                - batch_contact_normals (torch.Tensor): Tensor of shape [batch, cp, repeat_num, 3]
        """
        (
            contact_points,
            contact_normals,
        ) = self.handmodel.sample_contact_points_and_normal()  # [batch, cp, 3]
        contact_points = contact_points.reshape(self.batch, 1, -1, 3)  # [batch, 1, cp, 3]
        batch_contact_points = contact_points.repeat(1, repeat_num, 1, 1).transpose(
            1, 2
        )  # [batch, rp, cp, 3]->[batch, cp, rp ,3]
        contact_normals = contact_normals.reshape(self.batch, 1, -1, 3)  # [batch, 1, cp ,3]
        batch_contact_normals = contact_normals.repeat(1, repeat_num, 1, 1).transpose(
            1, 2
        )  # [batch, cp, rp ,3]
        return batch_contact_points, batch_contact_normals

    def orientation_to_RotVec(self, input_orientation: torch.Tensor) -> torch.Tensor:
        """
        quat/R6d to normal
        input_orientation: [batch, cp, 4]
        Converts orientation representations (quaternion or 6D rotation) to rotation vectors (normals).

        Args:
            input_orientation (torch.Tensor): Tensor of shape [batch, cp, 4] (for quaternion) or [batch, cp, 6] (for R6d).

        Returns:
            torch.Tensor: Tensor of shape [batch, cp, 3] representing the rotation vectors (normals).
        """
        ip_shape = input_orientation.shape
        input_orientation = input_orientation.reshape(-1, ip_shape[-1])  # [tp, 4]
        if self.input_orientation_type == "quat":
            input_orientation_r9 = ut_trans_torch.quaternion_to_rot(input_orientation)  # [tp, 4] -> [tp, 3,3]
        elif self.input_orientation_type == "R6d":
            input_orientation_r9 = ut_trans_torch.orthod6d_to_rot(input_orientation)  # [tp, 6] -> [tp, 3,3]

        ori_vectors = (
            torch.tensor([0, 0, 1])
            .float()
            .to(self.device)
            .reshape(1, 3, 1)
            .repeat(input_orientation_r9.shape[0], 1, 1)
        )  # [tp, 3, 1]
        tip_rotats_nr = torch.bmm(input_orientation_r9, ori_vectors)  # [tp, 3, 3] @ [tp, 3, 1] = [tp, 3, 1]
        return tip_rotats_nr.reshape(ip_shape[0], ip_shape[1], 3)

    def loss_custom_kinematics(self) -> torch.Tensor:
        """
        Computes custom kinematic constraint losses for different robot hand models.

        Returns:
            torch.Tensor: containing the custom kinematic losses for each batch.
        """
        loss_ck = torch.zeros(self.batch, device=self.device).to(self.device)
        j_bit = self.handmodel.orie_bit
        # TODO: Remove all sqrt for MSE, revise all abs to pow(), mean all objs
        if self.robot_name == "allegro":
            pass
        elif self.robot_name == "leaphand":
            pass
        elif self.robot_name == "barrett_hand":
            # left and right finger bases should be the same
            loss_ck = torch.abs(self.q_current[:, j_bit + 2] - self.q_current[:, j_bit + 5])
        elif self.robot_name == "robotiq3f":
            os_pf = torch.tensor([0, 4]).int() + j_bit  # joint id offset of two palm finger joints
            os_fj1 = torch.tensor([1, 5, 8]).int() + j_bit  # joint id offset of each finger's first joints
            os_fj2 = torch.tensor([2, 6, 9]).int() + j_bit  # joint id offset of each finger's second joints
            os_fj3 = torch.tensor([3, 7, 10]).int() + j_bit  # joint id offset of each finger's third joints
            # two palm finger joints: should be inverse
            loss_ck_pf = torch.pow(self.q_current[:, os_pf[0]] - self.q_current[:, os_pf[1]], exponent=2)
            # three fingers's first joints: should be identical
            first_joint_mean = torch.mean(self.q_current[:, os_fj1], dim=1, keepdim=True).detach()
            loss_ck_f1 = torch.pow(self.q_current[:, os_fj1] - first_joint_mean, exponent=2).mean(dim=1)
            # three fingers's second joints: should be 0
            loss_ck_f2 = torch.pow(self.q_current[:, os_fj2], exponent=2).mean(dim=1)
            # three fingers's third joints: should be identical, and inverse to 'finger's first joints'
            loss_ck_f3 = torch.pow(self.q_current[:, os_fj3] + first_joint_mean, exponent=2).mean(dim=1)
            loss_ck = (
                torch.sqrt((loss_ck_pf + loss_ck_f1 + loss_ck_f2 + loss_ck_f3) / 4) * 0.1
            )  # scaling factor for robotiq3f
        elif self.robot_name == "softpneu3f":
            # left and right finger bases should be the same
            loss_ck = torch.abs(self.q_current[:, j_bit + 0] - self.q_current[:, j_bit + 2])
        elif self.robot_name == "brunelhand":
            os_fj1 = torch.tensor([1, 3, 5, 7]).int() + j_bit  # joint id offset of each finger's first joints
            os_fj2 = (
                torch.tensor([2, 4, 6, 8]).int() + j_bit
            )  # joint id offset of each finger's second joints
            os_f4 = torch.tensor([5, 6]).int() + j_bit  # joint id offset of final 4th finger's joints
            os_f5 = torch.tensor([7, 8]).int() + j_bit  # joint id offset of final 5th finger's joints
            # Joint one and Joint two of each finger should be identical
            loss_ck_fj12 = torch.pow(self.q_current[:, os_fj1] - self.q_current[:, os_fj2], exponent=2).mean(
                dim=1
            )  # Joint:proxim == Joint:dist
            loss_ck_f45 = torch.pow(
                self.q_current[:, os_f4].sum(dim=1) - self.q_current[:, os_f5].sum(dim=1), exponent=2
            )  # Joints of 4th finger == Joints of 5th finger
            loss_ck = torch.sqrt((loss_ck_fj12 + loss_ck_f45) / 2)
        elif self.robot_name in [
            "robotiq2f",
            "finray2f",
            "finray3f",
            "finray4f",
            "antipodal_hand",
            "antipodal_model_lite",
        ]:
            # all joints should be the same
            joint_values = self.q_current[:, j_bit:]
            loss_ck = torch.abs(joint_values - joint_values.mean(dim=1, keepdim=True).detach()).sum(1)
        else:
            raise KeyError("self.robot_name has no fit with custom_kinematics.")
        return loss_ck

    def force_closure_loss(
        self, contact_points: torch.Tensor, contact_normals: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute force closure losses for a batch of contact points and normals.
        Inherent from [diverse-and-stable-grasp](https://github.com/tengyu-liu/diverse-and-stable-grasp/blob/main/utils/Losses.py)

        Args:
            contact_points (torch.Tensor): Tensor of shape [B, N, 3], where B is batch size and N is number of contacts.
            contact_normals (torch.Tensor): Tensor of shape [B, N, 3], contact normals for each contact point.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - loss_linear_independence: Tensor of shape [B], linear independence loss for each batch.
                - loss_force_closure: Tensor of shape [B], force closure loss for each batch.
        """

        def l2_norm(x):
            if len(x.shape) == 3:
                return torch.sum(x * x, (1, 2))
            if len(x.shape) == 2:
                return torch.sum(x * x, (1))
            raise ValueError

        def x_to_G(x):
            """
            x: B x N x 3
            G: B x 6 x 3N
            """
            B = x.shape[0]
            N = x.shape[1]
            Gx_matrix = (
                torch.tensor(
                    np.array(
                        [
                            [0, 0, 0, 0, 0, -1, 0, 1, 0],
                            [0, 0, 1, 0, 0, 0, -1, 0, 0],
                            [0, -1, 0, 1, 0, 0, 0, 0, 0],
                        ]
                    )
                )
                .float()
                .to(self.device)
            )
            xi_cross = torch.matmul(x, Gx_matrix).reshape([B, N, 3, 3]).transpose(1, 2).reshape([B, 3, 3 * N])
            eye3 = torch.tensor(np.eye(3).reshape(1, 1, 3, 3)).float().to(self.device)
            I = eye3.repeat([B, N, 1, 1]).transpose(1, 2).reshape([B, 3, 3 * N])
            G = torch.stack([I, xi_cross], 1).reshape([B, 6, 3 * N])
            return G

        def loss_8a(G):
            """
            G: B x 6 x 3N
            """
            Gt = G.transpose(1, 2)
            eye6 = torch.tensor(np.eye(6).reshape(1, 6, 6)).float().to(self.device)
            eps = torch.tensor(0.01).float().to(self.device)
            temp = eps * eye6
            temp = torch.matmul(G, Gt) - temp
            eigval = torch.linalg.eigh(temp)[0]
            rnev = torch.nn.functional.relu(-eigval)
            result = torch.sum(rnev * rnev, 1)
            return result

        def loss_8b(f, G):
            """
            G: B x 6 x 3N
            f: B x N x 3
            """
            B = f.shape[0]
            N = f.shape[1]
            return torch.nn.functional.relu(l2_norm(torch.matmul(G, f.reshape(B, 3 * N, 1))))

        G = x_to_G(contact_points)
        loss_linear_independence = loss_8a(G)
        loss_force_closure = loss_8b(contact_normals, G)
        return loss_linear_independence, loss_force_closure

    def penetration_loss(self) -> torch.Tensor:
        """
        Computes the penetration loss between hand model points and scene surface points.

        Depending on the `penetration_mode`, this function calculates the penetration loss using either
        hand-to-object surface points ("surface_penetration") or contact points ("contact_penetration").

        Returns:
            torch.Tensor: A tensor of shape [batch] containing the penetration loss for each batch element.
        """
        ## Penetration with Hand-to-Object Vectors and Object Normals
        if self.penetration_mode == "surface_penetration":
            batch_hand_points = self.get_batch_handmodel_surface_points(
                self.npts_scene_surface_points
            )  # [batch, hp, sp ,3]
            batch_scene_surface_points = self.batch_scene_surface_points_surface
            batch_scene_surface_grads = self.batch_scene_surface_grads_surface
        elif self.penetration_mode == "contact_penetration":
            batch_hand_points, _ = self.get_batch_handmodel_contact_points_and_normals(
                repeat_num=self.npts_scene_surface_points
            )  # [batch, cp , sp, 3]
            batch_scene_surface_points = self.batch_scene_surface_points_candidate
            batch_scene_surface_grads = self.batch_scene_surface_grads_candidate

        batch_hand_points = torch.as_tensor(batch_hand_points, device=self.device)
        batch_scene_surface_points = torch.as_tensor(batch_scene_surface_points, device=self.device)
        scene_to_hand_vect = batch_hand_points - batch_scene_surface_points  # [batch, hp, sp, 3]
        scene_to_hand_dist = scene_to_hand_vect.norm(dim=3)  # [batch, hp, sp]
        close_sh_pairs_inds = scene_to_hand_dist.min(dim=2)[1]  # [batch, hp, sp] -> # [batch, hp]
        ind_arr_0 = torch.arange(close_sh_pairs_inds.shape[0]).repeat(close_sh_pairs_inds.shape[1], 1).T
        ind_arr_1 = torch.arange(close_sh_pairs_inds.shape[1]).repeat(close_sh_pairs_inds.shape[0], 1)
        close_sh_pairs_vect = scene_to_hand_vect[
            ind_arr_0, ind_arr_1, close_sh_pairs_inds, :
        ]  # [batch, hp, sp, 3] -> # [batch, hp, 3]
        close_sh_pairs_nmls = batch_scene_surface_grads[
            ind_arr_0, ind_arr_1, close_sh_pairs_inds, :
        ]  # [batch, hp, sp, 3] -> # [batch, hp, 3]
        close_sh_pairs_dots = (
            torch.nn.functional.normalize(close_sh_pairs_vect)
            * torch.nn.functional.normalize(close_sh_pairs_nmls)
        ).sum(
            dim=2
        )  # [batch, hp]
        loss_penetration = torch.nn.functional.relu(close_sh_pairs_dots).mean(
            dim=1
        )  # [batch, hp] -> # [batch]
        return loss_penetration

    def compute_spacial_loss(self, penetration_check: bool) -> dict:
        """
        Compute the spatial loss for the current hand configuration.

        Args:
            penetration_check (bool): Whether to include the penetration loss in the computation.

        Returns:
            dict: A dictionary containing various loss components for the current batch.
        """
        ## Data Preparasion
        # Current contact states  # [batch, cp, 3]
        batch_cnt_pts, batch_cnt_nms = self.handmodel.sample_contact_points_and_normal()
        # Contact locations and index bank  # [batch, cp, 3]
        batch_CtPt = torch.clip((batch_cnt_pts / self.voxel_size), 0, self.voxel_num - 1).int()
        # ([batch, cp], [batch, cp], [batch, cp])
        batch_CtPt_inds = (batch_CtPt[..., 0], batch_CtPt[..., 1], batch_CtPt[..., 2])
        # Tip orientation score at Contacts # [4,80,80,80] -> [80,80,80,4] -> [batch, cp, 4]
        tip_orient_at_CTs = self.tip_orient_full.permute(1, 2, 3, 0)[batch_CtPt_inds]
        tip_rotvet_at_CTs = self.orientation_to_RotVec(tip_orient_at_CTs)  # [batch, cp, 4] -> [batch, cp, 3]
        # Contact weights
        tip_scoreweights_CTs = self.tip_score_full[batch_CtPt_inds]  # [80,80,80] -> [batch, cp]
        # relu(cos(v1,v2)) - [batch, cp]
        tip_orientweights_CTs = torch.nn.functional.relu((batch_cnt_nms * tip_rotvet_at_CTs).sum(dim=2))
        # Contacts weighted by tip_score and orientation match # [batch, cp, 1]
        tip_weights_at_CTs = (tip_scoreweights_CTs * tip_orientweights_CTs).detach().unsqueeze(-1)

        ## Loss of Tip focus
        size_tips, select_ratio = self.batch_tip_focal_points.shape[2], 2
        selected_indices = torch.randperm(size_tips, device=self.device)[: size_tips // select_ratio]
        selected_batch_tip_focal_points = self.batch_tip_focal_points[:, :, selected_indices, :]
        # [batch, cp, 3] -> [batch, cp, 1, 3] -> [batch, cp, sp/e, 3]
        batch_cnt_pts_fortips = batch_cnt_pts.unsqueeze(2).repeat(1, 1, size_tips // select_ratio, 1)
        # [batch, cp, sp/e, 3] -> [batch, cp, sp/e]
        batch_tip_cont_distances = (batch_cnt_pts_fortips - selected_batch_tip_focal_points).norm(dim=3)
        mini_tip_cont_distances = batch_tip_cont_distances.min(dim=2)[0]  # [batch, cp, sp/e] -> [batch, cp]
        # Calculate a coarse mean distance of contact points
        indx_batch = torch.arange(self.batch).unsqueeze(dim=1).repeat(1, self.npts_cnt_pts)
        indx_b = torch.stack([torch.randperm(self.npts_cnt_pts) for _ in range(self.batch)])
        distance_theshold = (batch_cnt_pts[indx_batch, indx_b, :] - batch_cnt_pts).norm(dim=2).mean() / 10
        # contact-tips close to stop # [batch, cp] -> [batch]
        loss_FQH = torch.nn.functional.relu(mini_tip_cont_distances - distance_theshold.detach()).mean(dim=1)

        ## Loss of Tip Scores Grads # [batch, cp, 3] -> [batch]
        tip_aggregate_ratio = 0.01
        batch_contact_points_star = batch_cnt_pts + self.tip_score_grad[batch_CtPt_inds] * tip_aggregate_ratio
        loss_QH = (batch_cnt_pts - batch_contact_points_star.detach()).norm(dim=2).mean(dim=1)

        ## Loss of Tip Rotations
        self.tip_rotvet_at_CTs_visulization = (batch_cnt_pts, tip_rotvet_at_CTs, batch_cnt_nms)
        loss_RH = (batch_cnt_nms - tip_rotvet_at_CTs).norm(dim=2).mean(dim=1)  # [batch, cp, 3] -> [batch]
        # [batch, cp, 3] -> [batch]
        # loss_RH = ((1 - torch.cosine_similarity(batch_contact_normals, tip_rotvet_at_CTs, dim=2))).mean(dim=1)

        ## Loss of contact centriod scores
        current_ConCent_pts = (batch_cnt_pts).mean(dim=1)  # [batch, cp, 3] * [batch, cp, 1] -> [batch, 3]
        # [batch, 3] -> [batch, wp, 3]
        batch_ConCent_pts = current_ConCent_pts.unsqueeze(1).repeat((1, self.wrench_focal_points.shape[0], 1))
        loss_wh, ind_wh = (batch_ConCent_pts - self.batch_wrench_focal_points.clone()).norm(dim=2).min(dim=1)
        loss_WH = loss_wh

        ## Loss of Force Closure
        ## determine the coordinate origins: closest wrench scores to the current_wrench_points # [batch, 3] -> [batch, wp, 3]
        # batch_ConCent_pts = current_ConCent_pts.unsqueeze(1).repeat((1, self.wrench_focal_points.shape[0], 1))
        # close_cw_inds = (batch_ConCent_pts - self.batch_wrench_focal_points.clone()).norm(dim=2).min(dim=1)[1]
        close_cw_inds = ind_wh
        batch_WrenCent_pts = self.wrench_focal_points[close_cw_inds, :]  # [wp, 3] -> [batch, 3]
        self.batch_WrenCent_points_visualization = batch_WrenCent_pts
        # weight the contacts:random sample with probabilities # [batch, cp, 3]
        # (or a. threshold to filterout contacts; or b. weight the contact again)
        wrenched_cnt_pts = batch_cnt_pts - batch_WrenCent_pts.unsqueeze(1).repeat((1, self.npts_cnt_pts, 1))
        # TODO: the normalize operation may be unnecessary. # [batch, cp, 3] * [batch, cp, 1]
        scaled_cnt_nms = batch_cnt_nms * torch.nn.functional.normalize(tip_weights_at_CTs, p=1.0, dim=1)
        loss_linear_independence, loss_force_closure = self.force_closure_loss(
            wrenched_cnt_pts, scaled_cnt_nms
        )

        ## Loss of Approaching Bias
        curr_hand_appr_vect = torch.matmul(
            self.handmodel.current_approaching_matrices, torch.tensor([0.0, 0.0, 1.0], device=self.device)
        )
        loss_approach_bias = torch.abs(self.target_appr_batch - curr_hand_appr_vect).mean(dim=1)

        ## Loss of Joint Limits  # [batch]
        loss_joint_range = (
            torch.nn.functional.relu(self.q_current[:, self.handmodel.orie_bit :] - self.q_joint_upper)
            + torch.nn.functional.relu(self.q_joint_lower - self.q_current[:, self.handmodel.orie_bit :])
        ).sum(dim=1)

        ## Loss of Custom Gripper's Kinematic Constraints # [batch]
        loss_custom_kine = self.loss_custom_kinematics()

        ## Loss of Scene-Hand Penetration
        if penetration_check:
            loss_pene = self.penetration_loss()
        else:
            loss_pene = torch.zeros(self.batch, device=self.device)

        losses = {
            "FQH": loss_FQH,
            "QH": loss_QH,
            "RH": loss_RH,
            "WH": loss_WH,
            "PN": loss_pene,
            "JR": loss_joint_range,
            "CK": loss_custom_kine,
            "LD": loss_linear_independence,
            "FC": loss_force_closure,
            "AB": loss_approach_bias,
            "CTS": tip_scoreweights_CTs.mean(dim=1),
        }
        return losses

    def step(self, penetration_check: bool) -> None:
        """
        Perform one optimization step for the spatial optimization process.

        Args:
            penetration_check (bool): Whether to include the penetration loss in the loss computation.
        """
        ## simplified MALA step
        if torch.rand(1) > torch.tensor(1 + self.global_step / 20).sigmoid():
            with torch.no_grad():
                q_noise = torch.normal(mean=0.0, std=self.learning_rate, size=self.q_current.shape).to(
                    self.device
                )
                self.q_current.add_(q_noise)
                self.handmodel.update_kinematics(q=self.q_current)  # set current joint states

        self.optimizer.zero_grad()  # zero gradient
        self.handmodel.update_kinematics(q=self.q_current)  # set current joint states
        self.losses = self.compute_spacial_loss(penetration_check)  # compute loss of current states

        losses_unscale = [
            self.losses["RH"],
            self.losses["WH"],
            self.losses["PN"],
            self.losses["FQH"] + self.losses["QH"],
            self.losses["JR"] + self.losses["CK"],
            self.losses["FC"] + self.losses["LD"],
            self.losses["AB"],
        ]
        self.losses["loss_all"] = sum(losses_unscale)

        if self.normalize_scale:
            post_weights = [1.0] * len(losses_unscale)  # piority of task goals, hand-agnostic
            RH_goal_weight, WH_goal_weight, PN_goal_weight = 5, 5, 10
            post_weights[0] = RH_goal_weight  # order as losses_unscale; RH
            post_weights[1] = WH_goal_weight  # order as losses_unscale; WH
            post_weights[2] = PN_goal_weight  # order as losses_unscale; PN
            loss_opti = torch.stack(
                self.loss_normalizer.update_and_scale(losses_unscale, post_weights), dim=0
            ).sum(dim=0)

        else:  # constant scale for debug
            scales = [
                self.scale_RH,
                self.scale_WH,
                self.scale_PN,
                self.scale_QH,
                self.scale_CK,
                self.scale_FC,
                self.scale_AB,
            ]
            loss_opti = torch.stack([x * y for x, y in zip(losses_unscale, scales)], dim=0).sum(dim=0)

        loss_opti.mean().backward()  # get gradient from engergy to joint states
        self.optimizer.step()  # optimize the joint state with obtained gradients

        self.losses["loss_opti"] = loss_opti
        self.global_step += 1

    def get_opt_q(self) -> torch.Tensor:
        """
        Returns the current optimized joint configuration.

        Returns:
            torch.Tensor: The current joint configuration tensor.
        """
        return self.q_current.detach()

    def set_opt_q(self, opt_q: torch.Tensor) -> None:
        """
        Set the current optimized joint configuration.

        Args:
            opt_q (torch.Tensor): The optimized joint configuration tensor to set.
        """
        self.q_current.copy_(opt_q.detach().to(self.device))

    def get_current_plotly_data(
        self,
        index: int = 0,
        color: str = "rgb(100, 0, 100)",
        opacity: float = 1.0,
        text: Optional[str] = None,
    ):
        """
        Get the current hand model's plotly data for visualization.

        Args:
            index (int): Index of the batch to visualize. Default is 0.
            color (str): Color for the plotly object. Default is "rgb(100, 0, 100)".
            opacity (float): Opacity for the plotly object. Default is 1.0.
            text (str, optional): Optional text annotation for the plotly object.

        Returns:
            dict: Plotly data dictionary for the current hand model state.
        """
        return self.handmodel.get_plotly_data(
            q=self.q_current, i=index, color=color, opacity=opacity, text=text
        )


class MeanScaleNormalizer:
    """
    Normalizes a list of loss tensors by their running mean using exponential moving average.

    Args:
        momentum (float, optional): Momentum factor for the running mean (between 0 and 1).
            Higher values give more weight to previous means. Default 0.5
        eps (float, optional): Small value to avoid division by zero. Default is 1e-8.
    """

    def __init__(self, momentum: float = 0.5, eps: float = 1e-8):
        self.m = momentum
        self.eps = eps
        self.mean = None

    def update_and_scale(self, losses: List[torch.Tensor], post_weights: List[float]) -> List[torch.Tensor]:
        """
        Update the running mean of each loss using exponential moving average and scale each loss.

        Args:
            losses (list[torch.Tensor]): List of loss tensors to normalize and scale.
            post_weights (list[float]): List of weights to apply to each normalized loss.

        Returns:
            list[torch.Tensor]: List of scaled and normalized loss tensors.
        """
        scaled = []
        if self.mean is None:
            self.mean = [0.0] * len(losses)
            for i, loss_i in enumerate(losses):
                self.mean[i] = loss_i.detach().numpy().mean()
                scaled.append(losses[i] / (self.mean[i] + self.eps) * post_weights[i])
        else:
            for i, loss_i in enumerate(losses):
                self.mean[i] = self.m * self.mean[i] + (1 - self.m) * loss_i.detach().numpy().mean()
                scaled.append(losses[i] / (self.mean[i] + self.eps) * post_weights[i])

        return scaled
