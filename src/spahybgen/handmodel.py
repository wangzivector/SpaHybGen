# Inherent from [GenDexGrasp](https://github.com/tengyu-liu/GenDexGrasp)
from typing import Optional, Union, Tuple, Any
import json
import os

import torch
import numpy as np
import pytorch_kinematics as pk
import transforms3d
import trimesh as tm
from plotly import graph_objects as go
from pytorch_kinematics.urdf_parser_py.urdf import URDF, Mesh
import trimesh.sample
import einops
import spahybgen.utils.utils_trans_torch as ut_trans_torch


class HandModel:
    def __init__(
        self,
        robot_name: str,
        urdf_path: str,
        mesh_path: str,
        contact_path: str,
        actuate_dofs: int = 0,
        batch_size: int = 1,
        orientation_type: str = "R6d",
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        hand_scale: float = 1.0,
        hand_approaching_matrix: Union[list, np.ndarray] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        hand_position_offset: float = 0.10,
        initial_joints: Optional[np.ndarray] = None,
    ) -> None:
        """Differential hand model for optimization, including the kinematics and geometry of the hand,
        and the contact point information for grasping

        Args:
            robot_name: The name of the hand model, should be consistent with the name in hand_infos.json
            urdf_path: The path to the urdf file of the hand model
            mesh_path: The path to the directory containing the mesh files of the hand model
            contact_path: The path to the json file containing the contact point information of the hand model
            actuate_dofs: The number of actuated degrees of freedom of the hand model
            batch_size: The batch size for optimization, used for parallel optimization of multiple grasps
            orientation_type: The type of orientation representation, should be "R6d" or "quat"
            device: The device for optimization, should be "cuda" or "cpu"
            hand_scale: The scale factor for the hand model, used for optimization
            hand_approaching_matrix: The approaching direction of the hand, used for optimization, should be a 3x3 matrix
            hand_position_offset: The position offset of the hand, used for optimization, should be a positive float
            initial_joints: The initial joint configuration of the hand model, should be a numpy array of shape (actuate_dofs,)
        """
        self.device = device
        self.robot_name = robot_name
        self.actuate_dofs = actuate_dofs
        self.orientation_type = orientation_type
        self.batch_size = batch_size
        self.scale = hand_scale
        self.tras_bit = 3
        ori_type_bits = {"R6d": 6, "quat": 4}
        self.orie_bit = self.tras_bit + ori_type_bits[orientation_type]

        ## load robot models
        self.robot = pk.build_chain_from_urdf(open(urdf_path).read()).to(
            dtype=torch.float, device=self.device
        )
        self.contact_point_dict = json.load(open(contact_path))
        visual = URDF.from_xml_string(open(urdf_path).read())
        self.robot_full = visual

        ## prepare geometries for visualization
        # self.current_approaching_matrices = None
        self.hand_approaching_matrix = hand_approaching_matrix
        self.approaching_matrices_tc = torch.Tensor(hand_approaching_matrix).to(device)
        self.hand_position_offset = hand_position_offset

        ## prepare point basis/normals of contact and surface point samples
        self.surface_points = {}
        self.surface_points_normal = {}
        self.contact_point_basis = {}
        self.contact_normals = {}

        ## Visualization variables
        self.mesh_verts = {}
        self.mesh_faces = {}

        ## load mesh
        for i_link, link in enumerate(visual.links):
            # print(f"Processing link #{i_link}: {link.name}")
            if len(link.visuals) == 0:
                continue
            if type(link.visuals[0].geometry) == Mesh:
                filename = link.visuals[0].geometry.filename
                if filename is not None:
                    mesh = tm.load(os.path.join(mesh_path, filename), force="mesh", process=False)
                else:
                    raise ValueError(f"Mesh filename is None for link {link.name}")
            else:
                print(type(link.visuals[0].geometry))
                raise NotImplementedError
            ## Scale mesh
            try:
                scale = np.array(link.visuals[0].geometry.scale).reshape([1, 3])
            except:
                scale = np.array([[1, 1, 1]])
            ## Mesh transformation
            try:
                rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
                translation = np.reshape(link.visuals[0].origin.xyz, [1, 3])
            except AttributeError:
                rotation = transforms3d.euler.euler2mat(0, 0, 0)
                translation = np.array([[0, 0, 0]])

            ## surface points and normals
            out_surface = trimesh.sample.sample_surface(mesh=mesh, count=128)
            pts = out_surface[0]
            pts_face_index = out_surface[1]
            pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)  # type: ignore
            pts *= scale
            pts = np.matmul(rotation, pts.T).T + translation
            ## pts_normal = np.matmul(rotation, pts_normal.T).T # This line is neglected

            pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
            pts_normal = np.concatenate([pts_normal, np.ones([len(pts_normal), 1])], axis=-1)
            self.surface_points[link.name] = (
                torch.from_numpy(pts).to(device).float().unsqueeze(0).repeat(batch_size, 1, 1)
            )
            self.surface_points_normal[link.name] = (
                torch.from_numpy(pts_normal).to(device).float().unsqueeze(0).repeat(batch_size, 1, 1)
            )

            ## visualization mesh
            self.mesh_verts[link.name] = np.array(mesh.vertices) * scale  # type: ignore
            self.mesh_verts[link.name] = np.matmul(rotation, self.mesh_verts[link.name].T).T + translation
            self.mesh_faces[link.name] = np.array(mesh.faces)  # type: ignore

            ## contact points
            if link.name in self.contact_point_dict:
                self.assign_contact_candidates(link.name, mesh, rotation, translation)

        ## joint limits
        self.inital_joint_limits()

        if initial_joints is not None:
            self.initial_joints = (
                torch.Tensor(initial_joints).to(self.device).unsqueeze(0).repeat(batch_size, 1)
            )
        else:
            self.initial_joints = self.joints_q_lower

    def assign_contact_candidates(
        self, link_name: str, mesh: Any, rotation: np.ndarray, translation: np.ndarray
    ) -> None:
        """Assign contact point candidates for a given link,
        and compute the contact point basis and normals in the world frame.

        Args:
            link_name: The name of link object from the URDF model
            mesh: mesh of link
            rotation: The rotation matrix for the link visual, in shape of (3, 3)
            translation: The translation vector for the link visual, in shape of (1, 3)
        """
        cpb = np.array(self.contact_point_dict[link_name])
        # if there is multiple sets, choose one, test it with allegro hand
        if len(cpb.shape) == 1:
            cpb = np.expand_dims(cpb, axis=0)
        for cpb_part_i in range(cpb.shape[0]):
            cpb_part = cpb[cpb_part_i]
            cpb_part_name = link_name + "+" + str(cpb_part_i)
            cp_basis = mesh.vertices[cpb_part] * self.scale
            cp_basis = np.matmul(rotation, cp_basis.T).T + translation
            cp_basis = torch.cat(
                [
                    torch.from_numpy(cp_basis).to(self.device).float(),
                    torch.ones([4, 1]).to(self.device).float(),
                ],
                dim=-1,
            )
            self.contact_point_basis[cpb_part_name] = cp_basis.unsqueeze(0).repeat(self.batch_size, 1, 1)
            v1 = cp_basis[1, :3] - cp_basis[0, :3]
            v2 = cp_basis[2, :3] - cp_basis[0, :3]
            v1 = v1 / torch.norm(v1)
            v2 = v2 / torch.norm(v2)
            self.contact_normals[cpb_part_name] = torch.cross(v1, v2).view([1, 3])
            self.contact_normals[cpb_part_name] = (
                self.contact_normals[cpb_part_name].unsqueeze(0).repeat(self.batch_size, 1, 1)
            )

    def inital_joint_limits(self) -> None:
        """Initialize the joint limits of the hand model, and compute the mid and variance"""
        self.joints = []
        for i in range(len(self.robot_full.joints)):
            if self.robot_full.joints[i].joint_type in ["revolute", "prismatic"]:
                self.joints.append(self.robot_full.joints[i])
        self.joints_q_mid = []
        self.joints_q_var = []
        self._joints_q_upper = []
        self._joints_q_lower = []
        for i in range(len(self.robot.get_joint_parameter_names())):
            for j in range(len(self.joints)):
                if self.joints[j].name == self.robot.get_joint_parameter_names()[i]:
                    joint = self.joints[j]
            assert joint.name == self.robot.get_joint_parameter_names()[i]
            print(f"Processing joint #{i}: {joint.name}")
            self.joints_q_mid.append((joint.limit.lower + joint.limit.upper) / 2)
            self.joints_q_var.append(((joint.limit.upper - joint.limit.lower) / 2) ** 2)
            self._joints_q_lower.append(joint.limit.lower)
            self._joints_q_upper.append(joint.limit.upper)

        self.joints_q_lower = torch.Tensor(self._joints_q_lower).repeat([self.batch_size, 1]).to(self.device)
        self.joints_q_upper = torch.Tensor(self._joints_q_upper).repeat([self.batch_size, 1]).to(self.device)

    @classmethod
    def load_hand_from_json(
        cls,
        hand_name: str,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        hand_scale: float = 1.0,
        model_path: str = "./handmodel",
    ):
        """Load hand model from json file, the json file should contain the necessary information
        for loading the hand model, including the urdf path, mesh path, contact point information, etc.

        Args:
            hand_name: The name of the hand model, should be consistent with the name in hand_infos.json
            batch_size: The batch size for optimization, used for parallel optimization of multiple grasps
            device: The device for optimization, should be "cuda" or "cpu
            hand_scale: The scale factor for the hand model, used for optimization

        Returns:
            _hand_model: The loaded hand model, an instance of HandModel class
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hand_infos = json.load(open(model_path + "/hand_infos.json"))
        urdf_path = hand_infos["urdf_path"][hand_name]
        mesh_path = hand_infos["meshes_path"][hand_name]
        actuate_dofs = hand_infos["dofs"][hand_name]
        contact_path = hand_infos["contact_path"][hand_name]
        hand_approaching_matrix = hand_infos["hand_approaching_matrix"][hand_name]
        hand_position_offset = hand_infos["position_offset"][hand_name]
        if hand_name in hand_infos["initial_joints"].keys():
            initial_joints = hand_infos["initial_joints"][hand_name]
        else:
            initial_joints = None
        _hand_model = cls(
            robot_name=hand_name,
            urdf_path=urdf_path,
            mesh_path=mesh_path,
            contact_path=contact_path,
            actuate_dofs=actuate_dofs,
            batch_size=batch_size,
            device=device,
            hand_scale=hand_scale,
            hand_approaching_matrix=hand_approaching_matrix,
            hand_position_offset=hand_position_offset,
            initial_joints=initial_joints,
        )
        return _hand_model

    def update_kinematics(self, q: torch.Tensor) -> dict:
        """Update the kinematics of the hand model based on the input joint configuration q,
        and return the current status of the hand model,
        including the global translation, global rotation,
        and the transformation matrices of each link.

        Args:
            q: The input joint configuration, in shape of (Batch, actuate_dofs)

        Returns:
            current_status: The current status of the hand model
        """
        # Translation
        self.global_translation = q[:, : self.tras_bit]
        # Orientation
        ## update for different orientation types: quat, R6d
        if self.orientation_type == "R6d":
            self.global_rotation = ut_trans_torch.robust_R6d2R9d(q[:, self.tras_bit : self.orie_bit])
        elif self.orientation_type == "quat":
            self.global_rotation = ut_trans_torch.quaternion_to_rot(q[:, self.tras_bit : self.orie_bit])

        else:
            raise KeyError("not recognized orientation_type: {}".format(self.orientation_type))

        # Joint
        self.current_status = self.robot.forward_kinematics(q[:, self.orie_bit :])
        # Optimize Hand_approaching direction
        self.current_approaching_matrices = torch.matmul(
            self.global_rotation, self.approaching_matrices_tc.inverse()
        )
        # self.current_approaching_matrices = self.global_rotation
        # self.current_approaching_matrices = self.approaching_matrices_tc
        return self.current_status

    def get_surface_points(
        self, q: Optional[torch.Tensor] = None, downsample_size: Optional[int] = None
    ) -> torch.Tensor:
        """Get the surface points of the hand model based on the current kinematics,
        and return the surface points in the world frame.

        Args:
            q: The input joint configuration, in shape of (Batch, actuate_dofs)
            downsample_size: The number of surface points to be returned, if None, all surface
                points will be returned, if not None, andomly downsampled to the specified size

        Returns:
            surface_points: The surface points of the hand model in the world frame
        """
        if q is not None:
            self.update_kinematics(q)
        surface_points = []
        for link_name in self.surface_points:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[
                    ..., :3
                ]
            )
        surface_points = torch.cat(surface_points, 1)
        assert (
            self.global_rotation is not None and self.global_translation is not None
        ), "Kinematics should be updated before getting surface points"
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(
            1, 2
        ) + self.global_translation.unsqueeze(1)
        if downsample_size is not None:
            surface_points = surface_points[:, torch.randperm(surface_points.shape[1])][:, :downsample_size]
        return surface_points * self.scale

    def get_surface_points_and_normals(
        self, q: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the surface points and normals of the hand model based on the current kinematics,
        and return the surface points and normals in the world frame.

        Args:
            q: The input joint configuration, in shape of (Batch, actuate_dofs)

        Returns:
            out:
            - surface_points: The surface points of the hand model in the world frame
            - surface_normals: The surface normals of the hand model in the world frame
        """
        if q is not None:
            self.update_kinematics(q=q)
        surface_points = []
        surface_normals = []

        for link_name in self.surface_points:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[
                    ..., :3
                ]
            )
            surface_normals.append(
                torch.matmul(trans_matrix, self.surface_points_normal[link_name].transpose(1, 2)).transpose(
                    1, 2
                )[..., :3]
            )
        surface_points = torch.cat(surface_points, 1)
        surface_normals = torch.cat(surface_normals, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(
            1, 2
        ) + self.global_translation.unsqueeze(1)
        surface_normals = torch.matmul(self.global_rotation, surface_normals.transpose(1, 2)).transpose(1, 2)
        return surface_points * self.scale, surface_normals

    def get_contact_points_average(
        self,
        q: Optional[torch.Tensor] = None,
        contact_point_part_indices: Optional[torch.Tensor] = None,
        contact_point_weights: Optional[torch.Tensor] = None,
        smooth: str = "softmax",
    ) -> torch.Tensor:
        """Get the contact points of the hand model based on the current kinematics,
        and return the contact points in the world frame.

        Args:
            q: The input joint configuration, in shape of (Batch, actuate_dofs)
            contact_point_part_indices: (Batch, num_contact_points), if None, all contact point basis will be used
            contact_point_weights: The weights for the contact point basis, in shape of (Batch, num_contact_points, num_basis)
            smooth: The method for normalizing the weights, should be "softmax" or "normalize"

        Returns:
            contact_points: The contact points of the hand model in the world frame, in shape of (Batch, num_contact_points, 3)
        """
        if contact_point_part_indices is None:
            contact_point_part_indices = torch.arange(
                0, len(self.contact_point_basis.keys()), device=self.device
            )
        if contact_point_weights is None:
            contact_num = list(self.contact_point_basis.values())[0].shape[-1]
            contact_point_weights = torch.ones(
                (self.batch_size, contact_point_part_indices.shape[0], contact_num), device=self.device
            )

        if smooth == "softmax":
            contact_point_weights = torch.nn.functional.softmax(contact_point_weights, dim=2)
        elif smooth == "normalize":
            contact_point_weights = torch.nn.functional.normalize(contact_point_weights, dim=2)

        if q is not None:
            self.update_kinematics(q)
        contact_point_basis_transformed = []
        # step 1: transform contact point basis
        for link_name in self.contact_point_basis:
            # get transformation
            link_name_ori = link_name.split("+")[0]
            trans_matrix = self.current_status[link_name_ori].get_matrix()
            # get contact point
            cp_basis = self.contact_point_basis[link_name]
            contact_normal_orig = self.contact_normals[link_name]
            # cp_basis: B x 4 x 3
            cp_basis_transformed = torch.matmul(trans_matrix, torch.transpose(cp_basis, 1, 2)).transpose(
                1, 2
            )[..., :3]
            contact_point_basis_transformed.append(cp_basis_transformed)  # B x 4 x 3
        contact_point_basis_transformed = torch.stack(contact_point_basis_transformed, 1)  # B x J x 4 x 3
        # step 2: collect contact point basis corresponding to each contact point
        contact_point_basis_transformed = contact_point_basis_transformed[
            torch.arange(0, len(contact_point_basis_transformed), device=self.device).unsqueeze(1).long(),
            contact_point_part_indices.long(),
        ]
        # step 3: compute contact point coordinates
        contact_point_basis_transformed = (
            contact_point_basis_transformed * contact_point_weights.unsqueeze(-1)
        ).sum(2)
        contact_point_basis_transformed = torch.matmul(
            self.global_rotation, contact_point_basis_transformed.transpose(1, 2)
        ).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return contact_point_basis_transformed * self.scale

    def get_contact_points_and_normal_average(
        self,
        q: Optional[torch.Tensor] = None,
        contact_point_part_indices: Optional[torch.Tensor] = None,
        contact_point_weights: Optional[torch.Tensor] = None,
        smooth: str = "softmax",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the contact points and normals of the hand model based on the current kinematics.
        The contact points are computed as the weighted average of the contact point basis,
        and the normals are transformed accordingly.

        Args:
            q: The input joint configuration, in shape of (Batch, actuate_dofs)
            contact_point_part_indices: (Batch, num_contact_points), if None, all contact point basis will be used
            contact_point_weights: The weights for the contact point basis, in shape of (Batch, num_contact_points, num_basis)
            smooth: The method for normalizing the weights, should be "softmax" or "normalize"

        Returns:
            out:
            - contact_points: The contact points of the hand model in the world frame, in shape of (Batch, num_contact_points, 3)
            - contact_normals: The contact normals of the hand model in the world frame, in shape of (Batch, num_contact_points, 3)
        """
        if contact_point_part_indices is None:
            contact_point_part_indices = torch.arange(
                0, len(self.contact_point_basis.keys()), device=self.device
            )
        if contact_point_weights is None:
            contact_num = list(self.contact_point_basis.values())[0].shape[-1]
            contact_point_weights = torch.ones(
                (self.batch_size, contact_point_part_indices.shape[0], contact_num), device=self.device
            )

        if smooth == "softmax":
            contact_point_weights = torch.nn.functional.softmax(contact_point_weights, dim=2)
        elif smooth == "normalize":
            contact_point_weights = torch.nn.functional.normalize(contact_point_weights, dim=2)

        if q is not None:
            self.update_kinematics(q)

        contact_point_basis_transformed = []
        contact_normal_transformed = []
        # step 1: transform contact point basis
        for link_name in self.contact_point_basis:
            # get transformation
            link_name_ori = link_name.split("+")[0]
            trans_matrix = self.current_status[link_name_ori].get_matrix()
            # get contact point
            cp_basis = self.contact_point_basis[link_name]
            contact_normal_orig = self.contact_normals[link_name]
            # cp_basis: B x 4 x 3
            cp_basis_transformed = torch.matmul(trans_matrix, cp_basis.transpose(1, 2)).transpose(1, 2)[
                ..., :3
            ]
            contact_point_basis_transformed.append(cp_basis_transformed)  # B x 4 x 3
            contact_normal_transformed.append(
                torch.matmul(trans_matrix[..., :3, :3], torch.transpose(contact_normal_orig, 1, 2)).transpose(
                    1, 2
                )
            )  # B x 1 x 3
        contact_point_basis_transformed = torch.stack(contact_point_basis_transformed, 1)  # B x J x 4 x 3
        contact_normal_transformed = torch.stack(contact_normal_transformed, 1)  # B x J x 1 x 3
        # step 2: collect contact point basis corresponding to each contact point -- torch.Size([1, 2, 4, 3])
        contact_point_basis_transformed = contact_point_basis_transformed[
            torch.arange(0, len(contact_point_basis_transformed), device=self.device).unsqueeze(1).long(),
            contact_point_part_indices.long(),
        ]
        contact_normal_transformed = contact_normal_transformed[
            torch.arange(0, len(contact_normal_transformed), device=self.device).unsqueeze(1).long(),
            contact_point_part_indices.long(),
        ].squeeze(
            2
        )  # B x J x 3
        # # step 3: compute contact point coordinates
        contact_point_basis_transformed = (
            contact_point_basis_transformed * contact_point_weights.unsqueeze(-1)
        ).sum(2)
        contact_point_basis_transformed = torch.matmul(
            self.global_rotation, contact_point_basis_transformed.transpose(1, 2)
        ).transpose(1, 2) + self.global_translation.unsqueeze(1)
        contact_normal_transformed = torch.matmul(
            self.global_rotation, contact_normal_transformed.transpose(1, 2)
        ).transpose(1, 2)
        return contact_point_basis_transformed * self.scale, torch.nn.functional.normalize(
            contact_normal_transformed, p=2, dim=2
        )

    def sample_contact_points_and_normal(
        self, q: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sampled contact points and normals of the hand model based on the current kinematics.
        The contact points are computed by interpolating the contact point basis using the sample matrix,
        and the normals are transformed accordingly.

        Args:
            q: The input joint configuration, in shape of (Batch, actuate_dofs)

        Returns:
            out:
            - contact_points: The sampled contact points of the hand model in the world frame, shape (Batch, num_points, 3)
            - contact_normals: The sampled contact normals of the hand model in the world frame, shape (Batch, num_points, 3)
        """
        if q is not None:
            self.update_kinematics(q)

        contact_point_basis_transformed = []
        contact_normal_transformed = []
        # step 1: transform contact point basis
        for link_name in self.contact_point_basis:
            # get transformation
            link_name_ori = link_name.split("+")[0]
            trans_matrix = self.current_status[link_name_ori].get_matrix()
            # get contact point
            cp_basis = self.contact_point_basis[link_name]
            contact_normal_orig = self.contact_normals[link_name]
            # cp_basis: B x 4 x 3
            cp_basis_transformed = torch.matmul(trans_matrix, cp_basis.transpose(1, 2)).transpose(1, 2)[
                ..., :3
            ]
            contact_point_basis_transformed.append(cp_basis_transformed)  # B x 4 x 3
            contact_normal_transformed.append(
                torch.matmul(trans_matrix[..., :3, :3], torch.transpose(contact_normal_orig, 1, 2)).transpose(
                    1, 2
                )
            )  # B x 1 x 3
        contact_point_basis_transformed = torch.stack(contact_point_basis_transformed, 1)  # B x J x 4 x 3
        contact_normal_transformed = torch.stack(contact_normal_transformed, 1)  # B x J x 1 x 3
        # step 2: collect contact point basis corresponding to each contact point -- torch.Size([1, 2, 4, 3])
        # # step 3: compute contact point coordinates
        contact_point_basis_transformed = torch.matmul(
            self.global_rotation.unsqueeze(1), contact_point_basis_transformed.transpose(2, 3)
        ).transpose(2, 3) + self.global_translation.unsqueeze(1).unsqueeze(1)
        contact_normal_transformed = torch.matmul(
            self.global_rotation.unsqueeze(1), contact_normal_transformed.transpose(2, 3)
        ).transpose(2, 3)
        ## average sample them
        sample_matrix = self._sample_contact_points
        contact_point_basis_transformed = contact_point_basis_transformed.unsqueeze(2).repeat(
            1, 1, len(sample_matrix), 1, 1
        )
        contact_point_basis_transformed = (
            contact_point_basis_transformed * sample_matrix.unsqueeze(0).unsqueeze(-1)
        ).sum(-2)
        contact_point_basis_transformed = einops.rearrange(
            contact_point_basis_transformed, "b h w c -> b (h w) c"
        )
        contact_normal_transformed = contact_normal_transformed.repeat(1, 1, len(sample_matrix), 1)
        contact_normal_transformed = einops.rearrange(contact_normal_transformed, "b h w c -> b (h w) c")
        return contact_point_basis_transformed * self.scale, torch.nn.functional.normalize(
            contact_normal_transformed, p=2, dim=2
        )

    @property
    def contact_points_num(self):
        return len(self.contact_normals) * len(self._sample_contact_points)

    @property
    def _sample_contact_points(self):
        rich_interpolate_thres = 3
        if len(self.contact_point_basis) <= rich_interpolate_thres:
            # if few contacts, 9 point samples, with following interpolation ratios related to four corners
            # the contact interpolation matrix
            return (
                torch.tensor(
                    [
                        [0.2, 0, 0.8, 0],
                        [0, 0.2, 0, 0.8],
                        [0, 0.8, 0, 0.2],
                        [0.8, 0, 0.2, 0],
                        [0.5, 0.5, 0, 0],
                        [0, 0.5, 0.5, 0],
                        [0.5, 0, 0, 0.5],
                        [0, 0.5, 0, 0.5],
                        [0, 0, 0.5, 0.5],
                    ]
                )
                .float()
                .to(self.device)
            )
        else:
            # too much contacts/links # 5 point samples
            return (
                torch.tensor(
                    [[0.2, 0, 0.8, 0], [0, 0.2, 0, 0.8], [0, 0.8, 0, 0.2], [0.8, 0, 0.2, 0], [0, 0.5, 0, 0.5]]
                )
                .float()
                .to(self.device)
            )

    def get_meshes_from_q(self, q: Optional[torch.Tensor] = None, i: int = 0) -> list:
        """
        Get the transformed meshes of the hand model for a given joint configuration.

        Args:
            q: The input joint configuration, in shape of (Batch, actuate_dofs). If None, uses current kinematics.
            i: The index of the batch to visualize.

        Returns:
            data: A list of trimesh.Trimesh objects representing the transformed meshes.
        """
        data = []
        if q is not None:
            self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(
                self.global_rotation[i].detach().cpu().numpy(), transformed_v.T
            ).T + np.expand_dims(self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(tm.Trimesh(vertices=transformed_v, faces=f))
        return data

    def get_plotly_data(
        self,
        q: Optional[torch.Tensor] = None,
        i: int = 0,
        color: str = "lightblue",
        opacity: float = 1.0,
        text: Optional[str] = None,
        light: Optional[dict] = None,
    ) -> list:
        """
        Generate Plotly Mesh3d objects for visualizing the hand model.

        Args:
            q: The input joint configuration, in shape of (Batch, actuate_dofs). If None, uses current kinematics.
            i: The index of the batch to visualize.
            color: The color of the mesh.
            opacity: The opacity of the mesh.
            text: Optional text annotation for the mesh.
            light: Optional lighting dictionary for Plotly Mesh3d.

        Returns:
            data: A list of plotly.graph_objects.Mesh3d objects representing the transformed meshes.
        """
        if light is None:
            light = dict(ambient=0.6, diffuse=0.6, roughness=1.0, specular=0.05, fresnel=4.0)
        data = []
        if q is not None:
            self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(
                self.global_rotation[i].detach().cpu().numpy(), transformed_v.T
            ).T + np.expand_dims(self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(
                go.Mesh3d(
                    x=transformed_v[:, 0],
                    y=transformed_v[:, 1],
                    z=transformed_v[:, 2],
                    i=f[:, 0],
                    j=f[:, 1],
                    k=f[:, 2],
                    color=color,
                    opacity=opacity,
                    text=text,
                    lighting=light,
                )
            )
        return data
