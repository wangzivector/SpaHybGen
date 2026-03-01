from spahybgen.handmodel import HandModel
import torch
import spahybgen.utils.utils_plotly as ut_plotly
from plotly import graph_objects as go


def test_loading_hand_models():
    """Test loading hand model"""
    hand_list = [
        "allegro_hand",
        "barrett_hand",
        "robotiq2f",
        "finray2f",
        "finray3f",
        "finray4f",
        "softpneu3f",
        "robotiq3f",
        "leaphand",
        "brunelhand",
    ]

    batch_size = 8
    downsample_size = 128

    for robot_name in hand_list:
        hand_model = HandModel.load_hand_from_json(robot_name, batch_size, hand_scale=1)

        init_opt_q = torch.zeros(batch_size, (3 + 6) + hand_model.actuate_dofs, device="cuda")
        init_opt_q[:, :3] = torch.tensor([0.0, 0.0, 0.0], device="cuda")  # translation
        init_opt_q[:, 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device="cuda")  # orientation
        init_opt_q[:, 9:] = hand_model.joints_q_lower  # hand joints

        hand_model.update_kinematics(init_opt_q)
        surface_points = hand_model.get_surface_points(init_opt_q, downsample_size=downsample_size)
        assert surface_points.shape == torch.Size([batch_size, downsample_size, 3])

        contact_points, contact_normals = hand_model.sample_contact_points_and_normal(q=init_opt_q)
        assert contact_points.shape[1] != 0
        assert contact_points.shape == contact_normals.shape

        # vis_data = hand_model.get_plotly_data(init_opt_q, color="lightblue", opacity=0.5)
        # trans_meshes = hand_model.get_meshes_from_q(init_opt_q)

        ## surface points visualization
        # vis_data.append(ut_plotly.plot_point_cloud(pts=surface_points.cpu().squeeze(0), color="blue"))

        ## contact points visualization
        # vis_data.append(ut_plotly.plot_point_cloud(pts=contact_points.cpu().squeeze(0), color="red"))
        # for i in range(10):
        #     vis_data.append(
        #         ut_plotly.plot_point_cloud(
        #             pts=(contact_points + 0.001 * i * contact_normals).cpu().squeeze(0), color="yellow"
        #         )
        #     )
        # fig = go.Figure(data=vis_data).show()
