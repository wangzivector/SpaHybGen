from typing import Optional
import spahybgen.utils.utils_rosvis as ut_vis
import spahybgen.inference as Inference
import numpy as np
import rospy
from spahybgen.utils.utils_trans_np import Transform


def visualize_grid(
    grid: np.ndarray,
    frame_id: str = "grid_ws",
    grid_length: float = 0.4,
    voxel_disc: int = 80,
    threshold: float = 0.1,
    pose: Optional[Transform] = None,
) -> None:
    """Visualize the input grid volume in RViz, with the quality values above the threshold
    Args:
        grid: The input grid volume, should be in shape of (1, D, H, W)
        frame_id: The frame id for visualization in RViz
        grid_length: The physical length of the grid volume, used for visualization
        voxel_disc: The discretization of the grid volume, used for visualization
        threshold: The threshold for highlighting the quality values, should be between 0 and 1
        pose: The pose of the grid volume in the world frame, in shape of (4, 4), or the origin
    """
    ut_vis.clear_grid(frame_id)
    ut_vis.draw_workspace(grid_length, frame=frame_id, pose=pose)
    ut_vis.draw_grid(grid, grid_size=grid_length / voxel_disc, threshold=threshold, frame_id=frame_id)


def visualize_inference(prediction: np.ndarray, voxel_size: float, threshold: float) -> None:
    """Visualize the inference prediction in RViz, including the contact poses and wrench positions

    Args:
        prediction: The inference prediction, in shape of (C, D, H, W),
            where C is the channel number for input grid, quality, rotation and wrench volumes
        voxel_size: The physical size of each voxel in the grid volume, used for visualization
        threshold: The threshold for highlighting the quality values, should be between 0 and 1
    """
    qual_vol_pro, rot_vol_pro, wren_vol_pro = prediction[1], prediction[2:-1], prediction[-1]
    contact_poses, contact_scores, wren_posis, wren_scores = Inference.select(
        qual_vol_pro, rot_vol_pro, wren_vol_pro, threshold, threshold
    )
    num_contact_poses = len(contact_poses)
    if num_contact_poses > 0:
        idx = np.random.choice(num_contact_poses, size=min(3000, num_contact_poses), replace=False)
        contact_poses, contact_scores = [contact_poses[idx_i] for idx_i in idx], np.array(contact_scores)[idx]
    rospy.loginfo("ut_vis.draw_vectors num_poses:{} with threshold : {}".format(num_contact_poses, threshold))
    tips_vectors = ut_vis.visualize_vectors_in_array(contact_poses, contact_scores, voxel_size)
    ut_vis.clear_vectors()
    ut_vis.draw_vectors(tips_vectors, "grid_ws")

    grid_wrench = np.ones_like(qual_vol_pro) * -1
    grid_wrench[wren_posis[:, 0], wren_posis[:, 1], wren_posis[:, 2]] = wren_scores
    ut_vis.draw_quality(grid_wrench, voxel_size, threshold=-1)
