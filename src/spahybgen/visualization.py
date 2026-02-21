
import spahybgen.utils.utils_rosvis as ut_vis
import spahybgen.inference as Inference
import numpy as np
import rospy


def visualize_grid(grid, frame_id = 'grid_ws', grid_length=0.4, voxel_disc=80, threshold=0.1, pose=None):
    ut_vis.clear_grid(frame_id)
    ut_vis.draw_workspace(grid_length, frame=frame_id, pose=pose)
    ut_vis.draw_grid(grid, grid_size=grid_length/voxel_disc, threshold=threshold, frame_id=frame_id)


def visualize_inference(prediction, voxel_size, threshold):
    qual_vol_pro, rot_vol_pro, wren_vol_pro = prediction[1], prediction[2:-1], prediction[-1]
    contact_poses, contact_scores, wren_posis, wren_scores = \
        Inference.select(qual_vol_pro, rot_vol_pro, wren_vol_pro, threshold, threshold)
    num_contact_poses = len(contact_poses)
    if num_contact_poses > 0:
        idx = np.random.choice(num_contact_poses, size=min(3000, num_contact_poses), replace=False)
        contact_poses, contact_scores = [contact_poses[idx_i] for idx_i in idx], np.array(contact_scores)[idx]
    rospy.loginfo("ut_vis.draw_vectors num_poses:{} with threshold : {}".format(num_contact_poses, threshold))
    tips_vectors = ut_vis.visualize_vectors_in_array(contact_poses, contact_scores, voxel_size)
    ut_vis.clear_vectors()
    ut_vis.draw_vectors(tips_vectors, "grid_ws")

    grid_wrench = np.ones_like(qual_vol_pro) * -1
    grid_wrench[wren_posis[:,0],wren_posis[:,1],wren_posis[:,2]] = wren_scores
    ut_vis.draw_quality(grid_wrench, voxel_size, threshold=-1)

