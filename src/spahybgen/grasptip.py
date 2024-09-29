import numpy as np
import pandas as pd
from spahybgen.utils.utils_trans_np import Transform, Rotation, weighted_average_quaternions

class Grasp(object):
    """Grasp parameterized as pose of a 2-finger robot hand.
    
    TODO(mbreyer): clarify definition of grasp frame
    """

    def __init__(self, pose, width, score, depth, finger_base_depth):
        self.pose = pose
        self.width = width
        self.score = score
        self.depth = depth
        self.finger_base_depth = finger_base_depth

def to_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation /= voxel_size
    width = grasp.width / voxel_size
    return Grasp(pose, width)


def from_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation *= voxel_size
    width = grasp.width * voxel_size
    return Grasp(pose, width)


def index_str2nums(str_nums, is_array=False):
    indexs = np.empty((0, 3))
    if not is_array: 
        nums = str_nums.split('/')
        return np.array([int(nums[0]), int(nums[1]), int(nums[2])])
        
    for tsdf_index in str_nums:
        nums = tsdf_index.split('/')
        index_uvw = [int(nums[0]), int(nums[1]), int(nums[2])]
        indexs = np.vstack([indexs, index_uvw])
    return indexs


def index_nums2str(nums):
    return str(nums[0]) + '/' + str(nums[1]) + '/' + str(nums[2])


def Graspnet2Grasp(grasps, finger_base_depth=0.02): 
    # Grasp frame transfermation: graspnetAPI to VGN
    Rotation_covert = np.array([[0,0,1.0],[0,1,0],[-1,0,0]])
    grasps.rotation_matrices = np.matmul(grasps.rotation_matrices, Rotation_covert)

    Translation_convert = np.zeros((len(grasps), 3))
    Translation_convert[:, 2] = finger_base_depth
    add_translation_mats = np.array(
        [np.matmul(grasps.rotation_matrices[mi], Translation_convert[mi]) for mi in range(len(grasps))]
    )
    grasps.translations -= add_translation_mats

    # Grasp Type definition: graspnetAPI to VGN
    grasp_targets = []
    for grasp in grasps:
        grasp_transformation = \
            np.vstack((np.hstack((grasp.rotation_matrix, grasp.translation[:, None])), [0, 0, 0 ,1]))
        grasp_target = Grasp(
            Transform.from_matrix((grasp_transformation)), 
            grasp.width, grasp.score, grasp.depth, finger_base_depth
        )
        grasp_targets.append(grasp_target)
    return grasp_targets


def Graspnets2Grasps(grasp_group, camera_pose, scene_id, ann_id):
    grasp_group = grasp_group.transform(camera_pose)
    grasp_group_vis = Graspnet2Grasp(grasp_group, finger_base_depth = 0.02)
    # vis.draw_grasps(sample_6d_grasp_group_vis)
    Columns = ["scene_id", "ann_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "depth", "finger_base_depth", "score"]
    df_grasps = pd.DataFrame(columns=Columns)

    for grasp in grasp_group_vis:
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width, depth, finger_base_depth, label = grasp.width, grasp.depth, grasp.finger_base_depth, grasp.score
        new_grasp_ser = pd.Series([scene_id, ann_id, qx, qy, qz, qw, x, y, z, width, depth, finger_base_depth, label], index=Columns)
        df_grasps = pd.concat([df_grasps, new_grasp_ser.to_frame().T], ignore_index=True)
    return grasp_group_vis, df_grasps


def Grasp2Tips(df_raw_grasps):
    tips_data = df_raw_grasps[["x", "y", "z", "qx", "qy", "qz", "qw", "score"]]
    grasp_size = df_raw_grasps.index.size
    handbase_vector, approach_vector = np.zeros((grasp_size, 3)), np.zeros((grasp_size, 3))
    handbase_vector[:, 1], approach_vector[:, 2] = df_raw_grasps["width"], \
        df_raw_grasps["depth"] + df_raw_grasps["finger_base_depth"]

    tsl_handbase_vector, tsl_approach_vector = [], []
    for mi in range(grasp_size):
        R_quat = [df_raw_grasps["qx"][mi], df_raw_grasps["qy"][mi], df_raw_grasps["qz"][mi], df_raw_grasps["qw"][mi]]
        RR = Rotation.from_quat(R_quat).as_matrix()
        tsl_handbase_vector.append(np.matmul(RR, handbase_vector[mi]))
        tsl_approach_vector.append(np.matmul(RR, approach_vector[mi]))

    tf_handbase_vector, tf_approach_vector = np.array(tsl_handbase_vector), np.array(tsl_approach_vector)
    hand_translactions = df_raw_grasps[["x", "y", "z"]].to_numpy()
    left_start = hand_translactions + tf_handbase_vector / 2 
    left_end = hand_translactions + tf_handbase_vector / 2 + tf_approach_vector
    right_start = hand_translactions - tf_handbase_vector / 2 
    right_end = hand_translactions - tf_handbase_vector / 2 + tf_approach_vector

    dual_tips = np.array([left_start, left_end, right_start, right_end]).transpose((1, 0, 2)).reshape((grasp_size, -1))
    tips_data.loc[:, ['lsx', 'lsy', 'lsz','lex', 'ley', 'lez','rsx', 'rsy', 'rsz','rex', 'rey', 'rez']] = dual_tips
    return tips_data


def Tips2TipsDF(tips_data, volume_size, grid_size, interp_ratios = [0.2, 1.0], scene_grid=None, grid_type=None):
    voxel_size = volume_size / grid_size
    left_start, left_end = tips_data[['lsx', 'lsy', 'lsz']].to_numpy(), tips_data[['lex', 'ley', 'lez']].to_numpy()
    right_start, right_end = tips_data[['rsx', 'rsy', 'rsz']].to_numpy(), tips_data[['rex', 'rey', 'rez']].to_numpy()
    rotations, scores = tips_data[['qx','qy', 'qz', 'qw']].to_numpy(), tips_data['score'].to_numpy()

    starts, ends = np.vstack((left_start, right_start)), np.vstack((left_end, right_end))
    rotations, scores = np.vstack((rotations, rotations)), np.hstack((scores, scores))
    interpolation_delta = ends - starts
    delta_length = np.linalg.norm(interpolation_delta, ord=2, axis=1)
    interpolation_delta_normalized = np.vstack([interpolation_delta[ni]/delta_length[ni] for ni in range(interpolation_delta.shape[0])])

    # counts, score_mean, rotation_mean [std] 
    generate_labels = pd.DataFrame(columns=['count', 'mean_score', 'mean_qx', 'mean_qy', 'mean_qz', 'mean_qw', 'tip_label'])
    start_delta_length, end_delta_length = delta_length * interp_ratios[0], delta_length * interp_ratios[1]

    for ind, (start, end) in enumerate(zip(starts, ends)):
        score, rotation, tip_label = scores[ind], rotations[ind], -1 if ind < len(starts)/2 else 1 # -1:'L' 1: 'R'
        start_dist, end_dist = start_delta_length[ind], end_delta_length[ind]
        intdelta_unit, dist_curr = interpolation_delta_normalized[ind], start_dist

        while dist_curr <= end_dist:
            # got current index
            voxel_cord = (start + intdelta_unit * dist_curr)
            # Got index df
            voxel_cord_uvw = np.round(voxel_cord / voxel_size).clip(0, grid_size-1).astype(int)
            tsdf_index = index_nums2str(voxel_cord_uvw)

            if tsdf_index in generate_labels.index:
                data_curr = generate_labels.loc[tsdf_index].to_numpy()
                count_curr, score_curr, tip_label_curr = data_curr[0], data_curr[1], data_curr[-1]
                rotation_curr = data_curr[2:-1]
                # update count
                new_count = count_curr + 1
                # update mean score
                new_score = score_curr + (score - score_curr)/new_count
                # update tip label
                new_tip_label = tip_label_curr + (tip_label - tip_label_curr) / new_count
                # update quaterion
                new_rotation = weighted_average_quaternions(np.vstack([rotation_curr, rotation]), [count_curr, 1])
            else:
                new_count, new_score, new_rotation, new_tip_label = 1, score, rotation, tip_label
            generate_labels.loc[tsdf_index] = np.hstack([new_count, new_score, new_rotation, new_tip_label])
            dist_curr += voxel_size

    mean_score = generate_labels['mean_score'].to_numpy()
    count_ratio = np.tanh(generate_labels['count'].to_numpy())
    tip_weight = np.absolute(generate_labels['tip_label'].to_numpy())
    weight_score =  mean_score * count_ratio * tip_weight
    
    if scene_grid is not None: 
        indexs_grasp = index_str2nums(generate_labels.index, is_array=True).astype(np.int16)
        position_occupy = scene_grid[0, indexs_grasp[:,0], indexs_grasp[:,1], indexs_grasp[:,2]]
        if grid_type == 'tsdf':
            weight_score = np.where(np.logical_and(position_occupy <= 0.5, position_occupy > 1e-3), 0, weight_score)
        elif grid_type == 'voxel':
            weight_score = np.where(position_occupy > 0.5, 0, weight_score)
        generate_labels['weighted_score'] = weight_score

    # transfer direction from approaching to tips
    rotations_aver = generate_labels[['mean_qx', 'mean_qy', 'mean_qz', 'mean_qw']].to_numpy()
    rotations_dire = generate_labels[['tip_label']].to_numpy()
    transfer_mat_90, transfer_mat_90_n = Rotation.from_rotvec([0.5*np.pi, 0.0, 0.0]).as_matrix(), \
        Rotation.from_rotvec([-0.5*np.pi, 0.0, 0.0]).as_matrix()

    rotations_tip_aver = []
    for ri in range(len(rotations_aver)):
        M_av = Rotation.from_quat(rotations_aver[ri]).as_matrix()
        new_av = np.matmul(M_av, transfer_mat_90 if rotations_dire[ri] < 0 else transfer_mat_90_n)
        rotations_tip_aver.append(Rotation.from_matrix(new_av).as_quat())
    rotations_tip_aver = np.vstack(rotations_tip_aver)
    generate_labels[['mean_qx', 'mean_qy', 'mean_qz', 'mean_qw']] = rotations_tip_aver
    return generate_labels


def Tips2WrensDF(tips_data, volume_size, grid_size, interp_ratios = [0.2, 1.0]):
    voxel_size = volume_size / grid_size
    left_start, left_end = tips_data[['lsx', 'lsy', 'lsz']].to_numpy(), tips_data[['lex', 'ley', 'lez']].to_numpy()
    right_start, right_end = tips_data[['rsx', 'rsy', 'rsz']].to_numpy(), tips_data[['rex', 'rey', 'rez']].to_numpy()
    scores = tips_data['score'].to_numpy()

    starts, ends = (left_start + right_start)/2, (left_end + right_end)/2
    interpolation_delta = ends - starts
    delta_length = np.linalg.norm(interpolation_delta, ord=2, axis=1)
    interpolation_delta_normalized = np.vstack([interpolation_delta[ni]/delta_length[ni] for ni in range(interpolation_delta.shape[0])])

    generate_labels = pd.DataFrame(columns=['count', 'mean_score'])
    start_delta_length, end_delta_length = delta_length * interp_ratios[0], delta_length * interp_ratios[1]

    for ind, (start, end) in enumerate(zip(starts, ends)):
        score = scores[ind]
        start_dist, end_dist = start_delta_length[ind], end_delta_length[ind]
        intdelta_unit, dist_curr = interpolation_delta_normalized[ind], start_dist

        while dist_curr <= end_dist:
            # got current index
            voxel_cord = (start + intdelta_unit * dist_curr)
            # Got index df
            voxel_cord_uvw = np.round(voxel_cord / voxel_size).clip(0, grid_size-1).astype(int)
            tsdf_index = index_nums2str(voxel_cord_uvw)

            if tsdf_index in generate_labels.index:
                data_curr = generate_labels.loc[tsdf_index].to_numpy()
                count_curr, score_curr = data_curr[0], data_curr[1]
                # update count
                new_count = count_curr + 1
                # update mean score
                new_score = score_curr + (score - score_curr)/new_count
            else:
                new_count, new_score = 1, score
            generate_labels.loc[tsdf_index] = np.hstack([new_count, new_score])
            dist_curr += voxel_size

    count_ratio = np.tanh(generate_labels['count'].to_numpy())
    mean_score = generate_labels['mean_score'].to_numpy()
    generate_labels['weighted_score'] =  mean_score * count_ratio

    return generate_labels