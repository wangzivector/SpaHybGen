from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from spahybgen.utils import utils_dataio
import spahybgen.grasptip as GraspType
import spahybgen.utils.utils_trans_np as ut_tranp


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, numsample=5000, orientation_type='quat', grid_type='tsdf', data_type='Indexed', tqdm_disable=False):
        self.root = root
        self.data_type = data_type
        self.numsample = numsample
        # self.augment = augment
        self.orientation_type = orientation_type
        self.grid_type = grid_type
        self.data = {
            'scene_tsdf_path': [],
            'scene_voxel_path': [],
            'contact_path': [],
            'wrench_path': [],
        }
        for f in tqdm(list((self.root).rglob("*.npz")), desc="Sorting Dataset Files", disable=tqdm_disable):
            if f.stem[-5:] != 'voxel':
                self.data["scene_tsdf_path"].append(f)
                self.data["contact_path"].append(f.with_name(f.stem + "_VoxelTips.csv"))
                self.data["wrench_path"].append(f.with_name(f.stem + "_VoxelWrens.csv"))
            else: 
                self.data["scene_voxel_path"].append(f)
        self.data["scene_tsdf_path"].sort()
        self.data["scene_voxel_path"].sort()
        self.data["contact_path"].sort()
        self.data["wrench_path"].sort()
        assert len(self.data["contact_path"]) == len(self.data["scene_tsdf_path"])
        assert len(self.data["contact_path"]) == len(self.data["scene_voxel_path"])
        assert len(self.data["contact_path"]) == len(self.data["wrench_path"])
        if not tqdm_disable: print("Finish Read Dataset scenes size: {0} at: {1}".format(len(self.data["contact_path"]), self.root))

    def __len__(self):
        return len(self.data["contact_path"])
    
    def __getitem__(self, i):
        if self.grid_type == 'tsdf':
            scene_path = self.data["scene_tsdf_path"][i] 
            scene_data = utils_dataio.read_tsdf_grid(scene_path, scene_id=None, ann_id=None) # [1, 80, 80, 80]
        elif self.grid_type == 'voxel':
            scene_path = self.data["scene_voxel_path"][i]
            scene_data = utils_dataio.read_voxel_grid(scene_path, scene_id=None, ann_id=None) # [1, 80, 80, 80]
        else:
            raise KeyError(" data.grid_type not implemented: {}".format(self.grid_type))
        contact_path = self.data["contact_path"][i]
        wrench_path = self.data["wrench_path"][i]
        df_contacts = utils_dataio.read_df(contact_path, scene_id=None, ann_id=None, name=None)
        # df_contacts['count', 'mean_score', 'mean_qx', 'mean_qy', 'mean_qz', 'mean_qw', 'tip_label', 'weighted_socre']
        df_wrenches = utils_dataio.read_df(wrench_path, scene_id=None, ann_id=None, name=None)
        # df_wrenches['count', 'mean_score']

        df_contact_index = df_contacts.index
        indexs_contact = GraspType.index_str2nums(df_contact_index, is_array=True).astype(np.uint16)
        scores_contact = df_contacts['weighted_score'].to_numpy()
        rotations_contact = df_contacts[['mean_qx', 'mean_qy', 'mean_qz', 'mean_qw']].to_numpy()
        
        df_wrench_index = df_wrenches.index
        indexs_wrench = GraspType.index_str2nums(df_wrench_index, is_array=True).astype(np.uint16)
        scores_wrench = df_wrenches['mean_score'].to_numpy()

        if self.data_type == 'Full':
            # Above is quat data
            if self.orientation_type == "quat": 
                std_rotation = [0, 0, 0, 1]
            elif self.orientation_type == "so3": 
                rotations_contact = ut_tranp.quat2so3(rotations_contact)
                std_rotation = [0, 0, 0]
            elif self.orientation_type == "R6d": 
                rotations_contact = ut_tranp.quat2R6d(rotations_contact)
                std_rotation = [0, 0, 1, 0, 1, 0]
            else: raise ValueError("Wrong orientation_type for dataset loading:{}".format(self.orientation_type))
            rotations_contact_com = np.tile(np.zeros_like(scene_data), (len(std_rotation), 1, 1, 1))
            rotations_contact_com[:] = np.array(std_rotation, dtype=np.float32).reshape((len(std_rotation), 1, 1, 1))
            scores_wrench_com = np.zeros_like(scene_data)
            scores_contact_com = np.zeros_like(scene_data)
            scores_contact_com[np.zeros(indexs_contact.shape[0]).astype(np.uint16), indexs_contact[:,0], indexs_contact[:,1], indexs_contact[:,2]] = scores_contact
            rotations_contact_com[:, indexs_contact[:,0], indexs_contact[:,1], indexs_contact[:,2]] = np.swapaxes(rotations_contact, 0, 1)
            scores_wrench_com[np.zeros(indexs_wrench.shape[0]).astype(np.uint16), indexs_wrench[:,0], indexs_wrench[:,1], indexs_wrench[:,2]] = scores_wrench
            
            return scene_data, (scores_contact_com, rotations_contact_com, scores_wrench_com)

        elif self.data_type == 'Indexed':
            ## Tips process
            focal_ratio_contact = 0.8
            num_focal_contact = int(self.numsample * focal_ratio_contact)
            num_nonfocal_surface_contact = int(self.numsample * ((1 - focal_ratio_contact) / 4))
            num_nonfocal_spatial_contact = self.numsample - (num_focal_contact + num_nonfocal_surface_contact)

            index_tsdf_surface = np.array(np.nonzero(scene_data.squeeze())).transpose()
            index_tsdf_spatial = np.argwhere(scene_data.squeeze() == 0)

            indexs_re_surface = np.random.choice(index_tsdf_surface.shape[0], size=index_tsdf_surface.shape[0], replace=False)
            index_tsdf_surface = index_tsdf_surface[indexs_re_surface]
            index_contact_appends = []
            for index_num in index_tsdf_surface:
                str_index = GraspType.index_nums2str(index_num)
                if str_index in df_contact_index: continue
                index_contact_appends.append(index_num)
                if len(index_contact_appends) == num_nonfocal_surface_contact: break

            indexs_re_spatial = np.random.choice(index_tsdf_spatial.shape[0], size=index_tsdf_spatial.shape[0], replace=False)
            index_tsdf_spatial = index_tsdf_spatial[indexs_re_spatial]
            for index_num in index_tsdf_spatial:
                str_index = GraspType.index_nums2str(index_num)
                if str_index in df_contact_index: continue
                index_contact_appends.append(index_num)
                if len(index_contact_appends) == (num_nonfocal_surface_contact + num_nonfocal_spatial_contact): break
            index_contact_appends = np.array(index_contact_appends)

            indexs_contact_focal = np.random.choice(indexs_contact.shape[0], size=num_focal_contact, replace=indexs_contact.shape[0] < num_focal_contact)
            indexs_contact_com = np.vstack([indexs_contact[indexs_contact_focal], index_contact_appends]).astype(np.uint16)
            scores_contact_com = np.hstack([np.clip(scores_contact[indexs_contact_focal], 0.0, 1.0), np.zeros(index_contact_appends.shape[0])])
            rotation_contact_appends = np.tile([0., 0., 0., 1.0], (index_contact_appends.shape[0], 1))
            rotations_contact_com = np.vstack([rotations_contact[indexs_contact_focal], rotation_contact_appends])
            
            reinds_contact = np.random.choice(indexs_contact_com.shape[0], size=indexs_contact_com.shape[0], replace=False)
            indexs_contact_com, scores_contact_com, rotations_contact_com = \
                indexs_contact_com[reinds_contact], scores_contact_com[reinds_contact], rotations_contact_com[reinds_contact]

            # Above is quat data
            if self.orientation_type == "quat": pass # as it is
            elif self.orientation_type == "so3": rotations_contact_com = ut_tranp.quat2so3(rotations_contact_com)
            elif self.orientation_type == "R6d": rotations_contact_com = ut_tranp.quat2R6d(rotations_contact_com)
            else: 
                raise ValueError("Wrong orientation_type for dataset loading:{}".format(self.orientation_type))
            
            ## Wrench process
            focal_ratio_wrench = 0.6
            num_focal_wrench = int(self.numsample/4 * focal_ratio_wrench)
            num_nonfocal_wrench = int(self.numsample/4 - num_focal_wrench)

            indxs_bank_ind = np.random.choice(scene_data.squeeze().shape[0], size=self.numsample * 3)
            indxs_bank = indxs_bank_ind.reshape((self.numsample, 3))
            index_wrench_appends = []
            for index_num in indxs_bank:
                str_index = GraspType.index_nums2str(index_num)
                if str_index in df_wrench_index: continue
                index_wrench_appends.append(index_num)
                if len(index_wrench_appends) >= num_nonfocal_wrench: break
            index_wrench_appends = np.array(index_wrench_appends)
            
            indxs_focal_wrench = np.random.choice(indexs_wrench.shape[0], size=num_focal_wrench, replace=indexs_wrench.shape[0] < num_focal_wrench)
            indexs_wrench_com = np.vstack([indexs_wrench[indxs_focal_wrench], index_wrench_appends]).astype(np.uint16)
            scores_wrench_com = np.hstack([np.clip(scores_wrench[indxs_focal_wrench], 0.0, 1.0), np.zeros(index_wrench_appends.shape[0])])
            
            reinds_wrench = np.random.choice(indexs_wrench_com.shape[0], size=indexs_wrench_com.shape[0], replace=False)
            indexs_wrench_com, scores_wrench_com = indexs_wrench_com[reinds_wrench], scores_wrench_com[reinds_wrench]

            ## All data of input, tips, and wrenches
            x, y, index = scene_data, (scores_contact_com, rotations_contact_com, scores_wrench_com), (indexs_contact_com, indexs_wrench_com)
            return x, y, index


    @staticmethod
    def _numpy_to_torch(data):
        return torch.from_numpy(data.astype(np.float32))


    @staticmethod
    def collate_fn_concatenate(batch):
        input = torch.from_numpy(np.array([item[0] for item in batch]))
        target_socres = []
        target_rots = []
        target_wrens = []
        indexs_contact = []
        indexs_wrench = []
        id_num = 0
        for item in batch:
            (target_socre, target_rot, target_wren) = item[1]
            target_socres.append(torch.from_numpy(target_socre))
            target_rots.append(torch.from_numpy(target_rot))
            target_wrens.append(torch.from_numpy(target_wren))
            batch_id_contact = np.ones((item[2][0].shape[0], 1)) * id_num
            indexs_contact.append(torch.from_numpy(np.hstack([batch_id_contact, item[2][0]])))
            batch_id_wrench = np.ones((item[2][1].shape[0], 1)) * id_num
            indexs_wrench.append(torch.from_numpy(np.hstack([batch_id_wrench, item[2][1]])))
            id_num += 1

        target_socres = torch.hstack(target_socres)
        target_rots = torch.vstack(target_rots)
        target_wrens = torch.hstack(target_wrens)
        indexs_contact = torch.vstack(indexs_contact).to(torch.int)
        indexs_wrench = torch.vstack(indexs_wrench).to(torch.int)
        target = (target_socres, target_rots, target_wrens)
        indexs = (indexs_contact, indexs_wrench)
        return [input, target, indexs]

    @staticmethod
    def collate_fn_full(batch):
        input = torch.from_numpy(np.array([item[0] for item in batch]))
        target_socres = []
        target_rots = []
        target_wrens = []
        id_num = 0
        for item in batch:
            (target_socre, target_rot, target_wren) = item[1]
            target_socres.append(torch.from_numpy(target_socre).unsqueeze(dim=0))
            target_rots.append(torch.from_numpy(target_rot).unsqueeze(dim=0))
            target_wrens.append(torch.from_numpy(target_wren).unsqueeze(dim=0))
            id_num += 1
        target_socres = torch.vstack(target_socres)
        target_rots = torch.vstack(target_rots)
        target_wrens = torch.vstack(target_wrens)
        target = (target_socres, target_rots, target_wrens)
        return [input, target]



def create_train_val_loaders(root, batch_size, val_split, data_type, kwargs):
    """
    ### TEST CODE ###
    """
    # load the dataset
    dataset = Dataset(root, numsample=5000, orientation_type='R6d', grid_type='voxel', data_type=data_type)
    # split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, 
        collate_fn=Dataset.collate_fn_concatenate if data_type == 'Indexed' else Dataset.collate_fn_full, 
        **kwargs
        )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=True, 
        collate_fn=Dataset.collate_fn_concatenate if data_type == 'Indexed' else Dataset.collate_fn_full, 
        **kwargs
        )
    return train_loader, val_loader


if __name__ == "__main__":
    dataset_root = Path("/home/smarnlab/SpatialHybridGen/spahybgen/dataset")

    use_cuda = torch.cuda.is_available()
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    data_type = 'Full' # Indexed Full
    val_split = 0.999
    batch_size = 4
    dataset = dataset_root
    train_loader, val_loader = create_train_val_loaders(
        dataset, batch_size, val_split, data_type, kwargs
    )
    count_it = 0

    if data_type == 'Indexed':
        for xs, ys, indexs in iter(train_loader):
            count_it += 1
            scores, rots, wrens = ys
            inds_contact, inds_wrench = indexs

            print('{} ==> xs: '.format(count_it), xs.shape, 'inds_contact: ', inds_contact.shape,
                'inds_wrench: ', inds_wrench.shape, 'scores: ', scores.shape, 'rots: ', 
                rots.shape, 'wrens: ', wrens.shape)
    else:
        for xs, ys in iter(train_loader):
            count_it += 1
            scores, rots, wrens = ys
            print('{} ==> xs: '.format(count_it), xs.shape, 'scores: ', scores.shape, 'rots: ', 
                rots.shape, 'wrens: ', wrens.shape)
    