import torch
from pathlib import Path
from spahybgen.dataset import Dataset
import torch.utils.data


def test_contact_dataset_loading():
    """Create the train and validation dataloaders for the dataset"""
    # Parameters:
    # root: The root directory of the dataset
    # batch_size: The batch size for the dataloaders
    # val_split: The ratio of the validation set size to the whole dataset size
    # data_type: The type of the output data. Can be "Indexed" or "Full"
    # kwargs: Additional keyword arguments for the dataloaders, such as num_workers and pin_memory
    use_cuda = torch.cuda.is_available()
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    val_split = 0.9
    batch_size = 4
    disc = 80
    ori_type = "R6d"
    ori_bits = 6
    numsample = 5000
    ratio_pose_wrench = 0.25
    grid_type = "voxel"
    root = Path("./dataset")

    for data_type in ["Full", "Indexed"]:
        dataset = Dataset(
            root,
            numsample=numsample,
            orientation_type=ori_type,
            grid_type=grid_type,
            data_type=data_type,
            ratio_gt_pose_to_wrench=ratio_pose_wrench,
        )
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=Dataset.collate_fn_concatenate if data_type == "Indexed" else Dataset.collate_fn_full,
            **kwargs,
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=Dataset.collate_fn_concatenate if data_type == "Indexed" else Dataset.collate_fn_full,
            **kwargs,
        )
        assert len(train_loader) == train_size // batch_size
        assert len(val_loader) == val_size // batch_size

        if data_type == "Full":
            for xs, ys in iter(train_loader):
                scores, rots, wrens = ys
                assert xs.shape == (batch_size, 1, disc, disc, disc)
                assert scores.shape == (batch_size, 1, disc, disc, disc)
                assert rots.shape == (batch_size, ori_bits, disc, disc, disc)
                assert wrens.shape == (batch_size, 1, disc, disc, disc)
                break

        elif data_type == "Indexed":
            for xs, ys, indexs in iter(train_loader):
                scores, rots, wrens = ys
                inds_contact, inds_wrench = indexs
                assert xs.shape == (batch_size, 1, disc, disc, disc)
                assert scores.shape == (batch_size * numsample,)
                assert rots.shape == (
                    batch_size * numsample,
                    ori_bits,
                )
                assert wrens.shape == (batch_size * numsample * ratio_pose_wrench,)
                assert inds_contact.shape == (batch_size * numsample, 4)
                assert inds_wrench.shape == (batch_size * numsample * ratio_pose_wrench, 4)
                break
