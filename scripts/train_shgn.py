# Inherent from [VGN](https://github.com/ethz-asl/vgn)

from typing import Any, Tuple
import argparse
from pathlib import Path
from datetime import datetime
from spahybgen.dataset import Dataset
from spahybgen.networks import get_network

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Average
import torch
import torch.utils.data
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F
from torchsummary import summary


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": args.loaders, "pin_memory": True} if use_cuda else {}
    ntargs = {"voxel_discreteness": 80, "orientation": args.orientation, "augment": args.augment}

    # create log directory
    time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    description = "{},net={},batch={},samp={},loss={}={}={},lr={:.0e},aug={},{},{},{}".format(
        time_stamp,
        args.net,
        args.batch_size,
        args.numsample,
        args.fn_score,
        args.orientation,
        args.fn_wrench,
        args.lr,
        args.augment,
        args.gridtype,
        args.datatype,
        args.description,
    ).strip(",")
    logdir = args.logdir / description
    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args.dataset,
        args.batch_size,
        args.val_split,
        args.numsample,
        args.orientation,
        args.gridtype,
        args.datatype,
        kwargs,
    )

    # build the network
    net = get_network(args.net, ntargs).to(device)
    # visulize network
    summary(
        net, (1, ntargs["voxel_discreteness"], ntargs["voxel_discreteness"], ntargs["voxel_discreteness"])
    )

    # define optimizer and metrics
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=1 / 2)  # args.epochs//5

    metrics = {
        "loss": Average(lambda out: out[3]),
        "MeanAbsoluteError_score": Average(lambda out: out[4]),
        "MeanAbsoluteError_rotate": Average(lambda out: out[5]),
        "MeanAbsoluteError_wrench": Average(lambda out: out[6]),
    }
    eval_metrics = {
        "loss": Average(lambda out: out[3]),
        "MeanAbsoluteError_score": Average(lambda out: out[4]),
        "MeanAbsoluteError_rotate": Average(lambda out: out[5]),
        "MeanAbsoluteError_wrench": Average(lambda out: out[6]),
    }

    # create ignite engines for training and validation
    trainer = create_trainer(
        net,
        optimizer,
        metrics,
        device,
        loss_fn,
        args.fn_score,
        args.orientation,
        args.fn_wrench,
        args.datatype,
    )
    evaluator = create_evaluator(
        net, eval_metrics, device, loss_fn, args.fn_score, args.orientation, args.fn_wrench, args.datatype
    )

    # log training progress to the terminal and tensorboard
    ProgressBar(persist=True, ascii=True).attach(trainer)

    data_writer = create_summary_writers_simple(net, device, logdir)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_train_process(engine):
        """Debug logger for training process

        Args:
            engine: engine
        """
        output, it = trainer.state.output, trainer.state.iteration
        length_ita = len(trainer.state.dataloader) // trainer.state.epoch_length  # type: ignore
        data_writer.add_scalar("loss_score_process", output[4], it * length_ita)  # type: ignore
        data_writer.add_scalar("loss_rot_process", output[5], it * length_ita)  # type: ignore
        data_writer.add_scalar("loss_wrench_process", output[6], it * length_ita)  # type: ignore

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine):
        epoch, metrics = trainer.state.epoch, trainer.state.metrics
        data_writer.add_scalar("loss", metrics["loss"], epoch)
        data_writer.add_scalar("MeanAbsoluteError_score", metrics["MeanAbsoluteError_score"], epoch)
        data_writer.add_scalar("MeanAbsoluteError_rotate", metrics["MeanAbsoluteError_rotate"], epoch)
        data_writer.add_scalar("MeanAbsoluteError_wrench", metrics["MeanAbsoluteError_wrench"], epoch)
        scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        epoch, metrics = trainer.state.epoch, evaluator.state.metrics
        data_writer.add_scalar("val_loss", metrics["loss"], epoch)
        data_writer.add_scalar("val_MeanAbsoluteError_score", metrics["MeanAbsoluteError_score"], epoch)
        data_writer.add_scalar("val_MeanAbsoluteError_rotate", metrics["MeanAbsoluteError_rotate"], epoch)
        data_writer.add_scalar("val_MeanAbsoluteError_wrench", metrics["MeanAbsoluteError_wrench"], epoch)

    # checkpoint model
    gst = lambda *_: trainer.state.epoch
    checkpoint_handler_tens = ModelCheckpoint(
        logdir,
        "spahybgen",
        n_saved=100,
        global_step_transform=gst,
        require_empty=True,
        save_as_state_dict=True,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=8), checkpoint_handler_tens, {args.net: net})

    # run the training loop
    trainer.run(train_loader, max_epochs=args.epochs)


def create_train_val_loaders(
    root: Path,
    batch_size: int,
    val_split: float,
    numsample: int,
    orientation: str,
    grid_type: str,
    data_type: str,
    kwargs,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """create training and validation data loaders
    Args:
        root: root directory of the dataset
        batch_size: batch size for training and validation
        val_split: ratio of validation set split from the whole dataset
        numsample: number of samples to be drawn from each scene in each epoch
        orientation: representation of the output orientation,
                        can be "quat", "so3" or "R6d"
        grid_type: type of input grid, can be "voxel" or "tsdf"
        data_type: type of data, can be "Indexed" or "Full"
        kwargs: additional arguments for DataLoader
    Returns:
        train_loader, val_loader
    """
    # load the dataset
    dataset = Dataset(
        root, numsample=numsample, orientation_type=orientation, grid_type=grid_type, data_type=data_type
    )
    # split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=Dataset.collate_fn_concatenate if data_type == "Indexed" else Dataset.collate_fn_full,
        **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=Dataset.collate_fn_concatenate if data_type == "Indexed" else Dataset.collate_fn_full,
        **kwargs
    )
    return train_loader, val_loader


def create_trainer(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: dict,
    device: torch.device,
    loss_fn: Any,
    fn_score: str,
    fn_rot: str,
    fn_wrench: str,
    datatype: str,
) -> Engine:
    """create ignite trainer for training the network
    Args:
        net: the network to be trained
        optimizer: optimizer for training
        metrics: dictionary of metrics to be logged during training
        device: device for training
        loss_fn: loss function for training
        fn_score: loss function for score prediction, can be "CEL", "MSEL"
        fn_rot: loss function for rotation prediction, can be "quat", "so3" or "R6d"
        fn_wrench: loss function for wrench prediction, can be "CEL", "MSEL"
        datatype: type of data, can be "Indexed" or "Full"
    Returns:
        trainer
    """

    def _update(_, batch):
        net.train()
        optimizer.zero_grad()

        # forward
        if datatype == "Indexed":
            x, y, index = prepare_batch_concatenate(batch, device, datatype)
            y_pred = select_concatenate(net(x), index)
        elif datatype == "Full":
            x, y = prepare_batch_concatenate(batch, device, datatype)
            y_pred = net(x)

        loss, loss_score, loss_rot, loss_wrench = loss_fn(y_pred, y, fn_score, fn_rot, fn_wrench)
        # backward
        loss.backward()
        optimizer.step()

        return x, y_pred, y, loss, loss_score, loss_rot, loss_wrench

    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer


def create_evaluator(
    net: torch.nn.Module,
    metrics: dict,
    device: torch.device,
    loss_fn: Any,
    fn_score: str,
    fn_rot: str,
    fn_wrench: str,
    datatype: str,
) -> Engine:
    """create ignite evaluator for evaluating the network
    Args:
        net: the network to be evaluated
        metrics: dictionary of metrics to be logged during evaluation
        device: device for evaluation
        loss_fn: loss function for evaluation
        fn_score: loss function for score prediction, can be "CEL", "MSEL"
        fn_rot: loss function for rotation prediction, can be "quat", "so3" or "R6d"
        fn_wrench: loss function for wrench prediction, can be "CEL", "MSEL"
        datatype: type of data, can be "Indexed" or "Full"
    Returns:
        evaluator
    """

    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            # forward
            if datatype == "Indexed":
                x, y, index = prepare_batch_concatenate(batch, device, datatype)
                y_pred = select_concatenate(net(x), index)
            elif datatype == "Full":
                x, y = prepare_batch_concatenate(batch, device, datatype)
                y_pred = net(x)

            loss, loss_score, loss_rot, loss_wrench = loss_fn(y_pred, y, fn_score, fn_rot, fn_wrench)
        return x, y_pred, y, loss, loss_score, loss_rot, loss_wrench

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def prepare_batch_concatenate(batch: tuple, device: torch.device, datatype: str) -> Any:
    """convert batch data to neat matrix

    Args:
        batch: original batch data from dataloader
        device: device for training or evaluation
        datatype: type of data, can be "Indexed" or "Full"

    Returns:
        out:
        - x: input data for the network
        - y: ground truth for the network output
        - index: index for selecting the output of the network when datatype is "Indexed"
    """
    if datatype == "Indexed":
        tsdf, (scores, rotations, wrenches), (indexs_contact, indexs_wrench) = batch
        tsdf = tsdf.to(device)
        scores = scores.float().to(device)
        rotations = rotations.to(device)
        wrenches = wrenches.float().to(device)
        indexs_contact = indexs_contact.to(torch.long).to(device)
        indexs_wrench = indexs_wrench.to(torch.long).to(device)
        # tsdf.shape:  torch.Size([32, 1, 80, 80, 80])
        return tsdf, (scores, rotations, wrenches), (indexs_contact, indexs_wrench)
    elif datatype == "Full":
        tsdf, (scores, rotations, wrenches) = batch
        tsdf = tsdf.to(device)
        scores = scores.float().to(device)
        rotations = rotations.to(device)
        wrenches = wrenches.float().to(device)
        return tsdf, (scores, rotations, wrenches)


def select_concatenate(out: Tuple, index: Tuple) -> Tuple[Tensor, Tensor, Tensor]:
    """select the output of the network according to the index when datatype is "Indexed"

    Args:
        out: output of the network, including score, rotation and wrench
        index: index for selecting the output of the network, including contact index for
        score and rotation, and wrench index for wrench

    Returns:
        score, rotation and wrench predictions selected from the network output
    """
    score_out, rot_out, wrench_out = out
    contact_indexs, wrench_indexs = index
    score = score_out[
        contact_indexs[:, 0], :, contact_indexs[:, 1], contact_indexs[:, 2], contact_indexs[:, 3]
    ].squeeze()
    rot = rot_out[contact_indexs[:, 0], :, contact_indexs[:, 1], contact_indexs[:, 2], contact_indexs[:, 3]]
    wrench = wrench_out[
        wrench_indexs[:, 0], :, wrench_indexs[:, 1], wrench_indexs[:, 2], wrench_indexs[:, 3]
    ].squeeze()
    return score, rot, wrench


def loss_fn(
    y_pred: Tuple[Tensor, Tensor, Tensor],
    y: Tuple[Tensor, Tensor, Tensor],
    fn_score: str,
    fn_rot: str,
    fn_wrench: str,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """compute the loss for training and evaluation

    Args:
        y_pred: output of the network, including score, rotation and wrench predictions
        y: ground truth for the network output, including score, rotation and wrench
        fn_score: loss function for score prediction, can be "CEL", "MSEL"
        fn_rot: loss function for rotation prediction, can be "quat", "so3" or "R6d"
        fn_wrench: loss function for wrench prediction, can be "CEL", "MSEL"

    Returns:
        out:
        - loss: total loss for training or evaluation
        - loss_score: loss for score prediction
        - loss_rot: loss for rotation prediction
        - loss_wrench: loss for wrench prediction
    """
    scores, rotations, wrenches = y
    score_pred, rotation_pred, wrench_pred = y_pred
    loss_score = _qual_loss_fn(score_pred, scores, fn_score)
    loss_rot, loss_rot_moni = _rot_loss_fn(rotation_pred, rotations, fn_rot)
    loss_wrench = _wrench_loss_fn(wrench_pred, wrenches, fn_wrench)
    if len(scores.shape) != len(loss_rot.shape):
        loss_rot = loss_rot.unsqueeze(dim=1)

    ## original
    loss = loss_score.mean() + (scores * loss_rot).mean() + loss_wrench.mean()
    return (
        loss,
        torch.abs(score_pred - scores).mean(),
        (scores * loss_rot_moni).mean(),
        torch.abs(wrench_pred - wrenches).mean(),
    )


def _qual_loss_fn(pred: Tensor, target: Tensor, loss_fn_name: str = "CEL") -> Tensor:
    """compute the loss for score prediction

    Args:
        pred: score prediction from the network
        target: ground truth for the score
        loss_fn_name: loss function for score prediction, can be "CEL", "MSEL"

    Returns:
        loss for score prediction
    """
    if loss_fn_name == "FCL":
        alpha, gamma, eps = 1, 1, 1e-6
        dis_soft = torch.abs(pred - target)
        focal_loss = -1 * alpha * dis_soft**gamma * torch.log((1.0 - dis_soft) + eps)
        return focal_loss

    elif loss_fn_name == "MSEL":
        return F.mse_loss(pred, target, reduction="none")

    elif loss_fn_name == "CEL":
        return F.binary_cross_entropy(pred, target, reduction="none")
    else:
        raise ValueError("Unknown loss function for score prediction.")


def _wrench_loss_fn(pred: Tensor, target: Tensor, loss_fn_name: str = "CEL") -> Tensor:
    """compute the loss for wrench prediction

    Args:
        pred: wrench prediction from the network
        target: ground truth for the wrench
        loss_fn_name: loss function for wrench prediction, can be "CEL", "MSEL

    Returns:
        loss for wrench prediction
    """
    if loss_fn_name == "FCL":
        alpha, gamma, eps = 1, 1, 1e-6
        dis_soft = torch.abs(pred - target)
        focal_loss = -1 * alpha * dis_soft**gamma * torch.log((1.0 - dis_soft) + eps)
        return focal_loss

    elif loss_fn_name == "MSEL":
        return F.mse_loss(pred, target, reduction="none")

    elif loss_fn_name == "CEL":
        return F.binary_cross_entropy(pred, target, reduction="none")
    else:
        raise ValueError("Unknown loss function for wrench prediction.")


def _rot_loss_fn(pred: Tensor, target: Tensor, loss_fn_name: str = "quat") -> Tuple[Tensor, Tensor]:
    """compute the loss for rotation prediction

    Args:
        pred: rotation prediction from the network
        target: ground truth for the rotation
        loss_fn_name: loss function for rotation prediction, can be "quat", "so3" or "R6d"

    Returns:
        loss for rotation prediction, and the loss for monitoring rotation prediction
    """
    if loss_fn_name == "quat":
        return _quat_loss_fn(pred, target)
    if loss_fn_name == "so3":
        return _so3_loss_fn(pred, target)
    if loss_fn_name == "R6d":
        return _R6d_loss_fn(pred, target)
    else:
        raise ValueError("Unknown loss function for rotation prediction.")


def _quat_loss_fn(pred: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """compute the loss for quaternion rotation prediction

    Args:
        pred: quaternion rotation prediction from the network, with shape [batch_size, 4]
        target: ground truth for the quaternion rotation, with shape [batch_size, 4]

    Returns:
        loss for quaternion rotation prediction, and the loss for monitoring quaternion
    """
    loss_q = 1.0 - torch.abs(torch.sum(pred * target, dim=1))
    return loss_q, loss_q


def _so3_loss_fn(pred: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """compute the loss for so3 rotation prediction

    Args:
        pred: so3 rotation prediction from the network, with shape [batch_size, 3]
        target: ground truth for the so3 rotation, with shape [batch_size, 3]

    Returns:
        loss for so3 rotation prediction, and the loss for monitoring so3
    """
    loss_so3_abs = torch.abs(pred - target).mean(dim=1)
    return loss_so3_abs, loss_so3_abs


def _R6d_loss_fn(pred: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """compute the loss for R6d rotation prediction

    Args:
        pred: R6d rotation prediction from the network, with shape [batch_size, 6]
        target: ground truth for the R6d rotation, with shape [batch_size, 6]

    Returns:
        loss for R6d rotation prediction, and the loss for monitoring R6d
    """
    loss_R6d_abs = torch.abs(pred - target).mean(dim=1)
    return loss_R6d_abs, loss_R6d_abs


def create_summary_writers_simple(net: torch.nn.Module, device: torch.device, log_dir: Path) -> SummaryWriter:
    """create a simple summary writer for logging training and validation results to tensorboard

    Args:
        net: the network to be trained
        device: device for training
        log_dir: directory for logging
    Returns:
        summary writer
    """
    logdata_path = log_dir / "logdata"
    logdata_writer = SummaryWriter(logdata_path, flush_secs=30)
    return logdata_writer


def create_summary_writers(
    net: torch.nn.Module, device: torch.device, log_dir: Path
) -> Tuple[SummaryWriter, SummaryWriter]:
    """create summary writers for logging training and validation results to tensorboard

    Args:
        net: the network to be trained
        device: device for training
        log_dir: directory for logging

    Returns:
        train_writer, val_writer
    """
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = SummaryWriter(train_path, flush_secs=30)
    val_writer = SummaryWriter(val_path, flush_secs=30)

    return train_writer, val_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="unet")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, default="data/runs")
    parser.add_argument("--description", type=str, default="default")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fn-score", type=str, default="CEL")
    parser.add_argument("--orientation", type=str, default="quat")
    parser.add_argument("--fn-wrench", type=str, default="CEL")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--numsample", type=int, default=2000)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--loaders", type=int, default=10)
    parser.add_argument("--gridtype", type=str, default="voxel")
    parser.add_argument("--datatype", type=str, default="Indexed")
    args = parser.parse_args()
    main(args)
