# Inherent from [VGN](https://github.com/ethz-asl/vgn)

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

from torch.utils import tensorboard
import torch.nn.functional as F
from torchsummary import summary


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": args.loaders, "pin_memory": True} if use_cuda else {}
    ntargs = {"voxel_discreteness": 80, "orientation": args.orientation, "augment":args.augment}

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
        args.dataset, args.batch_size, args.val_split, args.numsample, args.orientation, args.gridtype, args.datatype, kwargs
    )
    
    # build the network
    net = get_network(args.net, ntargs).to(device)
    # visulize network
    summary(net, (1, ntargs["voxel_discreteness"], ntargs["voxel_discreteness"], ntargs["voxel_discreteness"]))

    # define optimizer and metrics
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=1/2) # args.epochs//5

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
    trainer = create_trainer(net, optimizer, metrics, device, loss_fn, args.fn_score, args.orientation, args.fn_wrench, args.datatype)
    evaluator = create_evaluator(net, eval_metrics, device, loss_fn, args.fn_score, args.orientation, args.fn_wrench, args.datatype)

    # log training progress to the terminal and tensorboard
    ProgressBar(persist=True, ascii=True).attach(trainer)

    data_writer = create_summary_writers_simple(net, device, logdir)
    
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_train_process(engine):
        output, it = trainer.state.output, trainer.state.iteration
        length_ita = 6400 // trainer.state.epoch_length
        data_writer.add_scalar("loss_score_process", output[4], it * length_ita)
        data_writer.add_scalar("loss_rot_process", output[5], it * length_ita)
        data_writer.add_scalar("loss_wrench_process", output[6], it * length_ita)


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
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=8), checkpoint_handler_tens, {args.net: net}
    )

    # run the training loop
    trainer.run(train_loader, max_epochs=args.epochs)


def create_train_val_loaders(root, batch_size, val_split, numsample, orientation, grid_type, data_type, kwargs):
    # load the dataset
    dataset = Dataset(root, numsample=numsample, orientation_type=orientation, grid_type=grid_type, data_type=data_type)
    # split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, 
        collate_fn=Dataset.collate_fn_concatenate if data_type == 'Indexed' else Dataset.collate_fn_full,
        **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=True, 
        collate_fn=Dataset.collate_fn_concatenate if data_type == 'Indexed' else Dataset.collate_fn_full,
        **kwargs)
    return train_loader, val_loader


def create_trainer(net, optimizer, metrics, device, loss_fn, fn_score, fn_rot, fn_wrench, datatype):
    def _update(_, batch):
        net.train()
        optimizer.zero_grad()

        # forward
        if datatype == 'Indexed':
            x, y, index = prepare_batch_concatenate(batch, device, datatype)
            y_pred = select_concatenate(net(x), index)
        elif datatype == 'Full':
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


def create_evaluator(net, metrics, device, loss_fn, fn_score, fn_rot, fn_wrench, datatype):
    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            # forward
            if datatype == 'Indexed':
                x, y, index = prepare_batch_concatenate(batch, device, datatype)
                y_pred = select_concatenate(net(x), index)
            elif datatype == 'Full':
                x, y = prepare_batch_concatenate(batch, device, datatype)
                y_pred = net(x)

            loss, loss_score, loss_rot, loss_wrench = loss_fn(y_pred, y, fn_score, fn_rot, fn_wrench)
        return x, y_pred, y, loss, loss_score, loss_rot, loss_wrench

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def prepare_batch_concatenate(batch, device, datatype):
    if datatype == 'Indexed':
        tsdf, (scores, rotations, wrenches), (indexs_contact, indexs_wrench) = batch
        tsdf = tsdf.to(device)
        scores = scores.float().to(device)
        rotations = rotations.to(device)
        wrenches = wrenches.float().to(device)
        indexs_contact = indexs_contact.to(torch.long).to(device)
        indexs_wrench = indexs_wrench.to(torch.long).to(device)
        # tsdf.shape:  torch.Size([32, 1, 80, 80, 80])
        return tsdf, (scores, rotations, wrenches), (indexs_contact, indexs_wrench)
    elif datatype == 'Full':
        tsdf, (scores, rotations, wrenches) = batch
        tsdf = tsdf.to(device)
        scores = scores.float().to(device)
        rotations = rotations.to(device)
        wrenches = wrenches.float().to(device)
        return tsdf, (scores, rotations, wrenches)


def select_concatenate(out, index):
    score_out, rot_out, wrench_out = out
    contact_indexs, wrench_indexs = index
    score = score_out[contact_indexs[:, 0], :, contact_indexs[:, 1], contact_indexs[:, 2], contact_indexs[:, 3]].squeeze()
    rot = rot_out[contact_indexs[:, 0], :, contact_indexs[:, 1], contact_indexs[:, 2], contact_indexs[:, 3]]
    wrench = wrench_out[wrench_indexs[:, 0], :, wrench_indexs[:, 1], wrench_indexs[:, 2], wrench_indexs[:, 3]].squeeze()
    return score, rot, wrench


def loss_fn(y_pred, y, fn_score, fn_rot, fn_wrench):
    scores, rotations, wrenches = y
    score_pred, rotation_pred, wrench_pred = y_pred
    loss_score = _qual_loss_fn(score_pred, scores, fn_score)
    loss_rot, loss_rot_moni = _rot_loss_fn(rotation_pred, rotations, fn_rot)
    loss_wrench = _wrench_loss_fn(wrench_pred, wrenches, fn_wrench)
    if len(scores.shape) != len(loss_rot.shape):
        loss_rot = loss_rot.unsqueeze(dim=1)

    ## original
    loss = loss_score.mean() + (scores*loss_rot).mean() + loss_wrench.mean()
    return loss, torch.abs(score_pred - scores).mean(), (scores * loss_rot_moni).mean(), torch.abs(wrench_pred - wrenches).mean()


def _qual_loss_fn(pred, target, loss_fn_name="CEL"):
    if loss_fn_name == "FCL": 
        alpha, gamma, eps = 1, 1, 1e-6
        dis_soft = torch.abs(pred - target)
        focal_loss = -1 * alpha * dis_soft ** gamma * torch.log((1.0 - dis_soft) + eps)
        return focal_loss

    elif loss_fn_name == "MSEL": 
        return F.mse_loss(pred, target, reduction="none")

    elif loss_fn_name == "CEL": 
        return F.binary_cross_entropy(pred, target, reduction="none")


def _wrench_loss_fn(pred, target, loss_fn_name="CEL"):
    if loss_fn_name == "FCL": 
        alpha, gamma, eps = 1, 1, 1e-6
        dis_soft = torch.abs(pred - target)
        focal_loss = -1 * alpha * dis_soft ** gamma * torch.log((1.0 - dis_soft) + eps)
        return focal_loss

    elif loss_fn_name == "MSEL": 
        return F.mse_loss(pred, target, reduction="none")

    elif loss_fn_name == "CEL":
        return F.binary_cross_entropy(pred, target, reduction="none")


def _rot_loss_fn(pred, target, loss_fn_name="quat"):
    if loss_fn_name == "quat":
        return _quat_loss_fn(pred, target)
    if loss_fn_name == "so3":
        return _so3_loss_fn(pred, target)
    if loss_fn_name == "R6d":
        return _R6d_loss_fn(pred, target)


def _quat_loss_fn(pred, target):
    loss_q = 1.0 - torch.abs(torch.sum(pred * target, dim=1))
    return loss_q, loss_q


def _so3_loss_fn(pred, target):
    loss_so3_abs = torch.abs(pred - target).mean(dim=1)
    return loss_so3_abs, loss_so3_abs


def _R6d_loss_fn(pred, target):
    loss_R6d_abs = torch.abs(pred - target).mean(dim=1)
    return loss_R6d_abs, loss_R6d_abs


def create_summary_writers_simple(net, device, log_dir):
    logdata_path = log_dir / "logdata"
    logdata_writer = tensorboard.SummaryWriter(logdata_path, flush_secs=30)
    return logdata_writer


def create_summary_writers(net, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=30)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=30)

    return train_writer, val_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="vgn")
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
