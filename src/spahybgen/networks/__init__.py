import torch
from pathlib import Path
from . import vgn, unet_3d

def get_network(name, ntargs):
    models = {
        "vgn": vgn.VGN(**ntargs),
        "unet": unet_3d.UNet3D(**ntargs),
    }
    return models[name.lower()]


def load_network(path, device, ntargs):
    """Construct the neural network and load parameters from the specified file.
    Args:
        path: Path to the model parameters. The name must conform to `net_name_[_...]`.
    """
    model_name = Path(path).stem.split("_")[1]
    net = get_network(model_name, ntargs).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net