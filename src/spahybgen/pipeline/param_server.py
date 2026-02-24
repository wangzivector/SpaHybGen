from typing import Any
import yaml


class ParamX:
    """A simple class to convert a dictionary to an object with attributes,
    which can be used to store parameters in a config file
    """

    def __init__(self, yaml: dict) -> None:
        """Initialize the ParamX object by setting its attributes according to the input dictionary
        Args:
            yaml: a dictionary containing the parameters to be stored as attributes of the object
        """
        for k, v in yaml.items():
            setattr(self, k, v)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)


class GraspParameter:
    """A class to store the parameters for the grasp generation pipeline,
    which can be loaded from a yaml config file
    The parameters are organized into four groups: sensor, inference, optimization and robot_a,
    which can be accessed as attributes of the object, e.g. param.sensor, param.inference, etc.

    Args:
        yaml_path: the path to the yaml config file containing the parameters
        echo: whether to print the loaded parameters for verification
    """

    def __init__(self, yaml_path: str = "./config/grasp_generation.yaml", echo: bool = False) -> None:

        with open(yaml_path, "r") as file:
            yaml_params = yaml.safe_load(file)

        self.sensor = ParamX(yaml_params["sensor"])
        self.inference = ParamX(yaml_params["inference"])
        self.optimization = ParamX(yaml_params["optimization"])
        self.robot_a = ParamX(yaml_params["robot_a"])
        print("[Parameter]: Loaded yaml parameters from {}.".format(yaml_path))
        if echo:
            print("[Parameter]: Echo yaml parameters:")
            print(yaml_params)
            print("[Parameter]: Finished")
