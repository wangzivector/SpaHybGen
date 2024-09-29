import yaml

class ParamX:
    def __init__(self, yaml) -> None:
        for k, v in yaml.items():
            setattr(self, k, v)


class GraspParameter:
    def __init__(self, yaml_path = "./config/grasp_generation.yaml", echo=False) -> None:
        with open(yaml_path, 'r') as file: yaml_params = yaml.safe_load(file)

        self.sensor = ParamX(yaml_params['sensor'])
        self.inference = ParamX(yaml_params['inference'])
        self.optimization = ParamX(yaml_params['optimization'])
        self.robot_a = ParamX(yaml_params['robot_a'])
        print("[Parameter]: Loaded yaml parameters from {}.".format(yaml_path))
        if echo:
            print("[Parameter]: Echo yaml parameters:")
            print(yaml_params)
            print('[Parameter]: Finished')

