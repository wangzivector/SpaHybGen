# SpaHybGen: Learning Contact Representations in Real-World Clutter for General Robotic Grasping 

SpaHybGen generates grasp poses for general robotic hands in SE(3) clutter scenes using learning and optimization strategies. It uses the real grasping dataset GraspNet-1Billion to train the inference section. SpaHybGen can directly and robustly deploy any robotic hand with its URDF to actual clutter grasping in minutes, using a single depth camera.

> **IMPORTANT**: It is feasible to use your own robotic hands **without** [Contact Dataset Generation](#Contact-Dataset-Generation) and [Contact Training](#Contact-Training). To immediately use your custom robotic hands, please directly refer to [Pipeline: Grasp Generation](#pipeline-grasp-generation) (after setting the [Code Environment](#code-environment)).


<br>

<figure>
  <div align="center">
    <img src="assets/images/pipeline.jpg" width="95%" title="Pipeline of SpaHybGen">
  </div>
  <div align="center">
    <figcaption><b>Scene-Level Spatial Grasp Generation for General Robotic Hands</b></figcaption>
  </div>
</figure>

## Maintain schemes
✅ Autmatic objective-scaling strategy to grasp optimization, replacing constant hand-tuned scales. 
<br>✅ Replaced absolute path and magic number.
<br>✅ Polished contact assignment script.
<br>✅ Broke down functions and formated for clarity - [ongoing]
<br>✅ Add functional documentation and type hints - [ongoing]
- [ ] More in-script modular test [before 1st March 2026]
- [ ] **Task-oriented objectives and manipulation hand primitives** [before April 2026]


## Code Environment
We use Python 3.8 in Conda to train 3D U-Net, infer contacts, and optimize grasps.
All algorithmic procedures are coded in Pytorch and [Pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics). 
Environment setup for the real-world deployment refers to the following [Actual Grasping](#actual-grasping) section.

0. Create virtual env.: 
```bash
conda create --name spahybgen python=3.8
conda activate spahybgen
```
1. Install required packages with `pip` on virtual env or python3:
```bash
pip install -r ./assets/requirements.txt

export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True # for sklearn package error
pip install graspnetAPI # for Graspnet Dataset
```

2. Install the project locally in "editable" mode using pip: 
```bash
pip install -e .
```

> We understand that the setup of conda for GPU-based python packages can be tricky in varied machine and the above instruction may be insufficient for an error-free installation. Thus, we further share the specification of our installed env. in [environment.yml](assets/environment.yml) for reference.


## Contact Dataset Generation
<div align="center">
  <img src="assets/images/dataset_generation.jpg" width="85%" title="dataset_generation">
</div>

We release the generated contact dataset in [Google Drive](https://drive.google.com/drive/folders/1hs88Nh3Kx85hMYPT0tjwxXlCzFibeEXJ?usp=sharing). It includes 4.5GB training data and 4.2GB test data. 

If researchers expect to generate the contact dataset, please download the full [GraspNet-1Billion](https://graspnet.net/datasets.html) dataset and run the following command:
```bash
cd spahybgen
python scripts/generate_dataset.py --graspnet PATH_TO_GraspNet --output ./dataset/train
```
It will take tens of hours for the generation process (currently we have not parallelized it). 

Detailed descriptions of the contact generation process are presented in [scripts/generate_dataset.py](scripts/generate_dataset.py).

The contact dataset should be placed inside a `dataset` folder as: `spahybgen\dataset\train\scene_0000`.

## Contact Training 
<div align="center">
  <img src="assets/images/contact_inference.jpg" width="85%" title="contact_inference">
</div>

After generating or downloading the Contact Dataset in the previous step, run command to train a 3D U-Net:
```bash
python scripts/train_shgn.py --dataset dataset/train/ --net unet --orientation quat --gridtype voxel --batch-size 4 --numsample 3000 --epochs 64 --loaders 10 --gridtype voxel
```

The training logs and models are stored at `data/runs/`.

To facilitate reproduction, two trained models are also shared in [Google Drive](https://drive.google.com/drive/folders/1hs88Nh3Kx85hMYPT0tjwxXlCzFibeEXJ?usp=sharing) and [assets/trained_models/](assets/trained_models/), which contains two networks oriented to Voxel and TSDF input observations.

## Pipeline: Grasp Generation
The process of **grasp generation** includes `1.scene observation`, `2.contact inference`, `3.hand model` and `4.grasp optimization`.

### 1. Scene Observation
We enable two formats (Voxel and TSDF) as the scene observation in actual grasping tasks. To obtain observation, one can refer to the generated (downloaded) contact dataset in folder `spahybgen/dataset/`, where the `.npz` files are observations for grasping scenes in GraspNet-1Billion. 
Alternatively, practitioners capture scene volumes using a depth sensor, following the sensing pipeline at [src/spahybgen/pipeline/sensor_server.py](src/spahybgen/pipeline/sensor_server.py). 
> Two observation samples can be found in [assets/observations/](assets/observations/). You can load them with `np.load('assets/observations/scene_010_ann_0124_voxel.npz')["grid"]`.

### 2. Contact Inference
With the trained model and obtained observation, dense contact features can be reasoned before grasp optimization.

Note: If you want to individually test the `contact inference` module, please run:
```bash
python scripts/contact_inference_test.py
```

It will infer contact features using the observation `scene_010_ann_0124_voxel.npz` and model `spahybgen_unet_64_voxel.pt` in folder [assets/](assets/).

### 3. Hand Model
**More than ten robotic hands** are released in folder [\handmodel](\handmodel).
To construct a custom gripper in compatible format, please check these hand examples. Generally, one hand model can be generated within the following steps:

(1). Prepare the standard URDF file for the targeted robotic hand. The `CAD filepath` and `xml encoding information` in .urdf should be properly modified to match the code (for the targeted format, please refer to the released hand examples).

(2). Assign contact regions to the hand surface using the tool in [scripts/hand_contacts.ipynb](scripts/hand_contacts.ipynb).

(3). Append information of the custom hand to file [handmodel/hand_infos.json](handmodel/hand_infos.json), following the included format.


### 4. Grasp Optimization
With the inferred contact features and established hand model, grasp optimization is parallelized using [Pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics).

After running the contact inference script `contact_inference_test.py` in Step 2, the following cmd will optimize grasps using the specific hand and visualize the results using Web-based Plotly:
```bash 
python scripts/grasp_optimization_test.py --hand robotiq2f --max_iter 120 --batch_size 64
# AVAILABLE HAND MODELS: 
# 2F: [robotiq2f, finray2f, antipodal_hand] 
# 3F: [robotiq3f, softpneu3f] 
# 4F: [finray4f, leaphand] 
# 5F: [brunel_hand]
``` 

Note: If you want to individually test the grasp optimization module, download the `std_inference_result_from_clutter.npy` from [Google Drive](https://drive.google.com/drive/folders/1hs88Nh3Kx85hMYPT0tjwxXlCzFibeEXJ?usp=sharing) to the folder `./assets`, and rename it to `inference_results.npy`.


### **Full Pipeline**
To run the full algorithmic pipeline without hardware (using the default observation file and trained model contained in folder [assets/](assets/)), please directly run:
```bash
python scripts/generation_pipeline_test.py --hand robotiq2f --max_iter 120 --batch_size 64
```
This script is a combination of [2. Contact Inference](#2-contact-inference) and [4. Grasp Optimization](#4-grasp-optimization). Similarly, it will optimize grasps using Robotiq-2F and visualize the results using Web-based Plotly.


## Actual Grasping
<figure>
  <div align="center">
    <img src="assets/images/devices.jpg" width="90%" title="">
  </div>
  <div align="center">
    <figcaption><b>Used devices in actual grasping</b></figcaption>
  </div>
</figure>

- In our actual grasping experiments, ROS Melodic is used to coordinate the UR-5e, Azure RGB-D camera, and multiple robotic hands. To facilitate a fast setup of the actual grasping framework, we released the ROS-based communication interfaces of all devices and algorithm modules, as in the folder [src/spahybgen/pipeline/](src/spahybgen/pipeline/).

- Practitioners are expected to set up the custom hardware with ROS and modify their specific ROS topics in different files (camera: [config/grasp_generation.yaml](config/grasp_generation.yaml); robot arm:[src/spahybgen/pipeline/pose_node.py](src/spahybgen/pipeline/pose_node.py); robotic hands: [src/spahybgen/pipeline/gripper_node.py](src/spahybgen/pipeline/gripper_node.py)). 

- Finally, [scripts/realrobot_execution.py](scripts/realrobot_execution.py) details a single-gripper grasping pipeline which includes hardware execution.

> To enable ROS1 in Python3, please follow [Coding_Instruction](https://github.com/wangzivector/Coding_Instruction/blob/master/ROS_python3.md) to make `import rospy` and `import tf2_ros` working in python3.


## Demonstration
### 1. Semi-cluttered grasping with seven robotic hands

<figure>
  <div align="center">
    <a href="https://www.youtube.com/watch?v=f7hdpRCiMNM">
      <img src="assets/images/general-semi-ver.jpg" width="75%" title="Grasping performance for seven robotic hands">
    </a>
    </div>
    <div align="center">
    <a href="https://www.youtube.com/watch?v=f7hdpRCiMNM">
      <figcaption><b>Video: Semi-cluttered grasping with seven robotic hands and multi-gripper simultaneous grasping</b></figcaption>
    </a>
  </div>
</figure>

### 2. Multi-gripper simultaneous grasping
<figure>
  <div align="center">
    <a href="https://www.youtube.com/watch?v=f7hdpRCiMNM">
      <img src="assets/images/multigraspshardware.jpg" width="75%" title="">
    </a>
  </div>
  <div align="center">
    <a href="https://www.youtube.com/watch?v=f7hdpRCiMNM">
      <figcaption><b>Video: Multi-gripper simultaneous grasping</b></figcaption>
    </a>
  </div>
</figure>


### 3. Dynamic grasp update in dense clutter grasping

<figure>
  <div align="center">
    <a href="https://www.youtube.com/watch?v=SueBvBfRTTg">
      <img src="assets/images/dynamic-hardware-c.jpg" width="75%" title="Dynamic grasping setup">
    </a>
  </div>
  <div align="center">
    <a href="https://www.youtube.com/watch?v=SueBvBfRTTg">
      <img src="assets/images/dynamic-leap.jpg" width="75%" title="Video: Dynamic densely cluttered grasping (Roboitq-3F and LEAP Hand)">
    </a>
  </div>
  <div align="center">
    <a href="https://www.youtube.com/watch?v=SueBvBfRTTg">
      <figcaption><b>Video: Dynamic densely cluttered grasping (Roboitq-3F and LEAP Hand)</b></figcaption>
    </a>
  </div>
</figure>



## Cite
This research is not published.

## Acknowledge
This project is inspired by the excellent works [VGN](https://github.com/ethz-asl/vgn), [GenDexGrasp](https://github.com/tengyu-liu/GenDexGrasp), and [GraspNetAPI](https://github.com/graspnet/graspnetAPI).