import os
from typing import Any
import rospy
from datetime import datetime
from spahybgen.pipeline.param_server import GraspParameter
from spahybgen.pipeline.sensor_server import SensorClient
from spahybgen.pipeline.optimization_node import GraspOptNode
from spahybgen.pipeline.pose_node import PoseNode
from spahybgen.pipeline.gripper_node import GripperNode


def pipeline(cfg_file: str):
    """hardware managemetn pipeline for real world experiment execution

    Args:
        cfg_file: configuration file for hardwares
    """
    GP = GraspParameter(cfg_file, echo=True)

    ## All hardware nodes
    sensor_client = SensorClient(GP.sensor.voxel_disc, GP.sensor.grid_topic, GP.sensor.grid_request_topic)
    rospy.loginfo("[Pipeline]: sensor_client initialized.")

    grasp_node = GraspOptNode(
        GP.inference.ori_type,
        GP.inference.infer_topic,
        GP.sensor.voxel_disc,
        (GP.sensor.grid_length / GP.sensor.voxel_disc),
        GP.optimization.opti_rate,
        GP.robot_a.hand_name,
        GP.optimization.opti_batches,
        GP.optimization.opti_max_iter,
        GP.optimization.hand_scale,
        GP.optimization.tip_downsample,
        GP.optimization.wrench_downsample,
        GP.optimization.penetration_mode,
        GP.sensor.grid_type,
        GP.optimization.pent_check_split,
        GP.optimization.save_optimization,
        GP.optimization.focal_ratio,
    )
    rospy.loginfo("[Pipeline]: grasp_node initialized.")

    gripper_node = GripperNode(GP.robot_a.hand_name)
    rospy.loginfo("[Pipeline]: gripper_node initialized.")

    pose_node = PoseNode(robot_prefix=GP.robot_a.robot_id, camera_prefix=GP.sensor.camera_id)
    pose_node.characterize_grasp_transform(
        armend2gripper=gripper_node.armend2gripper,
        gripper2grip=gripper_node.gripper2grip,
        gripper_tfname=gripper_node.gripper_tfname,
        gripper_tfname_vis=gripper_node.gripper_tfname_vis,
    )
    rospy.loginfo("[Pipeline]: pose_node initialized.")

    ## Warning
    logdir = "real-grasp/{}".format(datetime.now().strftime("%m%d-%H%M%S"))
    os.makedirs(logdir)

    ## Pipeline
    trial_id, succ_count = 0, 0
    while not rospy.is_shutdown():
        if not input_signal("CURR [{}/{}]: Return to Initial Pose [9]:".format(succ_count, trial_id)):
            # Step: Go to Initial Pose for ready to capture image
            pose_node.publish_ur_pose("AJOINT", GP.robot_a.pose_init, repeat=True)
            # Wait Topic TCP Avoid Losing Msg
            rospy.sleep(0.10)

        if not input_signal("Ready to Fetch Image ? :"):
            continue
        sensor_client.request_grid_cmd(cmd="clear")
        grasp_node.clear_inference_buff()
        pose_camera = pose_node.fetch_tf_posrot("grid_ws", "fixed/rgb_camera_link")
        sensor_client.request_grid_cmd(cmd="pose", pose=pose_camera)
        rospy.loginfo("[Pipeline]: Start obtaining grid.")
        sensor_client.request_grid_cmd(cmd=GP.sensor.grid_type)
        rospy.sleep(2.0)
        while not grasp_node.have_inference_buff():
            rospy.logwarn("Optimzation not receive inference result, wait 0.5 seconds")
            rospy.sleep(0.5)
        grasp_pose, grasp_joints = grasp_node.optimization_pipeline(logdir, trial_id=trial_id)
        rospy.loginfo(
            "[Pipeline]: Get grasp_pose and grasp_joints: \n{}; {}".format(grasp_pose, grasp_joints)
        )

        pose_node.publish_dynamic_tf(
            father_frame="grid_ws", child_frame="object", PosRotVec=grasp_pose, add_prefix=(False, True)
        )
        pose_endposupper = pose_node.fetch_tf_posrot("base", "endpos_upper", add_prefix=(True, True))
        pose_objgrip = pose_node.fetch_tf_posrot("base", "object_grip_endpos", add_prefix=(True, True))
        pose_drop = pose_node.fetch_tf_posrot("base", GP.robot_a.pose_drop_id, add_prefix=(True, False))
        pose_drop_midd = pose_node.fetch_tf_posrot("base", "drop_point_midd", add_prefix=(True, False))

        action_flow = [
            {"name": "UR_TO_pose_upper", "type": "APOSE", "value": pose_endposupper},
            {"name": "GRIPPER_INIT", "type": "GRASP", "value": grasp_joints},
            {"name": "UR_TO_pose_grip", "type": "APOSE", "value": pose_objgrip},
            {"name": "GRIPPER_CLOSE", "type": "GRASP", "value": grasp_joints},
            {"name": "UR_RE_pose_upper", "type": "APOSE", "value": pose_endposupper},
            {"name": "UR_RE_pose_drop_midd", "type": "APOSE", "value": pose_drop_midd},
            {"name": "UR_RE_pose_drop", "type": "APOSE", "value": pose_drop},
            {"name": "GRIPPER_OPEN", "type": "GRASP", "value": grasp_joints},
            {"name": "UR_RE_pose_init", "type": "AJOINT", "value": GP.robot_a.pose_init},
        ]
        print("Excute action list:", action_flow)

        ex_ret = excute_graspactions(action_flow, gripper_node, pose_node, GP.robot_a.robot_id)
        if not ex_ret:
            if not input_signal("## Interrupt Trial [{}]: [COUNT]:enter, [RETURN]:nine ##".format(trial_id)):
                continue
        trial_id += 1
        if input_signal("## Trial [{}]: [SUCCESS]:enter, [FAIL]:nine ##".format(trial_id)):
            succ_count += 1

    rospy.loginfo("[Pipeline]: Everything done, spinning...")
    rospy.spin()


def input_signal(reminder: str = "Enter to contine", supple_txt: str = "[0:End, 9:Retn] "):
    """Control function for executing one step

    Args:
        reminder: description of step. Defaults to "Enter to contine".
        supple_txt: supplementary text to display before the reminder. Defaults to "[0:End, 9:Retn] ".

    Returns:
        whether to continue the execution flow, False if return 9, True if return other keys except 0, and exit if return 0.
    """
    a = input(supple_txt + reminder)
    if a == "0":
        rospy.signal_shutdown("Key interrupt down.")
        rospy.sleep(0.2)
        exit(0)
    return a != "9"


def excute_graspactions(action_flow: list, gripper_client: Any, pose_node: Any, action_prefix: str):
    """execution information output and execution

    Args:
        action_flow: multiple action list
        gripper_client: gripper node
        pose_node: node of robot arm
        action_prefix: description of action type

    Returns:
        execution state
    """
    for action in action_flow:
        if not input_signal(
            "[{}]: ".format(action_prefix)
            + "\n"
            + action["type"]
            + " "
            + action["name"]
            + " {} Execute ? :".format(action["value"])
        ):
            return False
        if action["type"] == "GRASP":
            gripper_client.grasp_execution(action["name"], action["value"])
        if action["type"] in ["APOSE", "AJOINT"]:
            pose_node.publish_ur_pose(action["type"], action["value"], repeat=True)
    return True


if __name__ == "__main__":
    import sys, signal

    ## Ctrl-C stop stuff
    def signal_handler(signal, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    if len(sys.argv) == 1:
        rospy.logwarn("Using default config file.")
    if len(sys.argv) > 2:
        raise KeyError("Too much args, only config file is needed.")
    cfg_file = str(sys.argv[1]) if len(sys.argv) == 2 else "./config/grasp_generation.yaml"

    rospy.init_node("grasp_pipeline_node" + cfg_file[-10:-5])
    pipeline(cfg_file)
