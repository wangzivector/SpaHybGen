## Gripper operation node
import math
import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int16MultiArray
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import JointState


"""
Gripper Client Node
"""
class GripperNode:
    def __init__(self, gripper):
        """
        Grippers: "robotiq2f", "robotiq3f", "finray2f", "finray4f", "softpneu3f", "brunelhand"
        """
        self.gripper = gripper

        ServerHint = {
            "robotiq2f": "roslaunch robotiq_2f_gripper_control external_robotiq_msgctl.launch",
            "robotiq3f": "roslaunch robotiq_3f_gripper_control robotiq_3f_gripper_TCP_nodes.launch",
            "finray2f": "rosrun rosserial_python serial_node.py /dev/ttyUSB0",
            "finray4f": "rosrun rosserial_python serial_node.py /dev/ttyUSB0",
            "softpneu3f": "roslaunch gripper_server machine_communication.launch yaml:=default_params_BSG.yaml port:=ttyUSB0",
            "brunelhand": "rosrun gripper_server brunelhand.py",
            "leaphand": "roslaunch leap_hand example.launch",
        }

        ExecutionFun = {
            "robotiq2f": self.robotiq2f_execution,
            "robotiq3f": self.robotiq3f_execution,
            "finray2f": self.finrays_execution,
            "finray4f": self.finrays_execution,
            "softpneu3f": self.softpneu3f_execution,
            "brunelhand": self.brunelhand_execution,
            "leaphand": self.leaphand_execution,
        }
        
        PublisherMsgName = {
            "robotiq2f":  "gripper_action_" + "ROBOTIQ_2F",
            "robotiq3f":  "gripper_action_" + "ROBOTIQ_3F",
            "finray2f": "easy_gripper_cmd",
            "finray4f": "easy_gripper_cmd",
            "softpneu3f": "gripper_action_" + "BSG",
            "brunelhand": "brunel_cmd",
            "leaphand": "/leaphand_node/cmd_allegro_timeit",
        }

        PublisherMsgType = {
            "robotiq2f": Float32MultiArray,
            "robotiq3f": Int16MultiArray,
            "finray2f": Vector3Stamped,
            "finray4f": Vector3Stamped,
            "softpneu3f": Float32MultiArray,
            "brunelhand": Float32MultiArray,
            "leaphand": JointState,
        }

        GripperTFName = {
            "robotiq2f": 'base_link',
            "robotiq3f": 'base',
            "finray2f": 'base_link',
            "finray4f": 'base_link',
            "softpneu3f": 'base_link',
            "brunelhand": 'base',
            "leaphand": 'base_link',
        }

        GripperTFName_vis = {
            "robotiq2f": 'robotiq2f1/base_link',
            "robotiq3f": 'robotiq3f1/base',
            "finray2f": 'finray2f1/base_link',
            "finray4f": 'finray4f1/base_link',
            "softpneu3f": 'softpneu3f1/base_link',
            "brunelhand": 'brunelhand1/base',
            "leaphand": 'leaphand1/base_link',
        }

        Armend2Gripper = {
            "robotiq2f": 0.00,
            "robotiq3f": 0.06,
            "finray2f": 0.12,
            "finray4f": 0.12,
            "softpneu3f": 0.08,
            "brunelhand": 0.02,
            "leaphand": 0.00,
        }

        Gripper2Grip = {
            "robotiq2f": 0.0, # because the optimal pose is the gripper pose already, not the object pose
            "robotiq3f": 0.0,
            "finray2f": 0.0,
            "finray4f": 0.0,
            "softpneu3f": 0.0,
            "brunelhand": 0.0,
            "leaphand": 0.0,
        }

        self.gripper_pulisher = rospy.Publisher(PublisherMsgName[gripper], PublisherMsgType[gripper], queue_size=1)
        
        self.gripper_tfname = GripperTFName[gripper]
        self.gripper_tfname_vis = GripperTFName_vis[gripper]
        self.armend2gripper = Armend2Gripper[gripper]
        self.gripper2grip = Gripper2Grip[gripper]
        
        self.grasp_execution = ExecutionFun[gripper]

        rospy.loginfo("[Gripper]: {} server: \n{}".format(gripper, ServerHint[gripper]))
    

    def robotiq2f_execution(self, action_name, joints):
        """
        opening_distance: the actual distance of opening [m] 
        palm_position: the position of angle or distance of some grippers
        """


    def robotiq3f_execution(self, action_name, joints):
        """
        opening_distance: the actual distance of opening [m] 
        palm_position: the position of angle or distance of some grippers
        rostopic pub /gripper_action_ROBOTIQ_3F  std_msgs/Int16MultiArray  '{data:[0, 0, 150, 255]}'  -1
        """


    def finrays_execution(self, action_name, joints):
        '''
        rosrun rosserial_python serial_node.py /dev/rfcomm0
        rostopic pub /easy_gripper_cmd geometry_msgs/Vector3Stamped   '{header: {frame_id:  STEP},  vector: {x: .0}}'  -1
        '''


    def softpneu3f_execution(self, action_name, joints):
        """
        opening_distance: the actual distance of opening [m]
        palm_position: the position of angle or distance of some grippers
        rostopic pub /gripper_action_BSG  std_msgs/Float32MultiArray  '{data:[0.0, 0.0, 0.0]}'  -1
        """


    def brunelhand_execution(self, action_name, joints):
        """
        opening_distance: the actual distance of opening [m] 
        palm_position: the position of angle or distance of some grippers
        rosrun gripper_server brunelhand.py
        rostopic pub /brunel_cmd  std_msgs/Float32MultiArray  '{data:[0.0, 0.0, 0.0, 0.0]}'  -1
        joint 1-4 -> [thumb, index, fore, rear-two]: 0.0: open, 1.0: close 
        """


    def leaphand_execution(self, action_name, joints):
        """
        self.leap_position = rospy.ServiceProxy('/leap_position', leap_position)
        #self.leap_velocity = rospy.ServiceProxy('/leap_velocity', leap_velocity)
        #self.leap_effort = rospy.ServiceProxy('/leap_effort', leap_effort)
        self.pub_hand = rospy.Publisher("/leaphand_node/cmd_ones", JointState, queue_size = 3) 
        """

