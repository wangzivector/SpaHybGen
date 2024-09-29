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
        data_to_send = Float32MultiArray()
        maximal_open = 0.085
        position = maximal_open - (joints[0] + joints[1])

        if action_name == "GRIPPER_CLOSE": position = 0.001
        if action_name == "GRIPPER_OPEN": position = maximal_open

        if position > maximal_open: position = maximal_open
        force, speed = 20.0, 0.02
        data_to_send.data = [position, speed, force]
        self.gripper_pulisher.publish(data_to_send)
        rospy.logwarn("Robotiq 2f openwidth: {} m".format(position))


    def robotiq3f_execution(self, action_name, joints):
        """
        opening_distance: the actual distance of opening [m] 
        palm_position: the position of angle or distance of some grippers
        rostopic pub /gripper_action_ROBOTIQ_3F  std_msgs/Int16MultiArray  '{data:[0, 0, 150, 255]}'  -1
        """
        palm_limit1, palm_limit2, palm_range = -0.1784, 0.1920, 0.1784 + 0.1920
        opening_init, opening_close, opening_range, opening_bias = 0, 1.2218, 1.2218, 0.2
        opening_position, palm_position = (joints[1] + joints[5] + joints[8])/3 - opening_bias, (joints[0] + joints[4])/2
        force, speed = 120, 255
        data_to_send = Int16MultiArray()

        if action_name == "GRIPPER_CLOSE": opening_position = opening_close
        if action_name == "GRIPPER_OPEN": opening_position, palm_position = opening_init, (palm_limit1)/2

        data_action = [int((opening_position-opening_init)/opening_range*255), 255-(int((palm_position-palm_limit1)/palm_range*200)+40), speed, force]

        for ind in range(len(data_action)):
            data_action[ind] = int(data_action[ind])
            if data_action[ind] > 255: data_action[ind] = 255
            if data_action[ind] < 0: data_action[ind] = 0

        data_to_send.data = data_action
        self.gripper_pulisher.publish(data_to_send)


    def finrays_execution(self, action_name, joints):
        '''
        rosrun rosserial_python serial_node.py /dev/rfcomm0
        rostopic pub /easy_gripper_cmd geometry_msgs/Vector3Stamped   '{header: {frame_id:  STEP},  vector: {x: .0}}'  -1
        '''
        maximal_open = 0.785 # radius
        open_bias = 0.01
        position = (maximal_open - joints.mean()) + open_bias

        if action_name == "GRIPPER_CLOSE": position = 0.001 # position / 2
        if action_name == "GRIPPER_OPEN": position = maximal_open
        if position > maximal_open: position = maximal_open
        
        data_to_send = Vector3Stamped()
        data_to_send.header.frame_id = "STEP"
        data_to_send.vector.x = position / maximal_open
        self.gripper_pulisher.publish(data_to_send)


    def softpneu3f_execution(self, action_name, joints):
        """
        opening_distance: the actual distance of opening [m]
        palm_position: the position of angle or distance of some grippers
        rostopic pub /gripper_action_BSG  std_msgs/Float32MultiArray  '{data:[0.0, 0.0, 0.0]}'  -1
        """
        maximal_open = math.pi/4
        max_palm_position = math.pi/2
        init_opening, init_palm = 0.0, math.pi/6
        palm_position, opening_position = (joints[0] + joints[2])/2, (joints[1] + joints[3] + joints[4])/3
        vacumming_act = 1.0

        if action_name == "GRIPPER_CLOSE": opening_position = maximal_open
        if action_name == "GRIPPER_OPEN": opening_position, palm_position = init_opening, init_palm
        
        opening_position, palm_position = opening_position/maximal_open, palm_position/max_palm_position
        data_action = [opening_position, palm_position, vacumming_act]
        for ind in range(len(data_action)):
            if data_action[ind] > 1: data_action[ind] = 1.0
            if data_action[ind] < 0: data_action[ind] = 0.0

        data_to_send = Float32MultiArray()
        # data_to_send.data = [0.04, 0.01, 20.0] # position, palm, vacumming [0.0 1.0]
        data_to_send.data = data_action
        self.gripper_pulisher.publish(data_to_send)


    def brunelhand_execution(self, action_name, joints):
        """
        opening_distance: the actual distance of opening [m] 
        palm_position: the position of angle or distance of some grippers
        rosrun gripper_server brunelhand.py
        rostopic pub /brunel_cmd  std_msgs/Float32MultiArray  '{data:[0.0, 0.0, 0.0, 0.0]}'  -1
        joint 1-4 -> [thumb, index, fore, rear-two]: 0.0: open, 1.0: close 
        """
        minimal_open = 0.0 # radius
        maximal_open = 1.57 # radius
        open_bias = 0.2
        close_position = maximal_open*0.8
        m0 = joints[0] + 3 * open_bias
        m1 = (joints[1] + joints[2])/2 - open_bias
        m2 = (joints[3] + joints[4])/2 - open_bias
        m3 = (joints[5] + joints[6] + joints[7] + joints[8])/4 - open_bias

        if action_name == "GRIPPER_CLOSE": m1, m2, m3 = close_position, close_position, close_position
        if action_name == "GRIPPER_OPEN": m0, m1, m2, m3 = minimal_open, minimal_open, minimal_open, minimal_open

        data_action = [m0/maximal_open, m1/maximal_open, m2/maximal_open, m3/maximal_open]

        for ind in range(len(data_action)):
            if data_action[ind] > 1: data_action[ind] = 1.0
            if data_action[ind] < 0: data_action[ind] = 0.0

        data_to_send = Float32MultiArray()
        data_to_send.data = data_action
        self.gripper_pulisher.publish(data_to_send)
        rospy.logwarn("Motor Values:{}, {}, {}, {}".format(m0, m1, m2, m3))
        rospy.logwarn("Data_action:{}".format(data_action))


    def leaphand_execution(self, action_name, joints):
        """
        self.leap_position = rospy.ServiceProxy('/leap_position', leap_position)
        #self.leap_velocity = rospy.ServiceProxy('/leap_velocity', leap_velocity)
        #self.leap_effort = rospy.ServiceProxy('/leap_effort', leap_effort)
        self.pub_hand = rospy.Publisher("/leaphand_node/cmd_ones", JointState, queue_size = 3) 
        std_msgs/Header header
        string[] name
        float64[] position
        float64[] velocity
        float64[] effort
        # open_bias = 0.2   
        Processing joint #0: 1
        Processing joint #1: 0
        Processing joint #2: 2
        Processing joint #3: 3
        Processing joint #4: 5
        Processing joint #5: 4
        Processing joint #6: 6
        Processing joint #7: 7
        Processing joint #8: 9
        Processing joint #9: 8
        Processing joint #10: 10
        Processing joint #11: 11
        Processing joint #12: 12
        Processing joint #13: 13
        Processing joint #14: 14
        Processing joint #15: 15
rostopic pub -1 /leaphand_node/cmd_allegro_timeit sensor_msgs/JointState -- '{stamp: now, frame_id: open}' '[]' '[0.0, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5,     0.5, 1.5, 0, 0, 0.5]' '[1]' '[]'
rostopic pub -1 /leaphand_node/cmd_allegro_timeit sensor_msgs/JointState -- '{stamp: now, frame_id: close}' '[]' '[-0, 1.57, 0.5, 0.5, 0, 1.57, 0.5, 0.5, 0, 1.5, 0.5, 0.5, 1.6, 0, 0, 1.0]' '[1]' '[]' 
        """
        ## remap joint index 
        joints = joints[[1,0,2,3,5,4,6,7,9,8,10,11,12,13,14,15]]
        if action_name == "GRIPPER_OPEN": joints = [0.0, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 1.5, 0, 0, 0.5]
        if action_name == "GRIPPER_CLOSE": joints = [-0, 1.57, 0.5, 0.5, 0, 1.57, 0.5, 0.5, 0, 1.5, 0.5, 0.5, 1.6, 0, 0, 1.0]

        stater = JointState()
        stater.velocity = [1] # time: seconds
        stater.position = joints
        self.gripper_pulisher.publish(stater)
        rospy.logwarn("Motor Values:{}".format(joints))
