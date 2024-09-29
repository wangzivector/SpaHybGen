import numpy as np
import torch
import time
import rospy
from pathlib import Path
import spahybgen.inference as Inference
from spahybgen.networks import load_network
from std_msgs.msg import Float32MultiArray
import spahybgen.visualization as Visualization


class InferenceServer:
    def __init__(self, grid_topic, infer_topic, model_path, grid_length, voxel_disc, ori_type, 
                 visual_inference) -> None:
        self.grid_length = grid_length
        self.visual_inference = visual_inference
        self.voxel_disc = voxel_disc

        ## Inintialize Inference Network
        ntargs = {"voxel_discreteness": voxel_disc, "orientation": ori_type, "augment": False}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(Path(model_path), self.device, ntargs)
        rospy.Subscriber(grid_topic, Float32MultiArray, self.inference_cb)
        self.infer_pub = rospy.Publisher(infer_topic, Float32MultiArray, queue_size=1, latch=True)


    def inference(self, gird_vol):
        if(len(gird_vol.shape) == 3): gird_vol = np.expand_dims(gird_vol, axis=0)

        ## grasp generation: inference
        tic = time.time()
        rospy.logwarn("gird_vol:{},{},{}".format(gird_vol.dtype, gird_vol.mean(), gird_vol.shape))
        qual_vol, rot_vol, wren_vol = Inference.predict(gird_vol, self.net, self.device)

        rospy.loginfo("[Inference]: Forward pass:{:.03f}s".format(time.time() - tic))

        tic = time.time()
        qual_vol_pro, rot_vol_pro, wren_vol_pro = Inference.process(qual_vol, rot_vol, wren_vol, gaussian_filter_sigma=1) # originally 3
        prediction = np.vstack([gird_vol, np.expand_dims(qual_vol_pro, axis=0), rot_vol_pro, 
                                np.expand_dims(wren_vol_pro, axis=0)])
        rospy.loginfo("[Inference]: Filter: {:.03f}s".format(time.time() - tic))
        return prediction
    
    
    def inference_cb(self, msg):
        grid = np.array(msg.data).astype(np.float32).reshape(self.voxel_disc, self.voxel_disc, self.voxel_disc)
        # np.save("data/observe/inspect_grid", grid) # save inspection grid to local
        rospy.loginfo("[Generation]: Received grid msg: {}".format(grid.shape))
        prediction = self.inference(grid)
        msg_prediction = Float32MultiArray(data=prediction.astype(np.float32).reshape(-1))
        # np.save("data/observe/prediction", prediction) # save prediction to local
        self.infer_pub.publish(msg_prediction)

        if self.visual_inference:
            visual_inference_threshold = 0.6
            Visualization.visualize_inference(prediction, self.grid_length/self.voxel_disc, visual_inference_threshold)
            

###
### Inference Server Instance
###
if __name__ == '__main__':
    rospy.init_node("inference_server")
    rospy.loginfo("[Inference]: Started sensor_server node.")

    from spahybgen.pipeline.param_server import GraspParameter
    GP = GraspParameter("./config/grasp_generation.yaml")

    azure_node = InferenceServer(
        GP.sensor.grid_topic, 
        GP.inference.infer_topic, 
        GP.inference.model_path, 
        GP.sensor.grid_length, 
        GP.sensor.voxel_disc, 
        GP.inference.ori_type, 
        GP.inference.visual_inference
    )
    
    rospy.loginfo("[Inference]: InferenceNode Created and Spinning.")
    rospy.spin()
    rospy.loginfo("[Inference]: InferenceNode Finished.")