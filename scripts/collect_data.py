from ast import arg
from sensor_msgs.msg import Image
import rospy
import ros_numpy
import numpy as np
import os
import argparse 
import cv2



def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--work-dir', help='data folder path')
    parser.add_argument('--name-index', help='the index of data')
    args = parser.parse_args()

    return args

class Collector():
    
    def __init__(self, path, index) -> None:
        self.path = path
        self.index = index
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.callback_depth)
        rospy.Subscriber("/camera/color/image_raw", Image, self.callback_rgb)
        os.chdir(path)

    def callback_rgb(self, data):
        data = ros_numpy.numpify(data)
        print(data.shape)
        print(data.dtype)
        with open('rgb_{0}.npy'.format(self.index), 'wb') as f:
            np.save(f, data)
            print("---collecting---")

    def callback_depth(self, data):
        data = ros_numpy.numpify(data)
        print(data.shape)
        print(data.dtype)
        with open('depth_{0}.npy'.format(self.index), 'wb') as f:
            np.save(f, data)
            print("---collecting---")

if __name__ == "__main__":
    rospy.init_node("data_collector", anonymous=True)
    args = parse_args()
    if args.work_dir is None:
        args.work_dir = "/home/xihelm/catkin_ws/src/tomato_grasp/data"
    pipeline = Collector(args.work_dir, args.name_index)

    rospy.spin()