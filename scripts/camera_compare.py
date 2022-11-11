from ast import arg
from sensor_msgs.msg import Image
import rospy
import ros_numpy
import numpy as np
import os
import argparse 
import cv2

import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--work-dir', help='data folder path')
    parser.add_argument('--name-index', help='the index of data')
    args = parser.parse_args()

    return args

class Collector():
    
    def __init__(self, path, index, averageCounter, averagedData) -> None:
        self.path = path
        self.index = index
        #rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.callback_depth)
        rospy.Subscriber("/zed2i/zed_node/depth/depth_registered", Image, self.callback_depth)
        os.chdir(path)

        self.averageCounter = averageCounter
        self.averagedData = averagedData

    def callback_depth(self, data):
        if self.averageCounter < 10:
            data = ros_numpy.numpify(data)
            print(type(data))
            print(data)
            self.averagedData = np.add(self.averagedData, data)
            self.averageCounter += 1
            print(pipeline.averagedData)
            print(" ''' Collected {0} data ''' ".format(self.averageCounter))
        else:
            print(pipeline.averagedData / 10)
            with open('zed2i_depth_trusses.npy', 'wb') as f:
                np.save(f, pipeline.averagedData / 10)
                print("--- Saved ---")        

        # with open('depth_{0}.npy'.format(self.index), 'wb') as f:


if __name__ == "__main__":
    rospy.init_node("camera_comparator", anonymous=True)
    args = parse_args()
    if args.work_dir is None:
        args.work_dir = "/home/xihelm/catkin_ws/src/tomato_grasp/compare_data"
    pipeline = Collector(path=args.work_dir, index=args.name_index, averageCounter=0, averagedData=np.zeros((621, 1104)))

    if pipeline.averageCounter == 9:
        print("after 10")
        print(pipeline.averagedData / 10)
        plt.hist(pipeline.averagedata / 10, ec="purple", fc="green", alpha=0.5)
        plt.ylabel('Depth (mm)')
        plt.xlabel('Frequency')
        plt.show()

    rospy.spin()