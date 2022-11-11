from ast import arg
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib
import cv2
from mmdet.apis import init_detector, inference_detector
from typing import List
from dataclasses import dataclass
from PIL import Image
import math
from cameraModel import PinholeCameraModel
import math
import copy
import numpy
import yaml

def mkmat(rows, cols, L):
    mat = numpy.matrix(L, dtype='float64')
    mat.resize((rows,cols))
    return mat

@dataclass
class Tomato:
    xy_centre: List[np.float32]
    bbox: List[int]
    mask: np.ndarray

config_file = '/home/xihelm/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/home/xihelm/mmdetection/laboro_tomato_little_48ep.pth'
path =  "/home/xihelm/catkin_ws/src/tomato_grasp/data/"
model = init_detector(config_file, checkpoint_file, device='cuda')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg-save', help='whether to save rendered images')
    args = parser.parse_args()

    return args

def segment_multiple_imgs_and_save(out_path = "/home/xihelm/catkin_ws/src/tomato_grasp/data_rendered/", number = 15):
    os.chdir(out_path)
    for i in range (1, number):
        img_bgr = np.load(path + "rgb_{0}.npy".format(i))
        img_rgb = img_bgr[...,::-1].copy()
        result = inference_detector(model, img_rgb)
        model.show_result(img_rgb, result, out_file='rgb_segmented{0}.jpg'.format(i))
        print("----- " + str(round(100 * i / 15, 1)) + "% completed -----" )

args = parse_args()
if args.seg_save is not None:
    segment_multiple_imgs_and_save()


def process_tomato_data(bboxes, masks):

    tomato_data = []
    for bbox, mask in zip(bboxes, masks):

        d_x = bbox[2] - bbox[0]
        d_y = bbox[3] - bbox[1]
        xy_centre = [bbox[0] + d_x * 0.5, bbox[1] + d_y * 0.5]
        tomato_data.append(Tomato(xy_centre, bbox, mask))

    return tomato_data

def plot_masks(masks, name):

    plt.figure()
    mask_total = None
    for mask in masks:
        if mask_total is not None:
            mask_total = np.logical_or(mask_total, mask)
        else:
            mask_total = mask

    plt.imshow(255 * mask_total)
    plt.savefig(name)
    plt.close("all")

def segment_stem_threshold(
        hsvimg,         
        min_hsv_h_peduncle=0.07,
        max_hsv_h_peduncle=0.50):

    """Extract the peduncle based on colour
    This extracts all the pixels from an image in the HSV space which are in the
    green area of the spectrum
    Args:
        hsvimg (ndarray): image of segmented truss with HSV channels
    Returns:
        keep_array (ndarray): boolean array of what pixels are in the range deemed to be green
    """
    
    featimg = np.stack(
        (hsvimg[:, :, 0].flatten(), hsvimg[:, :, 1].flatten(), hsvimg[:, :, 2].flatten()),
        axis=-1,
    )

    keep_array = np.logical_and(
        featimg[:, 0] > min_hsv_h_peduncle,
        featimg[:, 0] < max_hsv_h_peduncle,
    )

    return keep_array

camera_model = PinholeCameraModel()
with open('/home/xihelm/catkin_ws/src/tomato_grasp/data/camera_info.yaml') as f:
    cam_info = yaml.safe_load(f)

print(cam_info)
camera_model.fromCameraInfoDict(cam_info)


bboxes = []
masks = []

class DrawLineWidget(object):
    def __init__(self, image, depth_image):
        self.original_image = image
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = []
        self.grasp_xy = []
        self.line = []
        self.xyz = []

    @staticmethod
    def uvd2xyz(u, v, d):
        ray = camera_model.projectPixelTo3dRay((u, v))
        # normalize the ray so its Z-component equals 1.0
        ray_normalised = [i / ray[2] for i in ray]
        # multiply the ray by the depth; its Z-component
        # should now equal the depth value
        xyz= [i * d for i in ray_normalised]
        return xyz

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            if self.image_coordinates[0] == self.image_coordinates[1]:
                print('Grasp point: {}'.format(self.image_coordinates[0]))
                self.grasp_xy = [self.image_coordinates[0]]
                v = self.image_coordinates[0][0]
                u = self.image_coordinates[0][1]
                print("Depth: ", depth[u][v])
                self.xyz = self.uvd2xyz(u, v, depth[u][v])
                print("xyz: {}".format(self.xyz))
            else:
                print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
                self.line = [self.image_coordinates[0], self.image_coordinates[1]]
                # reference to first point 
                dy = self.image_coordinates[1][1] - self.image_coordinates[0][1]
                dx = self.image_coordinates[1][0] - self.image_coordinates[0][0]

                print("Rotation: {}".format(np.arctan2(dy, dx)))

            # Draw line
            cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (255,0,0), 2)
            cv2.imshow("image", self.clone) 

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

for i in range (16, 17): # 31
    img_index = 2
    img_index = i
    bboxes = []
    masks = []

    #img_bgr = np.load(path + "rgb_{0}.npy".format(img_index))
    img_bgr = cv2.imread("/home/xihelm/Pictures/m0.jpg")
    print(type(img_bgr))
    print(img_bgr.shape)
    #img_rgb = img_bgr[...,::-1].copy()
    img_rgb = img_bgr.copy()
    result = inference_detector(model, img_rgb)
    model.show_result(img_rgb, result, out_file='rgb_segmented.jpg')
    for bbox_list, mask_list in zip(result[0], result[1]):
        for bbox, mask in zip(bbox_list, mask_list):
            bboxes.append(bbox)
            masks.append(mask)

    tomato_data = process_tomato_data(bboxes, masks)

    # plot_masks([tom.mask for tom in tomato_data], "segmentation_tomatoes_laboro.png")

    # # Combine all the individual tomato masks into a single mask
    single_tomatoes_all = np.zeros(img_rgb.shape[:2])
    for tomato in tomato_data:
        single_tomatoes_all = np.logical_or(single_tomatoes_all, tomato.mask)
    tomato_data = []

    im = Image.fromarray(single_tomatoes_all)
    print(i)
    im.save("no_tomato_{0}.png".format(i))

    #    np.zeros_like(img_rgb),
    #)
    #ped_img = np.reshape(ped_img, img_rgb.shape)
    no_tomatoes = np.bitwise_not(single_tomatoes_all)
    gray = np.mean(img_rgb, axis=2)
    no_tomatoes = np.logical_and(gray, no_tomatoes)
    res = img_bgr * no_tomatoes[..., None]
    #print(no_tomatoes)
    plt.figure()
    plt.imshow(single_tomatoes_all) 

    plt.savefig("peduncles")
    # im = Image.fromarray(res)
    # im.save("no_tomato.png")

    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    # Threshold of blue in HSV space    bboxes = []
    masks = []
    lower_blue = np.array([60, 35, 140])
    upper_blue = np.array([100, 255, 255])

    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    result = cv2.bitwise_and(res,res, mask = mask)
    result = np.mean(result, axis=2)
    print(result.shape)
    #print(result.type)
    #result = cv2.HoughLinesP(result,1,np.pi/180,40,minLineLength=30,maxLineGap=30)
    # cv2.imwrite('DEBUG-plainMask.png', result)
    #hsvimg = matplotlib.colors.rgb_to_hsv(img_rgb / 255)
    #keep_array = segment_stem_threshold(hsvimg)
    #ped_img = np.where(
    #    np.repeat(keep_array[:, np.newaxis], 3, axis=1).reshape(img_rgb.shape[0], img_rgb.shape[1], 3),
    #    img_rgb,
    #    np.zeros_like(img_rgb)
    #)
    #ped_img = np.reshape(ped_img, img_rgb.shape)
    no_tomatoes = np.bitwise_not(single_tomatoes_all)
    gray = np.mean(img_rgb, axis=2)
    no_tomatoes = np.logical_and(gray, no_tomatoes)
    res = img_bgr * no_tomatoes[..., None]
    #print(no_tomatoes)
    # plt.figure()
    # plt.imshow(res)
    # plt.savefig("peduncles")
    im = Image.fromarray(res)
    # im.save("no_tomato.png")

    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    # Threshold of blue in HSV space
    lower_blue = np.array([60, 35, 140])
    upper_blue = np.array([100, 255, 255])

    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    result = cv2.bitwise_and(res,res, mask = mask)
    result = np.mean(result, axis=2)
    cv2.imwrite('DEBUG-plainMask_color_{0}.png'.format(i), result)
    
    edges = cv2.Canny(res,100,200)
    #print(result.type)
    #result = cv2.HoughLinesP(result,1,np.pi/180,40,minLineLength=30,maxLineGap=30)
    cv2.imwrite('DEBUG-plainMask_{0}.png'.format(i), edges)

    # depth = cv2.convertScaleAbs(depth)
    # edges_depth = cv2.Canny(depth, 100, 200)

    thinned = cv2.ximgproc.thinning(edges)
    cv2.imwrite('DEBUG-plainMask_{0}.png'.format(i), thinned)
    
    depth = np.load(path + "depth_{0}.npy".format(i))
    depth = np.divide(depth, 1000)
    print(depth)
    cv2.imwrite('DEBUG-plainMask_depth_{0}.png'.format(i), depth)
    # draw_line_widget = DrawLineWidget(image=thinned, depth_image=depth)
    # while True:
    #     cv2.imshow('image', draw_line_widget.show_image())
    #     key = cv2.waitKey(1)

    #     # Close program with keyboard 'q'
    #     if key == ord('q'):
    #         cv2.destroyAllWindows()
    #         exit(1)
