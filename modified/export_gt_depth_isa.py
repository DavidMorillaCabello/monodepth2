from __future__ import absolute_import, division, print_function

import os

import argparse
import numpy as np
import PIL.Image as pil

from utils import readlines
from kitti_utils import generate_depth_map #our data is formated as KITTI

def export_gt_depths_isa():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the ISA data',
                        required=True)

    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from', #TODO: Create split.
                        required=True)
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []
    for line in lines:

        folder,frame_id,_ = line.split()
        frame_id = int(frame_id)

        calib_dir = os.path.join(opt.data_path, folder.split("/")[0]) #The calibration was performed per day.
        velo_filename = os.path.join(opt.data_path, folder, "velodyne_points/data", "{:010d}.png".format(frame_id))

        gt_depth = generate_depth_map(calib_dir, velo_filename, 0, True) #0 is rgb camera and 1 thermal camera.

        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))

if __name__ == "__main__":
    
    export_gt_depths_isa()
                        
