from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class ISADataset(MonoDataset):
    """Superclass for different types of ISA dataset loaders
    """

    def __init__(self, *args, **kwargs):
        super(ISADataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.6951924739,  0,              0.365566426,    0],
                           [0,             0.7624195951,   0.3053229782,   0],
                           [0,             0,              1,              0],
                           [0,             0,              0,              1]], dtype=np.float32)  # Intrinsics

        # Shape of the full resolution image
        self.full_res_shape = (704, 576)

        # CAMERA SELECTION. Will get the image number to generate the depth map???
        self.side_map = {"rgb": 0, "the": 1,
                         "image_00": 0, "image_01": 1, "0": 0, "1": 1}

    def check_depth(self):
        '''Returns if the filename for depth exists in the path of the scene.
        '''
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index))
        )

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        '''Do a color transpose if do_flip is activated (Look into PIL loader)
        '''
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)


class ISARawDataset(ISADataset):
    '''ISA dataset which loads the original velodyne depth maps for ground truth
    '''

    def __init__(self, *args, **kwargs):
        super(ISARawDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)

        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(
            calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class ISADepthDataset(ISADataset):
    """ISA dataset which uses the updated ground truth depth maps
    """

    def __init__(self, *args, **kwargs):
        super(ISADepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
