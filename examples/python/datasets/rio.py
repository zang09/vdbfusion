#!/usr/bin/env python3
# MIT License
#
# # Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Original implementation by Federico Magistri
import os
import fnmatch

import cv2 as cv
import numpy as np
import open3d as o3d

class RIODataset:
    def __init__(self, data_source, get_color=False, split=None):

        self.name = data_source.split("/")[-1]

        self.get_color = get_color
        self.data_source = data_source + '/sequence'

        self.rgb_list = os.listdir(os.path.join(self.data_source, "color"))
        self.rgb_list.sort()
        self.rgb_list = fnmatch.filter(self.rgb_list, '*.jpg')

        self.depth_list = os.listdir(os.path.join(self.data_source, "depth"))
        self.depth_list.sort()
        self.depth_list = fnmatch.filter(self.depth_list, '*.pgm')

        self.pose_list = os.listdir(os.path.join(self.data_source, "pose"))
        self.pose_list.sort()
        self.pose_list = fnmatch.filter(self.pose_list, '*.txt')

    def __len__(self):

        return len(self.pose_list)

    def __getitem__(self, idx):
        # game_room2 is reference map
        if self.name == 'game_room1':
            t_list = [0.24480035901069641,
                    -0.96956980228424072,
                    -0.0026853315066546202,
                    0,
                    0.96956843137741089,
                    0.24480713903903961,
                    -0.0025742929428815842,
                    0,
                    0.0031533448491245508,
                    -0.0019734248053282499,
                    0.99999308586120605,
                    0,
                    0.38689333200454712,
                    -0.96944725513458252,
                    -0.36265155673027039,
                    1]

        # living_room1 is reference map
        elif self.name == 'living_room2':
            t_list = [0.31472310423851013,
                    0.9488682746887207,
                    -0.024466356262564659,
                    0,
                    -0.94895601272583008,
                    0.31510752439498901,
                    0.013779248110949993,
                    0,
                    0.020784223452210426,
                    0.018880847841501236,
                    0.99960571527481079,
                    0,
                    -1.3112174272537231,
                    -3.9697799682617188,
                    0.22058026492595673,
                    1]
        elif self.name == 'living_room3':
            t_list = [0.77222687005996704,
                    0.63531720638275146,
                    0.0061436998657882214,
                    0,
                    -0.63533353805541992,
                    0.77223718166351318,
                    0.00099064130336046219,
                    0,
                    -0.004115021787583828,
                    -0.0046682986430823803,
                    0.999980628490448,
                    0,
                    -0.35366585850715637,
                    1.0989818572998047,
                    -0.047962937504053116,
                    1]
        elif self.name == 'living_room4':
            t_list = [0.67985737323760986,
                    -0.73331701755523682,
                    0.0063448813743889332,
                    0,
                    0.73334181308746338,
                    0.67985272407531738,
                    -0.0031882103066891432,
                    0,
                    -0.0019756162073463202,
                    0.0068204938434064388,
                    0.99997484683990479,
                    0,
                    -2.1962716579437256,
                    0.033798474818468094,
                    0.050790634006261826,
                    1]
                    
        else:
            t_list = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

        transform = np.array(t_list)
        transform = transform.reshape(4, 4).T

        pose = np.loadtxt(os.path.join(self.data_source, "pose", self.pose_list[idx]))
        pose = np.matmul(transform, pose)
        pose_inv = np.linalg.inv(pose)
        
        # Depth
        depth = cv.imread(os.path.join(self.data_source, "depth", self.depth_list[idx]), -1)
        depth = np.array(depth, dtype=np.float32)

        depth_intrinsic = np.loadtxt(os.path.join(self.data_source, "_depth_intrinsics.txt"))
        depth_cam = o3d.camera.PinholeCameraIntrinsic(depth.shape[1], depth.shape[0], depth_intrinsic)

        cols, rows = np.meshgrid(
                np.linspace(0,
                            depth_cam.width - 1,
                            num=depth_cam.width),
                np.linspace(0,
                            depth_cam.height - 1,
                            num=depth_cam.height))
        
        im_x = (cols - depth_cam.intrinsic_matrix[0][2]) / depth_cam.intrinsic_matrix[0][0]
        im_y = (rows - depth_cam.intrinsic_matrix[1][2]) / depth_cam.intrinsic_matrix[1][1]
    
        # Color
        bgr = cv.imread(os.path.join(self.data_source, "color", self.rgb_list[idx]))
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

        color_intrinsic = np.loadtxt(os.path.join(self.data_source, "_color_intrinsics.txt"))
        color_cam = o3d.camera.PinholeCameraIntrinsic(rgb.shape[1], rgb.shape[0], color_intrinsic)

        blank_img = np.zeros((172, 224, 3), dtype=np.uint8)

        im_u = im_x * color_cam.intrinsic_matrix[0][0] + color_cam.intrinsic_matrix[0][2]
        im_v = im_y * color_cam.intrinsic_matrix[1][1] + color_cam.intrinsic_matrix[1][2]
        im_u = np.array(np.clip(np.round(im_u), 0,
                                color_cam.width - 1),
                                dtype=int)
        im_v = np.array(np.clip(np.round(im_v), 0,
                                color_cam.height - 1),
                                dtype=int)
        
        for u in range(depth_cam.width):
            for v in range(depth_cam.height):
                blank_img[v, u, :] = rgb[im_v[v, u], im_u[v, u], :]
        rgb = blank_img

        # cv.imshow('depth', depth)
        # cv.imshow('color', rgb)
        # cv.waitKey(3)

        rgb = o3d.geometry.Image(rgb)
        depth = o3d.geometry.Image(depth)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth, depth_scale=1000
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, depth_cam, pose_inv, project_valid_depth_only=True
        )

        # o3d.visualization.draw_geometries([pcd])

        xyz = np.array(pcd.points)
        colors = np.array(pcd.colors)

        xyz = np.array(xyz)
        colors = np.array(colors)

        if self.get_color:
            return xyz, colors, np.array(pose)

        return xyz, np.array(pose)
