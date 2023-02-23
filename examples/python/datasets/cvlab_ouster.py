import glob
import fnmatch
import sys
import os

import numpy as np
import open3d as o3d
from typing import Dict
import json

sys.path.append("..")

from utils.cache import get_cache, memoize
from utils.config import load_config
from utils.ouster_lidar import OusterData


class CVLabOusterDataset:
    def __init__(self, data_source, config_file: str):
        self.config = load_config(config_file)
        self.data_source = os.path.join(data_source, "")
        self.scan_folder = os.path.join(self.data_source, "range/")
        self.pose_file = os.path.join(self.data_source, "pose_graph.json")

        # Load scan files and poses
        self.scan_files = self.get_csv_filenames(self.scan_folder)
        self.poses = self.load_poses(self.pose_file)

        # Cache
        self.use_cache = True
        self.cache = get_cache(directory="cache/cvlab_ouster")

    def __len__(self):
        if len(self.scan_files) != len(self.poses):
            print("LiDAR scan:", len(self.scan_files))
            print("Pose given:", len(self.poses))
            exit(0)

        return len(self.scan_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.scan_folder, self.scan_files[idx])
        return self.getitem(file_path, idx, self.config)

    @memoize()
    def getitem(self, file_path: str, idx: int, config: Dict):
        pose = self.poses[idx]
        scan = self.csv_to_o3d_pc(file_path)

        points = np.asarray(scan.points)
        points = points[np.linalg.norm(points, axis=1) <= config.max_range]
        points = points[np.linalg.norm(points, axis=1) >= config.min_range]

        scan = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        scan.transform(pose) if config.apply_pose else None
        return np.asarray(scan.points), pose

    def load_poses(self, poses_file):
        poses = []

        with open(poses_file, 'r') as file:
            data = file.read()
            dict = json.loads(data)
            
            for dict2 in dict['nodes']:
                pose = np.array(dict2['pose'])
                pose = pose.reshape(4, 4)

                poses.append(pose.T)
                
        return poses

    def get_csv_filenames(self, scans_dir):
        # 000000.csv
        def get_cloud_number(csv_filename):
            number = csv_filename[:-4]
            return int(number)

        return sorted(fnmatch.filter(os.listdir(scans_dir), '*.csv'), key=get_cloud_number)

    def csv_to_o3d_pc(self, range_path):
        data = OusterData(range_path)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data.xyz)
        
        return pcd

