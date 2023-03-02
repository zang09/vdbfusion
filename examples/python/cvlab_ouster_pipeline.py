#!/usr/bin/env python3
# @file      kitti_pipeline.py
# @author    Ignacio Vizzo     [ivizzo@uni-bonn.de]
#
# Copyright (c) 2021 Ignacio Vizzo, all rights reserved
import argh

from datasets import CVLabOusterDataset as Dataset
from vdbfusion_ch_pipeline import VDBFusionPipeline as Pipeline


def main(
    data_name: str,
    data_path: str = "/Dataset/CVLAB/",
    config: str = "/vdbfusion/examples/python/config/cvlab_ouster.yaml",
    n_scans: int = -1,
    jump: int = 0,
    visualize: bool = True,
):
    """Help here!"""
    dataset = Dataset(data_path + data_name, config)

    # Mode: {map_only, make_scan, compare}
    pipeline = Pipeline(dataset, config, mode="compare", jump=jump, n_scans=n_scans, map_name=data_name)
    pipeline.run()
    pipeline.visualize() if visualize else None


if __name__ == "__main__":
    argh.dispatch_command(main)
