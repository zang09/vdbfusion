#!/usr/bin/env python3
# @file      scannet_pipeline.py
# @author    Ignacio Vizzo     [ivizzo@uni-bonn.de]
#
# Copyright (c) 2021 Ignacio Vizzo, all rights reserved
import argh

from datasets import FlatDataset as Dataset
from vdbfusion_ch_pipeline import VDBFusionPipeline as Pipeline


def main(
    map_name: str,
    data_source: str = "/Users/zang09/Haebeom/Dataset/FLAT/",
    config: str = "/Users/zang09/Haebeom/VScode_ws/vdbfusion/examples/python/config/flat.yaml",
    n_scans: int = -1,
    jump: int = 0,
    visualize: bool = True,
):
    """Help here!"""
    dataset = Dataset(data_source + map_name)
    
    # Mode: {map_only, make_scan, compare}
    data_name = data_source.split('/')[-2]
    pipeline = Pipeline(dataset, config, mode="make_scan", jump=jump, n_scans=n_scans, map_name=data_name + '_' + map_name)
    pipeline.run()
    pipeline.visualize() if visualize else None


if __name__ == "__main__":
    argh.dispatch_command(main)
