#!/usr/bin/env python3
# @file      scannet_pipeline.py
# @author    Ignacio Vizzo     [ivizzo@uni-bonn.de]
#
# Copyright (c) 2021 Ignacio Vizzo, all rights reserved
import argh

from datasets import RIODataset as Dataset
from vdbfusion_ch_pipeline import VDBFusionPipeline as Pipeline


def main(
    map_name: str,
    compare_name: str = "",
    data_source: str = "/Dataset/RIO/",
    config: str = "/vdbfusion/examples/python/config/rio.yaml",
    mode: str = "make_scan",
    n_scans: int = -1,
    jump: int = 0,
    visualize: bool = False,
):
    """Help here!"""
    dataset = Dataset(data_source + map_name)
    
    # Mode: {map_only, make_scan, compare}
    data_name = data_source.split('/')[-2]
    pipeline = Pipeline(dataset, config, mode=mode, jump=jump, n_scans=n_scans, \
                        map_name=data_name + '_' + map_name, compare_name=data_name + '_' + compare_name)
    pipeline.run()
    pipeline.visualize() if visualize else None


if __name__ == "__main__":
    argh.dispatch_command(main)
