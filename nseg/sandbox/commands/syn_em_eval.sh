#!/bin/bash

# Script for running segmentation evaluation on synthetic EM cubes and tracking results under cfg.eval.result_zarr_root

set -euxo pipefail

## Cube paths

v2s1_zarr=/cajal/scratch/projects/misc/mdraw/data/synthtreeseg/p_em_and_seg_v2_seed1.zarr
v18s1_zarr=/cajal/scratch/projects/misc/mdraw/data/synthtreeseg/p_em_and_seg_v18_seed1.zarr
#v18s2_zarr=/cajal/scratch/projects/misc/mdraw/data/synthtreeseg/p_em_and_seg_v18_seed2.zarr # seed2 cube is automatically covered by seed1 inference


## Model paths

# AFF @ STS2
m_aff_sts2_q1=/cajal/scratch/projects/misc/mdraw/lsdex/v1/train_mtlsd/10-17_01-09_tense-farad_sts2_aff_q1/model_checkpoint_400000.pt
m_aff_sts2_q2=/cajal/scratch/projects/misc/mdraw/lsdex/v1/train_mtlsd/10-17_01-10_direct-plant_sts2_aff_q2/model_checkpoint_400000.pt
m_aff_sts2_q3=/cajal/scratch/projects/misc/mdraw/lsdex/v1/train_mtlsd/10-17_01-10_primordial-patch_sts2_aff_q3/model_checkpoint_400000.pt

# LSD @ STS2
m_lsd_sts2_q1=/cajal/scratch/projects/misc/mdraw/lsdex/v1/train_mtlsd/12-13_02-08_lead-deed_sts2_lsd_q1/model_checkpoint_400000.pt
m_lsd_sts2_q2=/cajal/scratch/projects/misc/mdraw/lsdex/v1/train_mtlsd/12-13_02-11_insulated-speaker_sts2_lsd_q2/model_checkpoint_400000.pt
m_lsd_sts2_q3=/cajal/scratch/projects/misc/mdraw/lsdex/v1/train_mtlsd/12-13_02-11_optical-circle_sts2_lsd_q3/model_checkpoint_400000.pt

# AFF @ STS18
m_aff_sts18_q1=/cajal/scratch/projects/misc/mdraw/lsdex/v1/train_mtlsd/12-13_02-18_moist-tenement_sts18_lsd_q1/model_checkpoint_400000.pt
m_aff_sts18_q2=/cajal/scratch/projects/misc/mdraw/lsdex/v1/train_mtlsd/12-13_02-33_terminal-scattering_sts18_lsd_q2/model_checkpoint_400000.pt
m_aff_sts18_q3=/cajal/scratch/projects/misc/mdraw/lsdex/v1/train_mtlsd/12-13_02-33_broad-lion_sts18_lsd_q3/model_checkpoint_400000.pt

# LSD @ STS18
m_lsd_sts18_q1=/cajal/scratch/projects/misc/mdraw/lsdex/v1/train_mtlsd/12-13_10-31_friendly-aperture_sts18_aff_q1/model_checkpoint_400000.pt
m_lsd_sts18_q2=/cajal/scratch/projects/misc/mdraw/lsdex/v1/train_mtlsd/12-13_10-32_silver-monitor_sts18_aff_q2/model_checkpoint_400000.pt
m_lsd_sts18_q3=/cajal/scratch/projects/misc/mdraw/lsdex/v1/train_mtlsd/12-13_10-32_silver-record_sts18_aff_q3/model_checkpoint_400000.pt


## Segmentation sweeps on seed1 (validation) cubes. If seed can be found, it is automatically run with the optimal validation threshold.

# AFF @ STS2
nseg-segment eval.cube_root=$v2s1_zarr eval.checkpoint_path=$m_aff_sts2_q1 eval.waterz_threshold_sweep_linspace='[0.01, 0.50, 50]'
nseg-segment eval.cube_root=$v2s1_zarr eval.checkpoint_path=$m_aff_sts2_q2 eval.waterz_threshold_sweep_linspace='[0.01, 0.50, 50]'
nseg-segment eval.cube_root=$v2s1_zarr eval.checkpoint_path=$m_aff_sts2_q3 eval.waterz_threshold_sweep_linspace='[0.01, 0.50, 50]'

# LSD @ STS2
nseg-segment eval.cube_root=$v2s1_zarr eval.checkpoint_path=$m_lsd_sts2_q1 eval.waterz_threshold_sweep_linspace='[0.01, 0.50, 50]'
nseg-segment eval.cube_root=$v2s1_zarr eval.checkpoint_path=$m_lsd_sts2_q2 eval.waterz_threshold_sweep_linspace='[0.01, 0.50, 50]'
nseg-segment eval.cube_root=$v2s1_zarr eval.checkpoint_path=$m_lsd_sts2_q3 eval.waterz_threshold_sweep_linspace='[0.01, 0.50, 50]'

# AFF @ STS18
nseg-segment eval.cube_root=v18s1_zarr eval.checkpoint_path=$m_aff_sts18_q1 eval.waterz_threshold_sweep_linspace='[0.01, 0.50, 50]'
nseg-segment eval.cube_root=v18s1_zarr eval.checkpoint_path=$m_aff_sts18_q2 eval.waterz_threshold_sweep_linspace='[0.01, 0.50, 50]'
nseg-segment eval.cube_root=v18s1_zarr eval.checkpoint_path=$m_aff_sts18_q3 eval.waterz_threshold_sweep_linspace='[0.01, 0.50, 50]'

# LSD @ STS18
nseg-segment eval.cube_root=v18s1_zarr eval.checkpoint_path=$m_lsd_sts18_q1 eval.waterz_threshold_sweep_linspace='[0.01, 0.50, 50]'
nseg-segment eval.cube_root=v18s1_zarr eval.checkpoint_path=$m_lsd_sts18_q2 eval.waterz_threshold_sweep_linspace='[0.01, 0.50, 50]'
nseg-segment eval.cube_root=v18s1_zarr eval.checkpoint_path=$m_lsd_sts18_q3 eval.waterz_threshold_sweep_linspace='[0.01, 0.50, 50]'
