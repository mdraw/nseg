# **nseg** - Improving Neuron Instance Segmentation

## Installation

- From project root:

    conda env create -f environment.yml
    conda activate nseg
    pip install .

## Data preparation

Run in this order:

    python -m nseg.scripts.rename_gt
    python -m nseg.scripts.zarrify_j0126_gt
    python -m nseg.scripts.create_data_split

## Training

    nseg-train

## Small-scale inference and evaluation (small cubes)

    nseg-segment

## Large-scale inference and evaluation (whole dataset or ROI)

    python -m nseg.inference.full_eval
