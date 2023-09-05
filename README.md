# **nseg** - Improving Neuron Instance Segmentation


## Installation

Run from the repository clone directory:

    conda env create -f environment.yml
    conda activate nseg
    pip install .

## Configuration

All configuration files are in the `nseg/conf` directory. To make config changes you can either change the files themselves or pass config paths or option overrides through the [Hydra] (https://hydra.cc/docs/advanced/hydra-command-line-flags/) CLI of the entry point scripts.

## Data preparation (j0126)

Run in this order:

    python -m nseg.scripts.rename_gt
    python -m nseg.scripts.zarrify_j0126_gt
    python -m nseg.scripts.create_data_split

## Training

    python -m nseg.train_mtlsd

## Small-scale inference and evaluation (small cubes)

    nseg-segment

## Large-scale inference and evaluation (whole dataset or ROI)

    python -m nseg.inference.full_eval
