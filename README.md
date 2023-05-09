# **nseg** - Improving Neuron Instance Segmentation

## Installation

From project root:

    pip install .

## Data preparation

Run in this order:

    python -m nseg.scripts.rename_gt
    python -m nseg.scripts.zarrify_j0126
    python -m nseg.scripts.create_data_split

## Training

    nseg-train

## Inference

    nseg-segment
