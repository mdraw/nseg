wandb report link: https://api.wandb.ai/links/mdth/311xg0j9

## Notes

- All trainings use the baseline zebrafinch 3D U-Net implementation from the LSD paper's official code, including all relevant hyperparameters, minus the `ElasticAugment` transformations, which had been causing random crashes during training.
- All trainings were started 3 times with identical configuration each but different random seeds (repeated runs have the "_q2" and "_q3" prefixes).
- Scalar metrics of repeated (q2, q3) runs are grouped in wandb and the standard deviation of the values is displayed as colored areas around the lines
- Training data dirs given below are relative to `/cajal/scratch/projects/misc/mdraw/data/`
- Code is at https://github.com/mdraw/nseg - currently a private repo but anyone interested can request read permissions
- Dataset configs are in the nseg repo's `nseg/conf/dataset/` directory


## Details on experimental settings

### GT
- dataset config: `j0126.yaml`
- training data dir: `j0126_gt_zarr_split_c350/training/`

### Syn.
- dataset config: `synth3000mr.yaml`
- training data dir: `synth3000mr/`

### Co-train Syn. + GT (1:1)
- dataset config: `combined_j0126_synth_r1.yaml`
- training data dir: `combined_j0126_and_synth430mr_r1/`

### Co-train Syn. + GT (2:1)
- dataset config: `combined_j0126_synth_r2.yaml`
- training data dir: `combined_j0126_and_synth550mr_r2/`

### Syn. (real seg.)
- dataset config: `halfsynth.yaml`
- training data dir: `halfsynth_zarr_all_training/`

### Co-train Syn. (real seg.) + GT (1:1)
- dataset config: `combined_j0126_hsynth_tr_r1.yaml`
- training data dir: `combined_j0126_and_halfsynth_zarr_all_training_r1/`

### Co-train Syn. (real seg.) + GT (2:1)
- dataset config: `combined_j0126_hsynth_tr_r2.yaml`
- training data dir: `combined_j0126_and_halfsynth_zarr_all_training_r2/`