wandb report link: https://api.wandb.ai/links/mdth/311xg0j9

## Important note on agglomeration results in the training logs
**tl;dr:**
  - VOI scores in the wandb logs look bad due to a wrong threshold.
  - The trainings are okay, just the eval was broken. No need to re-run trainings. Doing offline eval with better settings leads to reasonable results.
  - E.g. last snapshot eval result on the **test set** (200k iterations) with improved agglomeration parameters was:
    - "GT": mean VOI 1.3029
    - "Co-train Syn. + GT (2:1)": mean VOI: 1.0783
    - ... other settings and standard deviations will follow

**long version**:

- For fragment agglomeration the settings from the official LSD code base are used (`waterz` with `score_function` `hist_quant_75` and threshold 1.0). I have now found out that these are **far from optimal and result in bad VOI scores**. Due to the too-high threshold of 1.0 most fragments were merged into one giant component.
- Keeping the score_function at hist_quant_75, I have determined the optimal threshold is at ~ 0.01 (by sweeping over different threshold values on the validation cube set).
- The trainings themselves, losses, affinities and fragments are not affected at all. Just the final segmentation agglomeration and resulting VOIs are much worse than they need to be.

Here are some test cube set results on manually chosen (configs A..D) and a value based on tuning on the validation set (config E):

- config A:
  - merge_function: hist_quant_75
  - threshold: 1.0
- config B:
  - merge_function: mean
  - threshold: 0.043
- config C:
  - merge_function: hist_quant_75
  - threshold: 0.5
- config D:
  - merge_function: hist_quant_75
  - threshold: 0.1
- config E:
  - merge_function: hist_quant_75
  - threshold: 0.01


### Step 158000

- (setting "GT") `/u/mdraw/lsdex/v1/train_mtlsd/05-10_01-03_noisy-wrap/model_checkpoint_158000.pth`
  - Mean VOI with config A: 2.8204
  - Mean VOI with config B: 1.5041
  - Mean VOI with config C: 2.8204
  - Mean VOI with config D: 1.7527
  - Mean VOI with config E: 1.2635


- (setting "Co-train Syn. + GT (2:1)") `/u/mdraw/lsdex/v1/train_mtlsd/05-10_22-44_coral-jigsaw_cotrain_r2__q3/model_checkpoint_158000.pth`
  - Mean VOI with config A: 2.3688
  - Mean VOI with config B: 1.6367
  - Mean VOI with config C: 2.1048
  - Mean VOI with config D: 1.2276
  - Mean VOI with config E: 1.2020


### Step 200000


- (setting "GT") `/u/mdraw/lsdex/v1/train_mtlsd/05-10_01-03_noisy-wrap/model_checkpoint_200000.pth`
  - Mean VOI with config E: 1.3029


- (setting "Co-train Syn. + GT (2:1)") `/u/mdraw/lsdex/v1/train_mtlsd/05-10_22-44_coral-jigsaw_cotrain_r2__q3/model_checkpoint_200000.pth`
  - Mean VOI with config E: 1.0783



## Other notes about the training experiments

- All trainings use the baseline zebrafinch 3D U-Net implementation from the LSD paper's official code, including all relevant hyperparameters, minus the `ElasticAugment` transformations, which had been causing random crashes during training.
- All trainings were started 3 times with identical configuration each but different random seeds (repeated runs have the "_q2" and "_q3" prefixes).
  * Exception: The "real seg." experiments only have incomplete repeated runs. The "_q2" and "_q3" runs were killed because the compute resources were needed for other trainings and "real seg." has been a low priority.
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