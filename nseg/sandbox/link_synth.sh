#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Please pass dataset size spec (430 or 550)."
  exit 1
fi

SIZE=$1

if [ $SIZE  -eq  430 ]; then
    RATIO=1
elif [[ $SIZE  -eq  550 ]]; then
    RATIO=2
else
    echo "Please pass dataset size spec (430 or 550)."
    exit 1;
fi


SYN_SRC=/cajal/nvmescratch/users/mdraw/data/synth${SIZE}mr/synth${SIZE}mr.zarr
DEST_DIR=/cajal/scratch/projects/misc/mdraw/data/combined_j0126_and_synth${SIZE}mr_r${RATIO}
DEST_STEM=synth${SIZE}mr

NUM_REAl_CUBES=21

NUM_SYNTH_CUBE_COPIES=$(("${NUM_REAl_CUBES} * ${RATIO}"))

mkdir ${DEST_DIR}

for COPY_IDX in $(seq ${NUM_SYNTH_CUBE_COPIES}); do
    echo ${SYN_SRC} ${DEST_DIR}/${DEST_STEM}_copy${COPY_IDX}.zarr
    ln -s ${SYN_SRC} ${DEST_DIR}/${DEST_STEM}_copy${COPY_IDX}.zarr
done


REAL_SRC_DIR=/cajal/scratch/projects/misc/mdraw/data/j0126_gt_zarr_split_c350/training/

for RF in ${REAL_SRC_DIR}/*.zarr; do
    echo ${RF} ${DEST_DIR}/
    ln -s ${RF} ${DEST_DIR}/
done
