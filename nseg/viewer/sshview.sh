#!/bin/bash

# Opens a remote neuroglancer viewer on a remote path via ssh
# and tunnels the connection for a local web browser.
# A locally accessible URL is printed to stdout.
# The tunnel and the server are both closed when this script terminates.

# Usage example:
# $ bash nseg/viewer/sshview.sh -f /u/mdraw/lsdex/v1/eval_zarr/results_04-28_16-04_uniform-region_j0126_gt_28.zarr -d ''

set -euo pipefail

if [ -z "$1" ]; then
  echo "Please give a remote file path and any other option as arguments"
  exit 1
fi

HOST=cajal
PORT=43434

# Run a neuroglancer viewer on cajal01 per ssh
# ssh $HOST "/cajal/nvmescratch/users/mdraw/anaconda/envs/nseg/bin/python -m nseg.viewer.nview --no-browser $@"

echo " == sshview: Opening tunnel to port $PORT"
ssh -nNT $HOST -L $PORT:localhost:$PORT &

## TODO: We need to know the random URL of the viewer first to enable this
# echo " == sshview: Launching local web browser"
# xdg-open http://localhost:$PORT/??? &


echo " == sshview: Launching remote server in background"
ssh $HOST "/cajal/nvmescratch/users/mdraw/anaconda/envs/nseg/bin/python -m nseg.viewer.nview --no-browser --port $PORT $@"
