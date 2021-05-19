#!/usr/bin/env bash

PATH_TO_DATA_ROOT="/home/daniel/novamia/mv_3d_cycle_gan/test"
EXPERIMENT="mv_hs_lc_3d_cycle_gan_2"

cmd="python test.py"
args=()
args+=(--dataroot "${PATH_TO_DATA_ROOT}")
args+=(--name "${EXPERIMENT}")
args+=(--model "pix2pix3d")
args+=(--dataset_mode "nifti")
args+=(--which_model_netG "unet_96")
args+=(--input_nc 1)
args+=(--output_nc 1)
args+=(--display_port 8078)

echo "Running command: \n ${cmd} ${args[@]}"
${cmd} ${args[@]}

