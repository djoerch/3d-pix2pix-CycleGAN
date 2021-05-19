#!/usr/bin/env bash

PATH_TO_DATA_ROOT="/home/daniel/novamia/mv_3d_cycle_gan/train"
EXPERIMENT="mv_hs_lc_3d_cycle_gan_2"


cmd="python train.py"
args=()
args+=(--dataroot "${PATH_TO_DATA_ROOT}")
args+=(--name "${EXPERIMENT}")
args+=(--model "pix2pix3d")
args+=(--pool_size 4)
args+=(--dataset_mode "nifti")
args+=(--which_model_netG "unet_96")
args+=(--input_nc 1)
args+=(--output_nc 1)
args+=(--display_port 8078)
args+=(--print_freq 1)

echo "Running command: \n ${cmd} ${args[@]}"
${cmd} ${args[@]}

