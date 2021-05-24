#!/usr/bin/env bash

# -- DESC: this script contains various tested experiment calls. Instead of executing
#   this file as it is, it is rather thought of a documentation of the investigated
#   experiment settings.


# NOTE: Must contain subfolders 'trainA' and 'trainB'. The `dataset_mode` 'nifti'
#   expects each style folder to contain subject folders in the format of the
#   normalised nifti datasets (e.g. 'normalised_6'). The subject folders may be soft
#   links to the physical subject folders in order to save storage space.
PATH_TO_DATAROOT="/home/daniel/novamia/mv_3d_cycle_gan_new"


# folders in which all experiments have a checkpoint/results folder
PATH_TO_CHECKPOINTS_ROOT="/mnt/data/daniel/3d_pix2pix_cyclegan/checkpoints"
PATH_TO_RESULTS_ROOT="/mnt/data/daniel/3d_pix2pix_cyclegan/results"


cmd_train="python train.py"
cmd_test="python test.py"

common_args=()
common_args+=(--pool_size "4")
common_args+=(--input_nc "1")
common_args+=(--output_nc "1")
common_args+=(--display_port "8078")
common_args+=(--print_freq "1")
common_args+=(--gpu_ids "0,1")
common_args+=(--batchSize "2")
common_args+=(--checkpoints_dir "${PATH_TO_CHECKPOINTS_ROOT}")
common_args+=(--display_id "1")
common_args+=(--dataset_mode "nifti")

common_train_args=()
common_train_args+=(--dataroot "${PATH_TO_DATAROOT}/train")

common_test_args=()
common_test_args+=(--dataroot "${PATH_TO_DATAROOT}/test")
common_test_args+=(--results_dir "${PATH_TO_RESULTS_ROOT}")

continue_train_args=()
continue_train_args+=(--continue_train)
continue_train_args+=(--which_epoch "latest")  # this is for loading correct weights
continue_train_args+=(--epoch_count "101")  # this is for log output only


# ####
# EXPERIMENTS
# ####

# 1. ...
exp_args=()
exp_args+=(--name "mv_hs_lc_3d_cycle_gan_4_test_4_unet128_LCtoHS_relu_nolsgan_niter")
exp_args+=(--model "pix2pix3d")
exp_args+=(--which_model_netG "unet_128")
exp_args+=(--output_activation_G "relu")
exp_args+=(--which_direction "BtoA")

exp_train_args=()
exp_train_args+=(--no_lsgan)
exp_train_args+=(--niter 200)
exp_train_args+=(--niter_decay 200)

"${cmd_train} ${common_args[*]} ${common_train_args[*]} ${exp_args[*]} ${exp_train_args[*]}"
"${cmd_test} ${common_args[*]} ${common_test_args[*]} ${exp_args[*]}"
