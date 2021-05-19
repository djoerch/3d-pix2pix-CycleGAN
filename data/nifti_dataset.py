import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
#from data.image_folder import make_dataset
import pickle
import numpy as np
from glob import glob
import nibabel as nib
import nibabel.processing


class NiftiDataset(BaseDataset):
    def initialize(self, opt):

        # set parameters for image samples (i.e. whole images)
        self.shape = (128, 256, 256)
        self.spacing = (1, 0.5, 0.5)
        self.params = dict(order=3, mode="constant", cval=0)

        self.opt = opt
        self.root = opt.dataroot

        self.paths_A = glob(os.path.join(self.root, "trainA", "*.nii.gz"))
        self.paths_B = glob(os.path.join(self.root, "trainB", "*.nii.gz"))

        assert len(self.paths_A) == len(self.paths_B), "Inconsistent number of images."

    def __getitem__(self, index):
        #returns samples of dimension [channels, z, x, y]

        path_A = self.paths_A[index]
        path_B = self.paths_B[index]

        img_A = nib.load(path_A)
        img_B = nib.load(path_B)

        # resampling to common voxel space
        # NOTE: assuming 'trainB' is lowres, 'trainA' is highres.
        img_A = nib.processing.resample_to_output(
            in_img=img_A,
            # NOTE: `to_vox_map` can be NiftiImage or sequence `(shape, affine)`.
            voxel_sizes=self.spacing,
            **self.params,
        )
        img_A = nib.processing.resample_from_to(
            from_img=img_A,
            # NOTE: `to_vox_map` can be NiftiImage or sequence `(shape, affine)`.
            to_vox_map=(self.shape, img_A.affine),
            **self.params,
        )
        img_B = nib.processing.resample_from_to(
            from_img=img_B,
            # NOTE: `to_vox_map` can be NiftiImage or sequence `(shape, affine)`.
            to_vox_map=img_A,
            **self.params,
        )

        data_A = torch.from_numpy(img_A.get_fdata()[None, :, :, :])
        data_B = torch.from_numpy(img_B.get_fdata()[None, :, :, :])

        return {
            'A': data_A,
            'B': data_B,
            'affine': img_A.affine,
            'A_paths': path_A,
            'B_paths': path_B,
        }

    def __len__(self):
        return len(self.paths_A)

    def name(self):
        return 'NiftiDataset'
