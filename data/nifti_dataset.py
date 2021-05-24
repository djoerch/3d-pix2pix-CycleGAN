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


from monai.data import CacheDataset
from monai.transforms import CopyItemsd

from episurfsr.data.transforms.factories import make_transform
from episurfsr.config.experiment_configuration import ExperimentType
from episurfsr.data.transforms.data_keys import DataKeys


class Settings:
    data_exp_type = ExperimentType.MULTI_VIEW
    gt_stack = "highres"
    lr_stack = "coronal"
    target_spacing = (0.8, 0.5, 0.5)
    rescaling_strategy = "STD_UINT16"

    FACTOR = 1000


class NiftiDataset(BaseDataset):
    def initialize(self, opt):

        # set parameters for image samples (i.e. whole images)
#        self.shape = (128, 256, 256)
#        self.spacing = (1, 0.5, 0.5)
#        self.params = dict(order=3, mode="constant", cval=0)

        self.opt = opt
        self.root = opt.dataroot

#        self.paths_A = sorted(glob(os.path.join(self.root, "trainA", "*.nii.gz")))
#        self.paths_B = sorted(glob(os.path.join(self.root, "trainB", "*.nii.gz")))

#        assert len(self.paths_A) == len(self.paths_B), "Inconsistent number of images."

        transforms = make_transform(Settings())
        transforms.transforms = transforms.transforms[:-1]  # remove `SelectItemsd` to retrieve affine later.
        transforms.transforms = transforms.transforms[:3] \
            + (CopyItemsd(keys=[DataKeys.DATA_KEY_GT_STACK], times=1, names=["nifti"]),) \
            + transforms.transforms[3:]

        self.ds = CacheDataset(
            data=[
                {DataKeys.DATA_KEY_SUBJECT_PATH: path}
                for path in glob(os.path.join(self.root, "*"))
            ],
            transform=transforms,
            num_workers=4,
        )

    def __getitem__(self, index):
        #returns samples of dimension [channels, z, x, y]

#        path_A = self.paths_A[index]
#        path_B = self.paths_B[index]

#        img_A = nib.load(path_A)
#        img_B = nib.load(path_B)

        # resampling to common voxel space
        # NOTE: assuming 'trainB' is lowres, 'trainA' is highres.
#        img_A = nib.processing.resample_to_output(
#            in_img=img_A,
            # NOTE: `to_vox_map` can be NiftiImage or sequence `(shape, affine)`.
#            voxel_sizes=self.spacing,
#            **self.params,
#        )
#        img_A = nib.processing.resample_from_to(
#            from_img=img_A,
#            # NOTE: `to_vox_map` can be NiftiImage or sequence `(shape, affine)`.
#            to_vox_map=(self.shape, img_A.affine),
#            **self.params,
#        )
#        img_B = nib.processing.resample_from_to(
#            from_img=img_B,
#            # NOTE: `to_vox_map` can be NiftiImage or sequence `(shape, affine)`.
#            to_vox_map=img_A,
#            **self.params,
#        )
#
#        # get data array from nifti image
#        data_A = img_A.get_fdata()[None, :, :, :]
#        data_B = img_B.get_fdata()[None, :, :, :]
#
#        # rescale image data to range (0, 1), where (p90->1)
#        data_A = (data_A - data_A.min())
#        data_A /= np.percentile(data_A, 90)
#        data_B = (data_B - data_B.min())
#        data_B /= np.percentile(data_B, 90)  # NOTE: need percentile of the 'positive' array to avoid division by 0.
#
#        # make torch tensor
#        data_A = torch.from_numpy(data_A)
#        data_B = torch.from_numpy(data_B)

#        return {
#            'A': data_A,
#            'B': data_B,
#            'affine': img_A.affine,
#            'A_paths': path_A,
#            'B_paths': path_B,
#        }

        sample = self.ds[index]

#        print(sample[DataKeys.DATA_KEY_GT_STACK].shape)
#        print(sample[DataKeys.DATA_KEY_LR_STACK].shape)

        return {
            'A': sample[DataKeys.DATA_KEY_GT_STACK][:, :128, :256, :256] / Settings.FACTOR,
            'B': sample[DataKeys.DATA_KEY_LR_STACK][:, :128, :256, :256] / Settings.FACTOR,
            'A_paths': self.ds.data[index][DataKeys.DATA_KEY_SUBJECT_PATH],
            'B_paths': self.ds.data[index][DataKeys.DATA_KEY_SUBJECT_PATH],
            'affine': sample['nifti'].affine,
        }

    def __len__(self):
#        return len(self.paths_A)
        return len(self.ds)

    def name(self):
        return 'NiftiDataset'
