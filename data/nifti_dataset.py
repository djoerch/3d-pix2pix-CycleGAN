import os.path

from data.base_dataset import BaseDataset
from glob import glob

from monai.data import CacheDataset
from monai.transforms import CopyItemsd

from episurfsr.data.transforms.factories import make_transform
from episurfsr.config.experiment_configuration import ExperimentType
from episurfsr.data.transforms.data_keys import DataKeys


def get_ranges_from_shape(input_shape, target_shape):
    """Return voxel range per dimension for target shape centered within input_shape."""

    diffs = [sz_in - sz_out for sz_in, sz_out in zip(input_shape, target_shape)]

    assert all(
        [diff > 0 for diff in diffs]
    ), f"Target shape larger than given input shape ({input_shape})."

    output_shape = [
        (diff//2, diff//2 + sz_out)
        for diff, sz_out in zip(diffs, target_shape)
    ]
    return output_shape


class Settings:
    data_exp_type = ExperimentType.MULTI_VIEW
    gt_stack = "highres"
    lr_stack = "coronal"
    target_spacing = (0.8, 0.5, 0.5)
    rescaling_strategy = "STD_UINT16"

    FACTOR = 1000
    SHAPE = (128, 256, 256)


class NiftiDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

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
        """returns samples of dimension [channels, z, x, y]"""

        sample = self.ds[index]

        r_x, r_y, r_z = get_ranges_from_shape(
            input_shape=sample[DataKeys.DATA_KEY_GT_STACK].shape[1:],
            target_shape=Settings.SHAPE,
        )

        return dict(
            A=sample[DataKeys.DATA_KEY_GT_STACK][
                 :, r_x[0]:r_x[1], r_y[0]:r_y[1], r_z[0]:r_z[1]
             ] / Settings.FACTOR,
            B=sample[DataKeys.DATA_KEY_LR_STACK][
                 :, r_x[0]:r_x[1], r_y[0]:r_y[1], r_z[0]:r_z[1]
             ] / Settings.FACTOR,
            A_paths=self.ds.data[index][DataKeys.DATA_KEY_SUBJECT_PATH],
            B_paths=self.ds.data[index][DataKeys.DATA_KEY_SUBJECT_PATH],
            affine=sample['nifti'].slicer[
                r_x[0]:r_x[1], r_y[0]:r_y[1], r_z[0]:r_z[1]
            ].affine,
        )

    def __len__(self):
        return len(self.ds)

    def name(self):
        return 'NiftiDataset'
