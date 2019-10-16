import os
import pathlib as Path

import nibabel as nib
import numpy as np


def fix_nii_exported_from_Amira(originalFile_name, wrongFile_name):
    original_im = nib.load(originalFile_name)
    wrong_im = nib.load(wrongFile_name)

    # to fix the .nii exported from Amira:
    #   1- you might need to flip the lattice (i.e. local image coords) or not (according to if flipping happened in
    #       readNifti() for this dataset (to fix issue #7725) or not)
    #   2- copy the affine transform (i.e. global scanner RAS+ coords) from the original data

    corrected_im = nib.Nifti1Image(np.flip(wrong_im.get_data()), original_im.affine)      # works for Sub_3-2018-09-11
    # corrected_im = nib.Nifti1Image(wrong_im.get_data(), original_im.affine)                 # works for Sub_7-2019-07-05
    corrected_im.to_filename(f'FIXED-{wrongFile_name}')



def replace_transform_nii(wrongFile_name):
    correct_affine = np.array([[-1.25, 0, 0, -59.66099548],                     # FOR SUB_3 only (use another one for sub_7)
                      [  0, 0, 1.25, 43.81939697],
                      [  0,          1.25,         0,          21.21069336],
                      [  0,           0,           0,           1        ]])
    wrong_im = nib.load(wrongFile_name)

    # corrected_im = nib.Nifti1Image(np.flip(wrong_im.get_data()), original_im.affine)
    corrected_im = nib.Nifti1Image(wrong_im.get_data(), correct_affine)
    os.chdir('./AFTER_CROPPING/')
    corrected_im.to_filename(f'{wrongFile_name}')

#########################################################################

if __name__ == '__main__':
    wrong_files = (Path('R1_mask_allBones.nii'),)

    for wrong_file_name in wrong_files:
        os.chdir('S:/datasets/Sub_3')
        replace_transform_nii(wrong_file_name)

    ############################################################

    ## read data
    # data_name = 'R1_t1-minus-t2.nii'
    # img = nib.load(data_name)                                                                  # 'R4_t1-minus-t2.nii' 'R1_wholeLegMask.labels.nii'
    # clipped_im = nib.Nifti1Image(img.get_data()[0:151, 42:158, 13:121], img.affine)  # works for Sub_3-2018-09-11
    # clipped_im.to_filename(f'CLIPPED-{data_name}')



