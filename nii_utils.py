import os
from pathlib import Path
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt

############################################################################

def fix_nii_exported_from_Amira(originalFile_name, wrongFile_name):
    original_im = nib.load(originalFile_name)
    wrong_im = nib.load(wrongFile_name)

    # to fix the .nii exported from Amira:
    #   1- you might need to flip the lattice (i.e. local image coords) or not (according to if flipping happened in
    #       readNifti() for this dataset (to fix issue #7725) or not)
    #   2- copy the affine transform (i.e. global scanner RAS+ coords) from the original data

    # corrected_im = nib.Nifti1Image(np.flip(wrong_im.get_data()), original_im.affine)      # works for Sub_3-2018-09-11
    corrected_im = nib.Nifti1Image(wrong_im.get_data(), original_im.affine)                 # works for Sub_7-2019-07-05
    corrected_im.to_filename(f'FIXED-{wrongFile_name}')

def replace_transform_nii(wrongFile_name):
    # correct_affine = np.array([[-1.25, 0, 0, -59.66099548],                     # FOR SUB_3 only (use another one for sub_7)
    #                   [  0, 0, 1.25, 43.81939697],
    #                   [  0,          1.25,         0,          21.21069336],
    #                   [  0,           0,           0,           1        ]])
    correct_affine = nib.load(f'C:/Users/bzfmuham/Desktop/R4_mask_allBones_xtndVolsEqual__defTo__R1_mask_allBones_xtndVolsEqual___at_rgAln-femur__4x2000__useLndmrks-False__maxStpLn-100x.nii').affine
    wrong_im = nib.load(str(wrongFile_name))

    # corrected_im = nib.Nifti1Image(np.flip(wrong_im.get_data()), original_im.affine)
    corrected_im = nib.Nifti1Image(wrong_im.get_data(), correct_affine)
    # os.chdir('./AFTER_CROPPING/')
    corrected_im.to_filename(f'{wrongFile_name.stem}_trnsfrmCrrctd_2.nii')

def extract_2d_sag_slice(dataset_folder, dataset_file, sagittal_slc_idx, output_dtype):
    os.chdir(dataset_folder)
    dataset = nib.load(dataset_file)

    ## the numpy array might have been read in the same or reverse order compared to Amira (see which gives the right slice)
    # sagital_slc = dataset.get_data()[:,:,(dataset.shape[2]-1)-sagittal_slc_idx]        # the numpy array is read in the reverse order compared to Amira
    sagital_slc = dataset.get_data()[:,:,sagittal_slc_idx]        # the numpy array is read in the reverse order compared to Amira

    sagital_slc_3d = np.zeros([*sagital_slc.shape,1], dtype=output_dtype)               # must convert it to 3d ndarray, otherwise elastix complains about bad determinant !
    sagital_slc_3d[:,:,0] = sagital_slc
    # plt.imshow(sagital_slc)
    # plt.show()
    im = nib.Nifti1Image(sagital_slc_3d, dataset.affine)
    im.to_filename(f"{dataset_file.stem}__slc-{sagittal_slc_idx}.nii")


#########################################################################

if __name__ == '__main__':

    # wrong_files = (Path("R1_mask_allBones_xtnd__dilated.nii"),)
    #
    # for wrong_file_name in wrong_files:
    #     os.chdir('S:/datasets/Sub_3')
    #     fix_nii_exported_from_Amira("R1_mask_allBones_xtnd.nii", wrong_file_name)
    #     # replace_transform_nii(wrong_file_name)

    ############################################################

    ## read data
    # data_name = 'R1_t1-minus-t2.nii'
    # img = nib.load(data_name)                                                                  # 'R4_t1-minus-t2.nii' 'R1_wholeLegMask.labels.nii'
    # clipped_im = nib.Nifti1Image(img.get_data()[0:151, 42:158, 13:121], img.affine)  # works for Sub_3-2018-09-11
    # clipped_im.to_filename(f'CLIPPED-{data_name}')

    ####################################################

    for dataset_file in (Path("R1_mask_allBones_xtnd.nii"), Path("R4_mask_allBones_xtnd.nii"), Path("gridIm_0.01_xtnd.nii")):
        extract_2d_sag_slice("S:/datasets/Sub_3", dataset_file, 94, np.uint8)


