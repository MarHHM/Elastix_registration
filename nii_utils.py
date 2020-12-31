import os
from pathlib import Path
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk

############################################################################

def fix_nii_exported_from_Amira(dataset_path, im_with_crrct_trnsfrm__name, im_with_wrng_trnsfrm__name):
    original_im = nib.load(f'{dataset_path}/{im_with_crrct_trnsfrm__name}')
    wrong_im = nib.load(f'{dataset_path}/{im_with_wrng_trnsfrm__name}')

    # to fix the .nii exported from Amira:
    #   1- you might need to flip the lattice (i.e. local image coords) or not (according to if flipping happened in
    #       readNifti() for this dataset (to fix issue #7725) or not)
    #   2- copy the affine transform (i.e. global scanner RAS+ coords) from the original data

    # corrected_im = nib.Nifti1Image(np.flip(wrong_im.get_data()), original_im.affine)      # works for Sub_3-2018-09-11
    corrected_im = nib.Nifti1Image(wrong_im.get_data(), original_im.affine)                 # works for Sub_7-2019-07-05
    corrected_im.to_filename(f'{dataset_path}/FIXED-{im_with_wrng_trnsfrm__name}')

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

def correct_transform(dataset_path, output_dir, srcOfCorrectTransform, imToCorrect, correctedImNewName):
    cwd = os.getcwd()
    os.chdir(dataset_path)

    im_to_correct = sitk.ReadImage(f'{output_dir}/{imToCorrect}')
    if im_to_correct.GetMetaData('bitpix') in ("8", "16"):                                          # as sitk might read 8-bit unsigned as 16-bit signed
        im_to_correct = sitk.Cast(im_to_correct, sitk.sitkUInt8)
    os.remove(f"{output_dir}/{imToCorrect}")                                                    # as it will be replaced by the corrected result in the following
    im_to_correct__arr = sitk.GetArrayFromImage(im_to_correct)

    if "nii" in imToCorrect:                                                                    # write result using nibabel (not sitk.WriteImage() as it screws the header (sform_code & qform_code & quaternion))
        corrected_im = nib.Nifti1Image(im_to_correct__arr.swapaxes(0, 2),                       # swapping the local im coords (aka. lattice) due to diff bet nibabel & itk image axes
                                        nib.load(srcOfCorrectTransform).affine)                                    # use the image resulting from elx registration with the affine transfrom coming from the original data (e.g. fixed im)
        corrected_im.to_filename(f'{output_dir}/{correctedImNewName}')
    else:               # e.g. if image is png --> # write directly though sitk
        sitk.WriteImage(im_to_correct__arr, f'{output_dir}/{correctedImNewName}')

    os.chdir(cwd)                                                                                       # back to the working directory before entering this method

# x,y,z are abitrary (just compare the limits whith what you delineate in Amira's crop editor)
def nii__crop_scan(dataset_path, scan_name, crop_x_lims, crop_y_lims, crop_z_lims):
    scan = nib.load(f"{dataset_path}/{scan_name}.nii")
    nda = scan.get_data()
    nda_crpd = nda[ crop_x_lims[0]:crop_x_lims[1],
                    crop_y_lims[0]:crop_y_lims[1],
                    crop_z_lims[0]:crop_z_lims[1]]
    scan_crpd = nib.Nifti1Image(nda_crpd, scan.affine)
    scan_crpd.to_filename(f"{dataset_path}/{scan_name}_crpd.nii")


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

    # for dataset_file in (Path("2d_highRes.nii"),):
    #     extract_2d_sag_slice("S:/datasets/s3_2/2d_highRes", dataset_file, 0, np.float32)

    #region    Extract 2d slice from s3_2\2d_highRes (in future, make it a general function)
    # slc_idx = 13
    #
    # os.chdir(r"S:\datasets\s3_2\2d_highRes")
    # # data = sitk.ReadImage("2d_highRes.nii", sitk.sitkFloat32) #ERROR: can't interpret transform (det = 0) !!
    # data = nib.load("2d_highRes.nii")
    # # WARNING: get_fdata() suffers from truncating the max intensity for some slices (e.g. [:, :, 1, 50])
    # im = data.get_fdata(caching="fill", dtype=np.float32)[:, :, 1, slc_idx]
    # ## both get_unscaled() & get_data() still don't solve the max intensity truncation experienced with some slices in get_fdata()
    # # im = data.dataobj.get_unscaled()[:, :, 1, slc_idx]
    # # im = data.get_data()[:, :, 1, slc_idx]
    # im = np.swapaxes(im, 0, 1)
    # sitk.WriteImage(sitk.GetImageFromArray(im), f"slc_{slc_idx}_extended_tex.tif")
    #endregion

    #%%
    # correct_transform(dataset_path =r'S:\datasets\s1_3',
    #                   output_dir = r'S:\datasets\s1_3',
    #                   srcOfCorrectTransform = 'I_f__57_MK_UTE_FA4_noWeight_tex__cropd.nii',
    #                   imToCorrect = 'I_f___57_MK_UTE_FA4_noWeight_tex__cropd2.labels.binary.nii',
    #                   correctedImNewName = 'I_f___57_MK_UTE_FA4_noWeight_mask_allBones__cropd.nii')

    #%%
    nii__crop_scan(r"S:\datasets\s4_2", "FIXED-im_middle.labels_tibia",
                   crop_x_lims=(110, 280),      # i get these values by manually delinieating edges on Amira's crop editor
                   crop_y_lims=(40, 155),
                   crop_z_lims=(20, 133))