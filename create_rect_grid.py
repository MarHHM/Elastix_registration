import subprocess
from pathlib import Path
import itertools

import nibabel as nib
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

import SimpleITK as sitk  # see doc at     https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html

from collections import defaultdict

####################################

def myshow(img, title=None, margin=0.05, dpi=80):
    nda = sitk.GetArrayViewFromImage(img)
    spacing = img.GetSpacing()
    ysize = nda.shape[0]
    xsize = nda.shape[1]
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi
    fig = plt.figure(title, figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
    extent = (0, xsize * spacing[1], 0, ysize * spacing[0])
    t = ax.imshow(nda,
                  extent=extent,
                  interpolation='hamming',
                  cmap='gray',
                  origin='lower')
    if (title):
        plt.title(title)
    plt.show()

############################################################################################

if __name__ == '__main__' :
    data_format = 'nii'
    vox_spacing = (1.0, 1.0, 1.0)        # voxel spacing (aka size) in mm, make sure it's like the spacing of the input images to registeration when read through sitk
    sigma = 0.01         # def: 0.25 (the less, the sharper lines)
    n_dim = 3

    dataset_path = Path("S:/datasets/s1_3/")
    gridIm_itk = sitk.GridSource(   size=(161, 191, 166),                        # size of the lattice (in pix)
                                    spacing = vox_spacing,                       # voxel spacing (aka size) in mm
                                    gridSpacing=10*np.array(vox_spacing),             # space bet grid lines (in pix)
                                    sigma=(sigma,)*n_dim,                           # (list multiplication) thickness of the grid lines (less is thinner lines (i.e. tighter gaussian))
                                    gridOffset=(0.0, 0.0, 0.0),
                                    outputPixelType=sitk.sitkUInt8 )  # def: sitkUInt16
    # gridIm_itk = sitk.GridSource(   size=(576, 468),                        # size of the lattice (in pix)
    #                                 spacing = vox_spacing,
    #                                 gridSpacing= 10*np.array(vox_spacing),             # space bet grid lines (in pix), convert to ndarray to avoid list multiplication ;)
    #                                 sigma=(sigma,)*n_dim,                           # (list multiplication) thickness of the grid lines (less is thinner lines (i.e. tighter gaussian))
    #                                 gridOffset=(0.0, 0.0),
    #                                 outputPixelType=sitk.sitkUInt8,  # def: sitkUInt16
    #                                   )

    gridIm_itk = sitk.RescaleIntensity(gridIm_itk, 0, 1)            # as sitk.GridSource outputs white as 255 & I want it to be 1 (the same i use for binary masks)

    # matplotlib.use('Qt5Agg')              # to specify which backend matplotlib uses (e.g. Qt5Agg - Gtk3Agg - TkAgg - WxAgg - Agg)
    # myshow(gridIm_itk, 'Grid Input')        # won't work with 3d data

    # # save as .nii
    if data_format == 'nii':
        affine_chosen =  nib.load(f"{dataset_path}/I_f__57_MK_UTE_FA4_noWeight__mask_allBones__cropd.nii").affine         # any data from the same subject is ok (all should be having the same "image-to-world transform")
        gridIm_nii = nib.Nifti1Image(sitk.GetArrayFromImage(gridIm_itk).swapaxes(0, 2),         # swapping the local im coords (aka. lattice) due to diff bet nibabel & itk image axes
                                     affine_chosen)
        gridIm_nii.to_filename(f'{dataset_path}/gridIm_{sigma}_3d.nii')
    else:
        sitk.WriteImage(gridIm_itk, f"{dataset_path}/gridIm.{data_format}")
