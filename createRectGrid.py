import subprocess
from pathlib import Path
import itertools

import nibabel as nib
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

import SimpleITK as sitk  # see doc at     https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html
import call_transformix

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
    sigma = 0.01         # def: 0.25 (the less, the sharper lines)
    n_dim = 3

    dataset_path = Path("S:/datasets/Sub_3")
    gridIm_itk = sitk.GridSource(   size=(256, 182, 142),                        # size of the lattice (in pix)
                                    spacing = (1.25, 1.25, 1.25),                       # voxel spacing (aka size) in mm
                                    gridSpacing=(1.0, 1.0, 1.0),             # space bet grid lines (in pix)
                                    sigma=(sigma,)*n_dim,                           # (list multiplication) thickness of the grid lines (less is thinner lines (i.e. tighter gaussian))
                                    gridOffset=(0.0, 0.0, 0.0),
                                    outputPixelType=sitk.sitkUInt8,  # def: sitkUInt16
                                      )

    # matplotlib.use('Qt5Agg')              # to specify which backend matplotlib uses (e.g. Qt5Agg - Gtk3Agg - TkAgg - WxAgg - Agg)
    # myshow(gridIm_itk, 'Grid Input')        # won't work with 3d data

    # # save as .nii
    affine_chosen =  nib.load(f"{dataset_path}/R1_vol_uncropd.nii").affine         # any data from the same subject is ok (all should be having the same "image-to-world transform")
    gridIm_nii = nib.Nifti1Image(sitk.GetArrayFromImage(gridIm_itk).swapaxes(0, 2),         # swapping the local im coords (aka. lattice) due to diff bet nibabel & itk image axes
                                 affine_chosen)
    gridIm_nii.to_filename(f'{dataset_path}/gridIm_{sigma}_3dVol.nii')