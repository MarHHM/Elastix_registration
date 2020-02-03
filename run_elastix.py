import nibabel as nib
import os
import subprocess

###############################################################

def correct_transform(outFldr, srcOfCorrectTransform, imToCorrect, correctedImNewName):
    orgAffine = nib.load(srcOfCorrectTransform).affine  # load the original transform to use it for the generated elastix result
    im = nib.load(f"{outFldr}/{imToCorrect}")
    im.set_sform(orgAffine);       im.set_qform(orgAffine)
    im.to_filename(f"{outFldr}/{correctedImNewName}")           # can't overwrite orginal image..
    os.remove(f"{outFldr}/{imToCorrect}")

################################################################

# 2D
outFldr = "result_2d_vol__meansSqr+rgdPen"              # DRAFT  -  result_meansSqr_rgdPen_2d  -  result_meansSqr_rgdPen  -  result_meansSqr_rgdPen_2d_with_fixedMask  -
parMapFile = "Par_2d.txt"              # Par0016.multibsplines.lung.sliding.txt  -  Par_3d.txt
i_f = "R1_2d_vol_uncropd__slc-60.bmp"                   # R1_mask_allBones_xtnd  -  R1_2d
i_m = "R4_2d_vol_uncropd__slc-60.bmp"                   # R4_mask_allBones_xtnd  -  R4_2d
i_grid = "gridIm_0.01_uncropd__slc-60.bmp"       # gridIm_0.01  -  gridIm_0.01_2d

# ## 3D
# outFldr = "result_3d_mask__meansSq+MultiBspln"              # DRAFT  -  result_meansSqr_rgdPen_2d  -  result_meansSqr_rgdPen  -  result_meansSqr_rgdPen_2d_with_fixedMask  -
# parMapFile = "Par_3d.txt"              # Par0016.multibsplines.lung.sliding.txt  -  Par_3d.txt
# i_f = "R1_mask_allBones_xtnd.nii"                   # R1_mask_allBones_xtnd.nii  -  R1_2d.bmp  -  R1_vol_uncropd
# i_m = "R4_mask_allBones_xtnd.nii"                   # R4_mask_allBones_tnd.nii  -  R4_2d.bmp
# i_grid = "gridIm_0.01.nii"       # gridIm_0.01.nii  -  gridIm_0.01_2d.bmp  -  gridIm_0.01_3dVol_uncropd.nii

################################################################

os.chdir("S:/datasets/Sub_3/")
subprocess.run(["elastix.exe",
                "-f", i_f, "-m", i_m, "-p", parMapFile, "-labels", "R1_mask_allBones_xtnd__dilated.nii", "-out", outFldr])
subprocess.run(["transformix.exe",
                "-in", i_grid, "-tp", f"{outFldr}/TransformParameters.0.txt", "-def", "all", "-jac", "all", "-out", outFldr])

# To correct "image-to-world" transform, as elastix doesn't preserve it while writing nifti
# Won't work with bmp
correct_transform(outFldr, i_f, "result.0.nii", "elxRslt.nii")
correct_transform(outFldr, i_f, "spatialJacobian.nii", "sptlJac.nii")
correct_transform(outFldr, i_f, "result.nii", "defGrid.nii")