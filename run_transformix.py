import os
import subprocess
import sys
from pathlib import Path

import SimpleITK as sitk  # see doc at     https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html
import nibabel as nib

##################################################################

#region->N.B.
    # - if the input is binary (e.g mask & rect_grid), make sure the that transform paramter file has (FinalBSplineInterpolationOrder 0) & (ResultImagePixelType "unsigned char") (& vice versa for texture input)
#endregion


dataset_path = Path("S:/datasets/s3_2/2d_highRes/")
transform_params_folder = r'registeration of texture\4x2k__sim-NMI__Spc-4__bendEnergyPen\4x2000__I_f-slc_24__I_m-slc_25__sim-NMI__Spc-4__bendEnergyPen'       # '4x10000__I_m-slc_13__sim-MI__Spc-4'
data_format = 'tif'

subprocess.run(["transformix.exe",
                # "-in", f"{dataset_path}/gridIm.{data_format}",
                "-in", f"{dataset_path}/sag1__Hoffa_mask/tPoint_25_mask_hoffa.{data_format}",
                "-tp", f"{dataset_path}/{transform_params_folder}/TransformParameters.0__for_binary_inputs.txt",
                # "-tp", f"{dataset_path}/{transform_params_folder}/TransformParameters.0.txt",
                "-out", f"{dataset_path}/{transform_params_folder}"
                ])

for im_to_correct_info in (
                           # (f"result.{data_format}", f"grid_deformed---{transform_params_folder}.{data_format}"),
                           (f"result.{data_format}", f"mask_Hoffa_pad__deformed---{transform_params_folder}.{data_format}"),
                          ):
    if data_format == "nii":
        nii_utils.correct_transform(dataset_path, elx_output_folder, I_f_filename, *im_to_correct_info)
    else:
        os.replace(Path(f"{dataset_path}/{transform_params_folder}/{im_to_correct_info[0]}"),
                   Path(f"{dataset_path}/{transform_params_folder}/{im_to_correct_info[1]}"))  # just renaming