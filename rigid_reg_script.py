import subprocess
from pathlib import Path
import itertools

import nibabel as nib
import numpy as np

import SimpleITK as sitk  # see doc at     https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html
import call_transformix

from collections import defaultdict

#%%
dataset_path = Path("S:/datasets/Sub_3/")                             # 'S:/datasets/Sub_3/'        # "S:/datasets/Sub_7/"        # "C:/Users/bzfmuham/OneDrive/Knee-Kinematics/elastix 4.9.0/elastix-4.9.0-example_2/exampleinput/"
I_f = 'R3'
I_m = 'R4'

I_f_filename = Path(f"{I_f}_mask_allBones.nii")                                                     # flexed,   "R1_t1-minus-t2.nii"                  '72_t1-minus-t2'
I_m_filename = Path(f"{I_m}_mask_patella.nii")                                      # extended, "R3_t1-minus-t2_rigidlyAligned.nii"   '70_t1-minus-t2_rigidlyAligned'
I_f_mask_filename = Path(f"{I_f}_mask_wholeLeg.nii")                                        # "R1_wholeLegMask.labels.nii"    '72_wholeLegMask.labels.nii' || R1-femur.nii
use_I_m_mask = False
I_m_mask_filename = Path(f'{I_m}______.nii')                                                                # mask_tibia  ||  R4_Patella.nii

reg_type = "RIGID"                                                              # 'RIGID' - 'NON_RIGID'
# n_itr = 1                                                               # 250 -> 2000 (good: 500) -  for all resolutions
# n_res = 1                                                                           # default: 4     (used for all registrations)

# rigid_alignment_transform__filename = f'____________'             # '{I_m}_to_{I_f}__trans__rigid_alignment__femur.txt' ||
use_rigidityPenalty = False
I_m_rigidityCoeffIm_filename = Path(f"{I_m}_mask_allBones.nii")

use_landmarks = False
I_f_landmarks_filename = Path(f"{I_f}_landmarks_femur.pts")                                                               # "Choosing_bone_markers/R1_landmarks.pts"
I_m_landmarks_filename = Path(f"{I_m}_landmarks_femur___transformed_from_{I_f}_landmarks_femur.pts")                                                               # "Choosing_bone_markers/R3_landmarks.pts"


#%% #check if the "image-to-world" affine tranforms are the same for all involved volumes & masks
I_f_affine = nib.load(f'{dataset_path}/{I_f_filename}').affine
I_m_affine = nib.load(f'{dataset_path}/{I_m_filename}').affine
I_f_mask_affine = nib.load(f'{dataset_path}/{I_f_mask_filename}').affine
I_m_rigidityCoeffIm_affine = nib.load(f'{dataset_path}/{I_m_rigidityCoeffIm_filename}').affine
if not ( np.array_equiv(I_f_affine, I_m_affine)
     and np.array_equiv(I_f_affine, I_f_mask_affine)
     and np.array_equiv(I_f_affine, I_m_rigidityCoeffIm_affine)
                                                    ) :
    print('!! ERROR: "image-to-world" affine transforms are not the same for all involved volumes & masks!')
    exit()








#%% MULTI registration
arr__n_itr = (1000,)    # def: (2000, 500, 1000,)
arr__n_res = (3,)               # def: (4, 1, 3, 2)

rigid_alignment_transform__filename = 'blahblah'

for n_itr, n_res in itertools.product(arr__n_itr, arr__n_res):
    I_deformed_filename = Path(f'{I_m_filename.stem}__rgAlnTo__{I_f_filename.stem}___at_'
                               f'__{n_res}x{n_itr}'
                               f'.nii'
                               )
    exit_code = subprocess.call(["python", "callElastix.py",                                        # using a subprocess for each iteration instead of normal function call to solve the "log file issue" (can't be recreated per process) -> see this issue  https://github.com/SuperElastix/SimpleElastix/issues/104
                                str(dataset_path), str(I_f_filename), str(I_m_filename), str(I_f_mask_filename), str(I_m_mask_filename), str(use_I_m_mask),
                                str(rigid_alignment_transform__filename), str(I_m_rigidityCoeffIm_filename), reg_type, str(n_itr),
                                str(n_res), str(use_rigidityPenalty), str(use_landmarks), str(I_f_landmarks_filename), str(I_m_landmarks_filename),
                                str(I_deformed_filename)])
    if exit_code != 0:
        print("ERROR DURING ELASTIX EXECUTION !!  -  skipping to next execution...")
        continue