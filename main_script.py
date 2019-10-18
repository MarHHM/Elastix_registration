import subprocess
from pathlib import Path
import itertools

import nibabel as nib
import numpy as np

import SimpleITK as sitk  # see doc at     https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html
import call_transformix

from collections import defaultdict

#%%
# dataset_path = "S:/datasets/Sub_7/"             # 'S:/datasets/Sub_3/'        # "S:/datasets/Sub_7/"        # "C:/Users/bzfmuham/OneDrive/Knee-Kinematics/elastix 4.9.0/elastix-4.9.0-example_2/exampleinput/"
# I_f_filename = "72_t1-minus-t2.nii"                                     # flexed,   "R1_t1-minus-t2.nii"                  '72_t1-minus-t2'
# I_m_filename = "70_t1-minus-t2_rigidlyAligned.nii"                                     # extended, "R3_t1-minus-t2_rigidlyAligned.nii"   '70_t1-minus-t2_rigidlyAligned'
# I_f_mask_filename = "72_wholeLegMask.labels.nii"                        # "R1_wholeLegMask.labels.nii"    '72_wholeLegMask.labels.nii'
# I_m_rigidityCoeffIm_filename = "70_boneMask_rigidlyAligned.nii"         # "R3_boneMask_rigidlyAligned.nii"    "70_boneMask_rigidlyAligned.nii"
# I_f_landmarks_filename = "---"            # "Choosing_bone_markers/R1_landmarks.pts"
# I_m_landmarks_filename = "---"            # "Choosing_bone_markers/R3_landmarks.pts"
# reg_type = "NON_RIGID"                   # 'RIGID' - 'NON_RIGID' - 'PRE_ALLIGNED_NON_RIGID'
# n_itr_rigid = 0
# n_itr_nonRigid = 2000                      # 250 -> 2000 (good: 500) -  for all resolutions
# n_res = 4                               # default: 4     (used for all registrations)
# use_rigidityPenalty = True
# use_landmarks = False
# I_deformed_filename = f'el - {I_m_filename.split("_")[0]} - use_rigidityPenalty={use_rigidityPenalty} - n_itr={n_itr_nonRigid}.nii'dataset_path = "S:/datasets/Sub_7/"             # 'S:/datasets/Sub_3/'        # "S:/datasets/Sub_7/"        # "C:/Users/bzfmuham/OneDrive/Knee-Kinematics/elastix 4.9.0/elastix-4.9.0-example_2/exampleinput/"


dataset_path = Path("S:/datasets/Sub_3/")                             # 'S:/datasets/Sub_3/'        # "S:/datasets/Sub_7/"        # "C:/Users/bzfmuham/OneDrive/Knee-Kinematics/elastix 4.9.0/elastix-4.9.0-example_2/exampleinput/"
I_f = 'R1'
I_m = 'R4'

I_f_filename = Path(f"{I_f}_mask_allBones.nii")                                                     # flexed,   "R1_t1-minus-t2.nii"                  '72_t1-minus-t2'
I_m_filename = Path(f"{I_m}_mask_allBones.nii")                                      # extended, "R3_t1-minus-t2_rigidlyAligned.nii"   '70_t1-minus-t2_rigidlyAligned'
I_f_mask_filename = Path(f"{I_f}_mask_wholeLeg.nii")                                        # "R1_wholeLegMask.labels.nii"    '72_wholeLegMask.labels.nii' || R1-femur.nii
use_I_m_mask = False
I_m_mask_filename = Path(f'{I_m}______.nii')                                                                # mask_tibia  ||  R4_Patella.nii

reg_type = "NON_RIGID"                                                              # 'RIGID' - 'NON_RIGID'
n_itr_rigid = 0
n_itr_nonRigid = 1                                                               # 250 -> 2000 (good: 500) -  for all resolutions
n_res = 1                                                                           # default: 4     (used for all registrations)

# rigid_alignment_transform__filename = f'____________'             # '{I_m}_to_{I_f}__trans__rigid_alignment__femur.txt' ||
use_rigidityPenalty = True
I_m_rigidityCoeffIm_filename = Path(f"{I_m}_mask_allBones.nii")

use_landmarks = False
I_f_landmarks_filename = Path(f"{I_f}---")                                                               # "Choosing_bone_markers/R1_landmarks.pts"
I_m_landmarks_filename = Path(f"{I_m}---")                                                               # "Choosing_bone_markers/R3_landmarks.pts"


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
arr__rigid_alignment_transform__filename = (Path(f'{I_m}_to_{I_f}__trans__rigid_alignment__allBones.txt'),
                                            Path(f'{I_m}_to_{I_f}__trans__rigid_alignment__femur.txt'),
                                            # Path(f'{I_m}_to_{I_f}__trans__rigid_alignment__tibia.txt'),
                                            # Path(f'{I_m}_to_{I_f}__trans__rigid_alignment__patella.txt')
                                            )
arr__n_itr = (1, 3)

for rigid_alignment_transform__filename,      n_itr_nonRigid in itertools.product(
    arr__rigid_alignment_transform__filename, arr__n_itr) :
    I_deformed_filename = Path(f'{I_m_filename.stem}___deformed_to___{I_f_filename.stem}___'
                               f'at_rigidAlignment={rigid_alignment_transform__filename.stem.split("__")[3]}__n_itr={n_itr_nonRigid}')
    subprocess.call(["python", "callElastix.py",                                        # using a subprocess for each iteration instead of normal function call to solve the "log file issue" (can't be recreated per process) -> see this issue  https://github.com/SuperElastix/SimpleElastix/issues/104
                     str(dataset_path), str(I_f_filename), str(I_m_filename), str(I_f_mask_filename), str(I_m_mask_filename), str(use_I_m_mask),
                     str(rigid_alignment_transform__filename), str(I_m_rigidityCoeffIm_filename), reg_type, str(n_itr_rigid), str(n_itr_nonRigid),
                     str(n_res), str(use_rigidityPenalty), str(use_landmarks), str(I_f_landmarks_filename), str(I_m_landmarks_filename),
                     str(I_deformed_filename)])

    ## deform "I_m-related masks" & calc DSC for each
    overlapFilter = sitk.LabelOverlapMeasuresImageFilter()
    DSC_dict = defaultdict(list)
    # VolSimilarity_dict = defaultdict(list)
    # Jaccard_dict = defaultdict(list)
    pMap_filename = Path(f'{I_deformed_filename.stem}/TransformParameters.0.txt')
    arr__mask_type = ('mask_wholeLeg', 'mask_allBones')
    for mask_type in arr__mask_type:
        im_to_deform___filename = Path(f'{I_m}_{mask_type}.nii')
        output_filename = Path(f'{im_to_deform___filename.stem}__deformed.nii')
        call_transformix.call_transformix(dataset_path, im_to_deform___filename, pMap_filename, output_filename)

        mask_I_f = sitk.ReadImage(f'{dataset_path}/{I_f}_{mask_type}.nii')
        mask_I_m_deformed = sitk.ReadImage(f'{dataset_path}/{I_deformed_filename.stem}/{output_filename.stem}/{output_filename}')
        overlapFilter.Execute(mask_I_f, mask_I_m_deformed)
        DSC_dict[f'{mask_type}'] = overlapFilter.GetDiceCoefficient()
        # VolSimilarity_dict[f'{mask_type}'] = overlapFilter.GetVolumeSimilarity()
        # Jaccard_dict[f'{mask_type}'] = overlapFilter.GetJaccardCoefficient()

    f = open(f"{dataset_path}/{I_deformed_filename.stem}/DSC.txt", "w+")
    print(f'--> DSC using the nonrigid transform in   {I_deformed_filename.stem}')
    for mask_type in arr__mask_type:
        # writre to console
        print(f'\t  DSC for {mask_type} = {DSC_dict[f"{mask_type}"]}')
        # print(f'\t  VolSimilarity for {mask_type} = {VolSimilarity_dict[f"{mask_type}"]}')
        # print(f'\t  Jaccard coeff for {mask_type} = {Jaccard_dict[f"{mask_type}"]}')
        print(f'----------------------------------------------------')

        # write to file
        f.write(f'DSC for {mask_type} = {DSC_dict[f"{mask_type}"]} \n')
        # f.write(f'VolSimilarity for {mask_type} = {VolSimilarity_dict[f"{mask_type}"]} \n')
        # print(f'\t  Jaccard coeff for {mask_type} = {Jaccard_dict[f"{mask_type}"]}')
        f.write(f'---------------------------------------------------- \n')
    f.close()


#%% Single registration (one set of params)
# I_deformed_filename = f'{I_m_filename.split("_")[0]} - rigidly aligned to femur only - I_m mask=wholeLeg - 250x12.nii'
# I_deformed_filename = f'{I_m_filename.split("_")[0]} - AFTER CLIPPING.nii'
#     subprocess.call(["python", "callElastix.py",                                        # using a subprocess for each iteration instead of normal function call to solve the "log file issue" (can't be recreated per process) -> see this issue  https://github.com/SuperElastix/SimpleElastix/issues/104
#                      dataset_path, I_f_filename, I_m_filename, I_f_mask_filename, I_m_mask_filename, str(use_I_m_mask), rigid_alignment_transform__filename, I_m_rigidityCoeffIm_filename, reg_type, str(n_itr_rigid),
#                      str(n_itr_nonRigid), str(n_res), str(use_rigidityPenalty), str(use_landmarks), I_f_landmarks_filename, I_m_landmarks_filename, I_deformed_filename])
#

#%% Transformix ###
# dataset_path = "S:/datasets/Sub_3/"             # 'S:/datasets/Sub_3/'        # "S:/datasets/Sub_7/"        # "C:/Users/bzfmuham/OneDrive/Knee-Kinematics/elastix 4.9.0/elastix-4.9.0-example_2/exampleinput/"
# im_to_deform___filename = "R4_mask_allBones.nii"             # "R3_t1-minus-t2_rigidlyAligned.nii"     "70_t1-minus-t2_rigidlyAligned.nii"
# # trnsfrmx_pMap = elstx_transform_pMap[0]
# pMap_filename = 'R4_to_R1__trans__rigid_alignment__femur.txt'
# output_filename = f'R4_mask_allBones___rigidly_aligned_to_R1_femur.nii'
#
# call_transformix.call_transformix(dataset_path, im_to_deform___filename, pMap_filename, output_filename)



#%% DICE only
# mask_I_f = sitk.ReadImage(f'{dataset_path}/{I_f}_mask_allBones.nii')
# mask_I_m_deformed = sitk.ReadImage(f'{dataset_path}/R4_mask_allBones___deformed_to___R1_mask_allBones___at_rigidAlignment=femur/'
#                                    f'R4_mask_allBones___deformed_to___R1_mask_allBones___at_rigidAlignment=femur.nii')
# overlapFilter = sitk.LabelOverlapMeasuresImageFilter()
# overlapFilter.Execute(mask_I_f, mask_I_m_deformed)
#
# print(f'DSC = {overlapFilter.GetDiceCoefficient()}')
# print(f'VolSim = {overlapFilter.GetVolumeSimilarity()}')



#%% # transformix & DSC
# overlapFilter = sitk.LabelOverlapMeasuresImageFilter()
# pMap_filename = f'{dataset_path}/R4_vol_deformed_to_R1___rigidAlignment=patella/TransformParameters.0.txt'
# DSC_dict = defaultdict(list);       VolSimilarity_dict = defaultdict(list);       Jaccard_dict = defaultdict(list)
# arr__mask_type = ('mask_allBones',)
# for mask_type in arr__mask_type:
#     im_to_deform___filename = f'{I_m}_{mask_type}.nii'
#     output_filename = f'{im_to_deform___filename.stem}__deformed__2.nii'
#     call_transformix.call_transformix(dataset_path, im_to_deform___filename, pMap_filename, output_filename)
#
#     # mask_I_f = sitk.ReadImage(f'{dataset_path}/{I_f}_{mask_type}.nii')
#     mask_I_f = sitk.ReadImage(f'{dataset_path}/{I_f}_{mask_type}.nii')
#     mask_I_m_deformed = sitk.ReadImage(f'{dataset_path}/{output_filename.stem}/{output_filename}')
#     overlapFilter.Execute(mask_I_f, mask_I_m_deformed)
#     DSC_dict[f'{mask_type}'] = overlapFilter.GetDiceCoefficient()
#     VolSimilarity_dict[f'{mask_type}'] = overlapFilter.GetVolumeSimilarity()
#     Jaccard_dict[f'{mask_type}'] = overlapFilter.GetJaccardCoefficient()
#
# # print(f'--> DSC using the nonrigid transform in   {I_deformed_filename.stem}')
# print(f'--> Label overlap meaures:')
# for mask_type in arr__mask_type:
#     print(f'\t  DSC for {mask_type} = {DSC_dict[f"{mask_type}"]}')
#     print(f'\t  VolSimilarity for {mask_type} = {VolSimilarity_dict[f"{mask_type}"]}')
#     print(f'\t  Jaccard coeff for {mask_type} = {Jaccard_dict[f"{mask_type}"]}')
#     print(f'----------------------------------------------------')







#%% ############# Im view (sitk -> calls ImageJ)  #############
#
# # Object oriented interface:
# image_viewer = sitk.ImageViewer()       # should be able to find ImageJ
# image_viewer.SetTitle('I_f')
# image_viewer.Execute(I_f)
# # image_viewer.SetTitle('I_m')
# # image_viewer.Execute(I_m)
# image_viewer.SetTitle('result im')
# image_viewer.Execute(resultImage)
# # image_viewer.SetTitle('deformation field')        # ImageJ can't view a 3d vector field
# # image_viewer.Execute(deformation_fld)


