import os
import subprocess
from pathlib import Path
import itertools
import glob

import nibabel as nib
import numpy as np

import SimpleITK as sitk  # see doc at     https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html

from collections import defaultdict
import nii_utils
from matplotlib import pyplot as plt

# import ctypes
# from win10toast import ToastNotifier

#######################################################################################

def correct_transform(dataset_path, outFldr, srcOfCorrectTransform, imToCorrect, correctedImNewName):
    orgAffine = nib.load(f"{dataset_path}/{srcOfCorrectTransform}").affine  # load the original transform to use it for the generated elastix result
    im = nib.load(f"{outFldr}/{imToCorrect}")
    im.set_sform(orgAffine);       im.set_qform(orgAffine)
    im.to_filename(f"{outFldr}/{correctedImNewName}")           # can't overwrite orginal image..
    os.remove(f"{outFldr}/{imToCorrect}")

def plot_convergence(elx_output_folder):
    ## plots the convergence for every resolution (total metric vs. iteration)

    itr_info__filenames = glob.glob(f'{elx_output_folder}/IterationInfo.0.R*.txt')     # a file per resolution
    n_res = len(itr_info__filenames)
    fig, ax_lst = plt.subplots(n_res, 1)
    fig.canvas.set_window_title(Path(f'{elx_output_folder}').stem)

    # for each resolution: open corrsesponding iterationInfo file, extract metric vector (vs itrs) & plot it in the corresponding axes in the subplot
    for itr_info__filename, curr_res in zip(itr_info__filenames, list(range(n_res))):
        with open(itr_info__filename, 'r') as itr_info__file:
            itr_lst = itr_info__file.readlines()
            metric_vec = list()
            for itr in itr_lst[1:]:             # skip first row as it's the header
                metric_vec.append(itr.split('\t')[1])      # 2nd entry in the row is the total metric for this iteration

            metric_vec = np.array(metric_vec, dtype=float) # VIP, otherwise plt will take AGES to plot !!
            ax_lst[curr_res].plot(metric_vec)
            ax_lst[curr_res].set_title(f'Resolution {curr_res}')
            ax_lst[curr_res].set_xlabel('iterations')
            ax_lst[curr_res].set_ylabel('total metric')
            # ax_lst[i].yticks(np.linspace(metric_vec.min(), metric_vec.max(), num=5))

def call_elastix(**kwargs):
    ##region --> important params
    similarity_metric = "AdvancedMattesMutualInformation"  # AdvancedNormalizedCorrelation - AdvancedMeanSquares (works well with masks) -  AdvancedMattesMutualInformation (better for vol) - NormalizedMutualInformation  -  "TransformRigidityPenalty"  - AdvancedKappaStatistic
    # meansSq__useNormalization = "false"  # def: false
    mutInf__histBins = "32"     # use '2' for registering binary masks, otherwise 32
    erodeMask = "false"  # [Par0017] If the mask serves as region of interest, set it to false. If the mask indicates which pixels are valid, then set it to true. If you do not use a mask, the option doesn't matter. .Better to use it for multi-res registration    (see   http://lists.bigr.nl/pipermail/elastix/2011-November/000627.html) & important if "RandomCoordinate" sampler is used with a fixed image mask in use as well
    rgdtyWt_vec = 3*('0.1',) + (kwargs['n_res']-3)*('4',)         # list multiplication (concatenation)
    #endregion


    print(f" ######################  elastix ->  {kwargs['I_deformed_filename']}  ######################")
    cwd = os.getcwd()
    os.chdir(kwargs['dataset_path'])
    data_format = Path(kwargs["I_f_filename"]).suffix.split(".")[1]
    # sitk.ProcessObject_GlobalDefaultDebugOn()       # to see detailed debug info (e.g. image readers... etc.)

    # VIP: ElastixImageFilter() will be used ONLY for preparation, not for actual registeration (i.e. i will not use elx.execute, but rather the elastix.exe of the separately built elastix version (updated to 5.0) )
    elx = sitk.ElastixImageFilter()

    output_dir = Path(kwargs["I_deformed_filename"]).stem
    os.makedirs(output_dir, exist_ok=True)
    I_m = sitk.ReadImage(f'{kwargs["I_m_filename"]}')       # just to infer relevant image info, not for actual registeration
    # if data_format == "nii" and I_m.GetMetaData('bitpix') in ("8", "16"):                                        # as sitk might read 8-bit unsigned as 16-bit signed
    if "mask" in str(kwargs['I_m_filename']):
        I_m = sitk.Cast(I_m, sitk.sitkUInt8)

    ### Setting pMap ###
    pMap = defaultdict(list)
    if kwargs['reg_type'] == 'RIGID':
        pMap = sitk.GetDefaultParameterMap('rigid')             # 'translation' - 'rigid' - 'affine' - 'bspline' (defaults)
        pMap['AutomaticScalesEstimation'] = ['true']
    elif kwargs['reg_type'] == 'NON_RIGID':
        pMap = sitk.ReadParameterFile('C:/Users/bzfmuham/OneDrive/Knee-Kinematics/Code_and_utils/'
                                      'Elastix_registration/Par0004.bs_base.NRP08.All.txt')
        # elx.SetInitialTransformParameterFileName(kwargs['rigid_alignment_transform__filename'])
        pMap["Transform"] = ["BSplineTransform"]
        pMap['BSplineTransformSplineOrder'] = ['3']

    # Input-related params
    pMap['ErodeMask'] = [erodeMask]
    pMap['DefaultPixelValue'] = [str(sitk.GetArrayFromImage(I_m).min())]  # sets pixel values outside the moving image grid (at interpolation) -> set it to <= the min in your dataset (i.e. bgd)

    ## Registration approach
    pMap['Registration'] = ['MultiMetricMultiResolutionRegistration']
    pMap['UseDirectionCosines'] = ['true']                 # def: true. Highly recommended, as it takes into account the "image-to-world" affine transform that comes from the scanner.

    ## Multi-resolution settings (b-spline grid & image pyramid)
    pMap['NumberOfResolutions'] = [str(kwargs["n_res"])]        # default: 4
    pMap['FinalGridSpacingInVoxels'] = [str(kwargs["spc"])]             # couldn't go lower than 2 for Sub_3 dataset. what is used in Staring 2007: 8.0; will be automatically approximated for each dimension during execution
    pMap['GridSpacingSchedule'] = [str(2 ** (kwargs["n_res"] - 1 - i)) for i in range(kwargs["n_res"])]  # generate using "list comprehension", default for 4 resolutions: ['2**3', '2**2', '2**1', '2**0']
    pMap['FixedImagePyramid'] = ['FixedSmoothingImagePyramid']  # no downsampling needed when a random sampler is used (what's used in Staring 2007 was 'RecursiveImagePyramid' but they use )
    pMap['MovingImagePyramid'] = ['MovingSmoothingImagePyramid']
    imPyrSchdl_lst = []
    for i in range(kwargs["n_res"]):
        for j in range(I_m.GetDimension()):  # just to repeat the factor for all dims per resolution
            imPyrSchdl_lst.append(str(2 ** (kwargs["n_res"] - 1 - i)))
    pMap['ImagePyramidSchedule'] = imPyrSchdl_lst  # MUST be specified for all resolutions & DIMS in order to work correctly, otherwise the default schedule is created !!
    pMap['WritePyramidImagesAfterEachResolution'] = ['false']

    ## Metric0: similarity
    pMap['Metric'] = (similarity_metric,)
    if similarity_metric == "AdvancedMeanSquares":
        pMap['UseNormalization'] = (meansSq__useNormalization,)     # only effective in case "similarity_metric = AdvancedMeansSquare"
    elif similarity_metric in ('AdvancedMattesMutualInformation', 'NormalizedMutualInformation') :
        pMap['NumberOfHistogramBins'] = (mutInf__histBins,)
    if similarity_metric == "AdvancedMeanSquares" and data_format == "nii" and I_m.GetMetaData('bitpix') == "32":
        print("!!!!!! BETTER TO USE MI FOR TEXTURE DATA !!!!!!!!")
    pMap['UseFastAndLowMemoryVersion'] = ['true']          # def: true. false can take up HUGE RAM !!
    pMap['Metric0Weight'] = ['1']

    ## Metric1 (nonrigid): rigidity penalty [Staring 2007]
    if kwargs['reg_type'] == 'NON_RIGID' and kwargs["use_rigidityPenalty"]:
        ## orthonormality-based rigidity [Staring 2007]
        pMap['Metric'] = pMap['Metric'] + ('TransformRigidityPenalty',)
        pMap['MovingRigidityImageName'] = [str(kwargs['I_rigidity_coeff__filename'])]
        # pMap['MovingRigidityImageName'] = ['mask_allBones_63.nii']
        pMap['Metric1Weight'] = rgdtyWt_vec
        pMap['LinearityConditionWeight'] = ['100.0']  # originaly in Staring 2007: 100.0
        pMap['OrthonormalityConditionWeight'] = ['1.0']  # originaly in Staring 2007: 1.0     (rigidity preservation - most dominant)
        pMap['PropernessConditionWeight'] = ['2.0']  # originaly in Staring 2007: 2.0          (volume preservation)
        pMap['DilateRigidityImages'] = ['false', 'false', 'true', 'true']       # def: true   (extend region of rigidity to force rigidity of the inner region)
        pMap['DilationRadiusMultiplier'] = ['2.0']  # originaly in Staring 2007: 2.0 (def: 1x)    (multiples of the grid spacing of the B-spline transform (so, differs with the resolution))

        ## distance-preserving rigidity penalty [Kim 2013]
        # pMap['Metric'] = pMap['Metric'] + ('DistancePreservingRigidityPenalty',)
        # pMap['Metric1Weight'] = rgdtyWt_vec
        # pMap['SegmentedImageName'] = [kwargs['I_rigidity_coeff__filename']]
        # # pMap['PenaltyGridSpacingInVoxels'] = ("4",)*I_m.GetDimension()                  # CONSTANT over all resolutions!! org: ("4","4","1") for 3d
        # pMap['PenaltyGridSpacingInVoxels'] = ("4", "4", "1")


    ## Metric2: Corresponding landmarks (MUST add it as the last metric)
    if kwargs["use_landmarks"]:
        # elx.SetFixedPointSetFileName(kwargs["I_f_landmarks_filename"])
        # elx.SetMovingPointSetFileName(kwargs["I_m_landmarks_filename"])
        pMap['Metric'] = pMap['Metric'] + ('CorrespondingPointsEuclideanDistanceMetric',)
        metric_CorrLndmrks_wt = 0.01
        if kwargs['reg_type'] == 'NON_RIGID' and kwargs["use_rigidityPenalty"]:
            pMap['Metric2Weight'] = [str(metric_CorrLndmrks_wt)]
        else:
            pMap['Metric1Weight'] = [str(metric_CorrLndmrks_wt)]

    ## for regularizing the resulting deformation at denser grids
    # pMap['Metric'] = pMap['Metric'] + ('TransformBendingEnergyPenalty',)

    ### Optimizer params
    pMap['Optimizer'] = ['AdaptiveStochasticGradientDescent']      # 16 optimizers avail.
    # pMap['ShowExactMetricValue'] = ['true']                         # only if you need it
    pMap['MaximumNumberOfIterations'] = [str(kwargs['n_itr'])]          # reasonable range: 250 -> 2000 (500 is a good compromise)
    pMap['AutomaticParameterEstimation'] = ['true']
    pMap['ASGDParameterEstimationMethod'] = ['Original']       # Original || DisplacementDistribution-> more efficient estimation of intial step size (Qiao et al., 2016 in elastix doc)
    pMap['UseAdaptiveStepSizes'] = ['true']
    maximumStepLengthMultiplier = 1  # def: 1 (but best result with (spc = 4.0, rigidityPenaltyWtAtFullRes = 4.0) is at 100 ( R4 to R1 registration (bone mask)))
    meanVoxSpc = np.array(I_m.GetSpacing()).mean()
    pMap['MaximumStepLength'] = [str(meanVoxSpc*maximumStepLengthMultiplier)]                       # Default: mean voxel spacing of I_f & I_m (1.25 for Sub_3). It's the maximum voxel displacement between two iterations. The larger this parameter, the more aggressive the optimization (if too large but the deformation is small, it can overshoot & result in bad registeration!!).
    # pMap['ValueTolerance'] = ['1']                           # seems ineffective for 'AdaptiveStochasticGradientDescent'
    # pMap['GradientMagnitudeTolerance'] = ['0.000944']        # seems ineffective for 'AdaptiveStochasticGradientDescent'

    ### I_f sampler params
    pMap['ImageSampler'] = ['RandomCoordinate']        # {Random, Full, Grid, RandomCoordinate}
    numSaptialSamples = 5000                            # def: 5000. Don't go lower than 2000
    pMap['NumberOfSpatialSamples'] = [str(numSaptialSamples*1)]
    pMap['NewSamplesEveryIteration'] = ['true']        # def: False. But highly recommended to use it specailly with random samplers.
    pMap['UseRandomSampleRegion'] = ['false']
    # pMap['SampleRegionSize'] = ['imSz_X_mm/3', 'imSz_Y_mm/3', 'imSz_Z_mm/3']         # if 'UseRandomSampleRegion' is used, this param is effective. VIP -> must specified in mm & for each dim (not each res !). Default: imSz_mm/3

    ### interpolators params
    intrpltr_BsplineOrdr = 5            # avail range: 0 -> 5
    pMap['FixedImageBSplineInterpolationOrder'] = [str(intrpltr_BsplineOrdr)]  # When using a RandomCoordinate sampler, the fixed image needs to be interpolated @ each iteration
    pMap['Interpolator'] = ['BSplineInterpolator']                 # use 'LinearInterpolator' for faster performance
    pMap['BSplineInterpolationOrder'] = [str(intrpltr_BsplineOrdr)]
    if "mask" in str(kwargs['I_m_filename']):
        pMap['FinalBSplineInterpolationOrder'] = ['0']             # A MUST when deforming a "binary" image (i.e. segmentation)
        pMap["ResultImagePixelType"] = ["unsigned char"]
    else:
        pMap['FinalBSplineInterpolationOrder'] = [str(intrpltr_BsplineOrdr)]
        pMap["ResultImagePixelType"] = ["float"]
    pMap['ResampleInterpolator'] = ['FinalBSplineInterpolator']           # To generate the final result, i.e. the deformed result of the registration

    ## Result-related params
    pMap['ResultImageFormat'] = (data_format,)
    pMap['WriteTransformParametersEachResolution'] = ['true']
    pMap['WriteResultImageAfterEachResolution'] = ['false']
    pMap['WriteResultImageAfterEachIteration'] = ['false']
    pMap['WriteTransformParametersEachIteration'] = ['false']


    elx.SetParameterMap(pMap)
    # elx.SetLogToConsole(True)  # to see output of each iteration
    # elx.SetLogToFile(True)
    elx.WriteParameterFile(elx.GetParameterMap()[0], f'{output_dir}/pMap0.txt')     # will be read by elastix.exe for the actual registration  ;)
    elx.PrintParameterMap()     # just for confirmation

    try:
        subprocess.run(["elastix.exe",
                        "-f", kwargs['I_f_filename'],
                        "-m", kwargs['I_m_filename'],
                        "-fMask", kwargs['I_f_mask_filename'],
                        # "-fMask", f"{kwargs['dataset_path']}/sag1__Hoffa_mask/tPoint_{Path(kwargs['I_f_filename']).stem.split('_')[1]}_mask_hoffa.{data_format}",
                        "-p", f"{output_dir}/pMap0.txt",
                        # "-t0", f"{kwargs['dataset_path']}/{kwargs['rigid_alignment_transform__filename']}",
                        # "-labels", "R1_mask_allBones_xtnd__dilated.nii",
                        "-out", output_dir])
        # elx.execute()
    except:
        return False


    ### clip the result to the range of I_m (to clean interpolation noise at edges)
    # resultImage = elx.GetResultImage()
    resultImage = sitk.ReadImage(f'{output_dir}/result.0.{data_format}')
    if data_format == "nii" and resultImage.GetMetaData('bitpix') in ("8", "16"):                                          # as sitk might read 8-bit unsigned as 16-bit signed
        resultImage = sitk.Cast(resultImage, sitk.sitkUInt8)
    os.remove(f"{output_dir}/result.0.{data_format}")            # as it will be replaced by the corrected result in the following
    resultImage_arr = sitk.GetArrayFromImage(resultImage)
    resultImage_arr = resultImage_arr.clip(min=sitk.GetArrayFromImage(I_m).min(), max=sitk.GetArrayFromImage(I_m).max())

    if data_format == "nii":           # write result using nibabel (not sitk.WriteImage() as it screws the header (sform_code & qform_code & quaternion))
        result_im__read_sitk_write_nib = nib.Nifti1Image(resultImage_arr.swapaxes(0, 2),        # swapping the local im coords (aka. lattice) due to diff bet nibabel & itk image axes
                                                    nib.load(kwargs['I_f_filename']).affine)                                    # use the image resulting from elx registration with the affine transfrom coming from the original data (e.g. fixed im)
        try:            # in case a result with the same filename exists
            result_im__read_sitk_write_nib.to_filename(f'{output_dir}/elx_rslt---{kwargs["I_deformed_filename"]}')
            # sitk.WriteImage(resultImage, f'{elx.kwargs[GetOutputDirectory()}/{"I_deformed_filename"]}')
        except:
            result_im__read_sitk_write_nib.to_filename(f'{output_dir}/__TEMP_NAME__{kwargs["n_res"]}x{kwargs["n_itr"]}.{dataFrmt}')
    else:               # e.g. if image is png --> # write directly though sitk
        sitk.WriteImage(sitk.GetImageFromArray(resultImage_arr), f'{output_dir}/elx_rslt---{kwargs["I_deformed_filename"]}')


    os.chdir(cwd)               # back to the working directory before entering this method
    return True

def non_rigid_reg(dataset_path, I_f, I_m):
    #region--> inputs & important params (CHECK REGULARILY)
    ## TEST params
    # arr__n_res = (2,)                                                                     # def: (4, 1, 3, 2)
    # arr__n_itr = (2,)                                                                     # from 1 to 5x
    # arr__spc = (64,)                                                                 # finalGridSpacingInVoxels - def: 4
    ## Actual params
    arr__n_res = (4,)                                       # def: (4, 1, 3, 2)
    arr__n_itr = (2000,)                                     # from 1 to 5x
    arr__spc = (8,)                                         # np.linspace(4, 0.5, num=8)

    struct_to_register = 'vol'  # tex || mask_allBones || mask_hoffa
    crop_state = ''  # '' || 'uncropd' || 'xtnd' || 'cropd'
    data_format = 'nii'
    similarity_metric_abbrvn_in_elx_output = 'MI'       # this abbreviation MUST be like the variable "similarity_metric" in call_elastix()
    use_rigidityPenalty = True
    other_info_to_add_to_elx_output_folder_name = '__withRgdtyPnlty_and_fMask'            # '__bendEnrgPen__fMask_hoffa__noFixedMask

    # i_grid = "gridIm_0.01"
    I_rigidity_coeff__filename = Path(f"{I_m}__mask_allBones__{crop_state}.{data_format}")        # N.B. rigidity image:  orthonormailty-based -> I_m  -  distance-preserving-based -> I_f || BUG: doesn,'t work when the name begins with a number !!                                                                                            # {I_m}_mask_allBones__{crop_state}.{data_format}  -  {I_f}_mask_allBones__{crop_state}.{data_format}
    arr__use_landmarks = (False,)
    #endregion



    #region--> Constructing file names
    I_f_filename = Path(f"{I_f}__{struct_to_register}__{crop_state}.{data_format}")                                                     # flexed,   "R1_t1-minus-t2.{data_format}"                  '72_t1-minus-t2'
    I_m_filename = Path(f"{I_m}__{struct_to_register}__{crop_state}.{data_format}")                                      # extended, "R3_t1-minus-t2_rigidlyAligned.{data_format}"   '70_t1-minus-t2_rigidlyAligned'
    I_f_mask_filename = Path(f"{I_f}__mask_wholeLeg__{crop_state}.{data_format}")                                        # "R1_wholeLegMask.labels.{data_format}"    '72_wholeLegMask.labels.{data_format}' || R1-femur.{data_format}
    use_I_m_mask = False
    I_m_mask_filename = Path(f'{I_m}______.{data_format}')                                                                # mask_tibia  ||  R4_Patella.{data_format}
    reg_type = "NON_RIGID"                                                              # 'RIGID' - 'NON_RIGID'
    arr__rigid_alignment_transform__filename = (Path(f'{I_m}_to_{I_f}__trans__rigid_alignment__femur.txt'),)
    n_lndmrks = 5
    I_f_landmarks_filename = Path(f"{I_f}_landmarks_femur_{n_lndmrks}.pts")                                                               # "Choosing_bone_markers/R1_landmarks.pts"
    I_m_landmarks_filename = Path(f"{I_m}_landmarks_femur___transformed_from_{I_f}_landmarks_femur_{n_lndmrks}.pts")                                                               # "Choosing_bone_markers/R3_landmarks.pts"
    # endregion

    #region--> Check if the "image-to-world" affine tranforms are the same for all involved volumes & masks
    # if data_format == 'nii' :
    #     I_f_affine = nib.load(f'{dataset_path}/{I_f_filename}').affine
    #     I_m_affine = nib.load(f'{dataset_path}/{I_m_filename}').affine
    #     I_f_mask_affine = nib.load(f'{dataset_path}/{I_f_mask_filename}').affine
    #     I_m_rigidityCoeffIm_affine = nib.load(f'{dataset_path}/{I_rigidity_coeff__filename}').affine
    #     if not ( np.array_equiv(I_f_affine, I_m_affine) and
    #              np.array_equiv(I_f_affine, I_f_mask_affine) and
    #              np.array_equiv(I_f_affine, I_m_rigidityCoeffIm_affine)
    #            ):
    #         print('!! ERROR: "image-to-world" affine transforms are not the same for all involved volumes & masks!')
    #         exit(-1)
    #endregion

    #region--> MULTI registration
    for rigid_alignment_transform__filename,      spc,      n_itr,      n_res,      use_landmarks     in itertools.product(
        arr__rigid_alignment_transform__filename, arr__spc, arr__n_itr, arr__n_res, arr__use_landmarks):
        rmZero = lambda i: i if i % 1 else int(i)
        I_deformed_filename = Path(f''
                                   # f'{I_m}_defTo_{I_f}_{struct_to_register}__{crop_state}___at_'
                                   # f'rgAln-{rigid_alignment_transform__filename.stem.split("__")[3]}_'
                                   # f'{I_m.split("_")[1]}_to_{I_f.split("_")[1]}'
                                   f'{I_m.split("_")[-1]}_to_{I_f.split("_")[-1]}'
                                   f'__{struct_to_register}'
                                   f'__{n_res}x{rmZero(n_itr/1000)}k'
                                   f'__sim-{similarity_metric_abbrvn_in_elx_output}'
                                   # f'__rigidityIm-moving'
                                   f'__Spc-{spc}'
                                   # f'__RigPenSpc-4-4-1'
                                   # f'__RgdtyWt-100
                                   f'{other_info_to_add_to_elx_output_folder_name}'
                                   f'.{data_format}')
        elx_output_folder = f"{dataset_path}/{I_deformed_filename.stem}"
        reg_done = call_elastix(dataset_path=dataset_path,
                                I_f_filename=I_f_filename,
                                I_m_filename=I_m_filename,
                                I_f_mask_filename=I_f_mask_filename,
                                I_m_mask_filename=I_m_mask_filename,
                                use_I_m_mask=use_I_m_mask,
                                rigid_alignment_transform__filename=rigid_alignment_transform__filename,
                                I_rigidity_coeff__filename=I_rigidity_coeff__filename,
                                reg_type=reg_type,
                                n_itr=n_itr,
                                n_res=n_res,
                                spc=spc,
                                use_rigidityPenalty=use_rigidityPenalty,
                                use_landmarks=use_landmarks,
                                I_f_landmarks_filename=I_f_landmarks_filename,
                                I_m_landmarks_filename=I_m_landmarks_filename,
                                I_deformed_filename=I_deformed_filename,
                                )
        if not reg_done:
            print("ERROR DURING ELASTIX EXECUTION !! check log file.. skipping to next execution...")
            f = open(f"{elx_output_folder}/ERROR.txt", "w+")              # just to indicate that this registration has already exited with error, not still running
            f.close()
            continue

        # plot_convergence(elx_output_folder)

        # ctypes.windll.user32.MessageBoxW(0, f"{I_deformed_filename} has finished", f"{I_deformed_filename} has finished", style="0")



        ### run transformix to generate deformed rectangular grid, det(spatial Jacobian) & deformation field
        subprocess.run(["transformix.exe",
                        # "-in", f"{dataset_path}/gridIm.{data_format}",
                        "-tp", f"{elx_output_folder}/TransformParameters.0.txt",
                        "-def", "all",
                        "-jac", "all",
                        "-out", elx_output_folder])
        for im_to_correct_info in ((f"spatialJacobian.{data_format}", f"J---{I_deformed_filename}"),
                                    # (f"result.{data_format}", f"grid_deformed---{I_deformed_filename}")
                                  ):
            if data_format == "nii":
                nii_utils.correct_transform(dataset_path, elx_output_folder, I_f_filename, *im_to_correct_info)
            else:             # just renaming
                os.replace(f"{elx_output_folder}/{im_to_correct_info[0]}", f"{elx_output_folder}/{im_to_correct_info[1]}")

        print(f'---- Registeration of  \"{I_deformed_filename.stem}\"  finished successfully. ----')


    # ## deform "I_m-related masks" & calc DSC for each
    # overlapFilter = sitk.LabelOverlapMeasuresImageFilter()
    # DSC_dict = defaultdict(list)
    # # VolSimilarity_dict = defaultdict(list)
    # # Jaccard_dict = defaultdict(list)
    # pMap_filename = Path(f'{I_deformed_filename.stem}/TransformParameters.0.txt')
    # arr__mask_type = ('mask_wholeLeg', 'mask_allBones')
    # for mask_type in arr__mask_type:
    #     im_to_deform___filename = Path(f'{I_m}_{mask_type}.{data_format}')
    #     output_filename = Path(f'{im_to_deform___filename.stem}__deformed.{data_format}')
    #     call_transformix.call_transformix(dataset_path = dataset_path,
    #                                       im_to_deform__filename = im_to_deform___filename,
    #                                       pMap_path = pMap_filename,
    #                                       output_filename = output_filename,
    #                                       )
    #
    #     mask_I_f = sitk.ReadImage(f'{dataset_path}/{I_f}_{mask_type}.{data_format}')
    #     mask_I_m_deformed = sitk.ReadImage(f'{dataset_path}/{I_deformed_filename.stem}/{output_filename.stem}/{output_filename}')
    #     overlapFilter.Execute(mask_I_f, mask_I_m_deformed)
    #     DSC_dict[f'{mask_type}'] = overlapFilter.GetDiceCoefficient()
    #     # VolSimilarity_dict[f'{mask_type}'] = overlapFilter.GetVolumeSimilarity()
    #     # Jaccard_dict[f'{mask_type}'] = overlapFilter.GetJaccardCoefficient()
    #
    # f = open(f"{dataset_path}/{I_deformed_filename.stem}/DSC.txt", "w+")
    # print(f'--> DSC using the nonrigid transform in   {I_deformed_filename.stem}')
    # for mask_type in arr__mask_type:
    #     # write to console
    #     print(f'\t  DSC for {mask_type} = {DSC_dict[f"{mask_type}"]}')
    #     # print(f'\t  VolSimilarity for {mask_type} = {VolSimilarity_dict[f"{mask_type}"]}')
    #     # print(f'\t  Jaccard coeff for {mask_type} = {Jaccard_dict[f"{mask_type}"]}')
    #     print(f'----------------------------------------------------')
    #
    #     # write to file
    #     f.write(f'DSC for {mask_type} = {DSC_dict[f"{mask_type}"]} \n')
    #     # f.write(f'VolSimilarity for {mask_type} = {VolSimilarity_dict[f"{mask_type}"]} \n')
    #     # print(f'\t  Jaccard coeff for {mask_type} = {Jaccard_dict[f"{mask_type}"]}')
    #     f.write(f'---------------------------------------------------- \n')
    # f.close()
#endregion

# Single registration (one set of params)
# I_deformed_filename = f'{I_m_filename.split("_")[0]} - rigidly aligned to femur only - I_m mask=wholeLeg - 250x12.{dataFrmt}'
# I_deformed_filename = f'{I_m_filename.split("_")[0]} - AFTER CLIPPING.{dataFrmt}'
#     subprocess.call(["python", "call_elastix.py",                                        # using a subprocess for each iteration instead of normal function call to solve the "log file issue" (can't be recreated per process) -> see this issue  https://github.com/SuperElastix/SimpleElastix/issues/104
#                      dataset_path, I_f_filename, I_m_filename, I_f_mask_filename, I_m_mask_filename, str(use_I_m_mask), rigid_alignment_transform__filename, I_rigidity_coeff__filename, reg_type, str(n_itr_rigid),
#                      str(n_itr), str(n_res), str(use_rigidityPenalty), str(use_landmarks), I_f_landmarks_filename, I_m_landmarks_filename, I_deformed_filename])

# DICE only
# mask_I_f = sitk.ReadImage(f'{dataset_path}/{I_f}_mask_allBones.{dataFrmt}')
# mask_I_m_deformed = sitk.ReadImage(f'{dataset_path}/R4_mask_allBones___deformed_to___R1_mask_allBones___at_rigidAlignment=femur/'
#                                    f'R4_mask_allBones___deformed_to___R1_mask_allBones___at_rigidAlignment=femur.{dataFrmt}')
# overlapFilter = sitk.LabelOverlapMeasuresImageFilter()
# overlapFilter.Execute(mask_I_f, mask_I_m_deformed)
#
# print(f'DSC = {overlapFilter.GetDiceCoefficient()}')
# print(f'VolSim = {overlapFilter.GetVolumeSimilarity()}')



## transformix & DSC
# overlapFilter = sitk.LabelOverlapMeasuresImageFilter()
# pMap_filename = f'{dataset_path}/R4_vol_deformed_to_R1___rigidAlignment=patella/TransformParameters.0.txt'
# DSC_dict = defaultdict(list);       VolSimilarity_dict = defaultdict(list);       Jaccard_dict = defaultdict(list)
# arr__mask_type = ('mask_allBones',)
# for mask_type in arr__mask_type:
#     im_to_deform___filename = f'{I_m}_{mask_type}.{dataFrmt}'
#     output_filename = f'{im_to_deform___filename.stem}__deformed__2.{dataFrmt}'
#     call_transformix.call_transformix(dataset_path, im_to_deform___filename, pMap_filename, output_filename)
#
#     # mask_I_f = sitk.ReadImage(f'{dataset_path}/{I_f}_{mask_type}.{dataFrmt}')
#     mask_I_f = sitk.ReadImage(f'{dataset_path}/{I_f}_{mask_type}.{dataFrmt}')
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


#region--> Im view (sitk -> calls ImageJ)  #############
#Object oriented interface:
# image_viewer = sitk.ImageViewer()       # should be able to find ImageJ
# image_viewer.SetTitle('I_f')
# image_viewer.Execute(I_f)
# # image_viewer.SetTitle('I_m')
# # image_viewer.Execute(I_m)
# image_viewer.SetTitle('result im')
# image_viewer.Execute(resultImage)
# # image_viewer.SetTitle('deformation field')        # ImageJ can't view a 3d vector field
# # image_viewer.Execute(deformation_fld)
#endregion

#region --> calcualte MSD & MAD between I_f & deformed I_m (WARNING: SIMILARITY METRICS NOT GOOD FOR EVALUATING REGISTERATION RESULTS for "texture" images, specailly mutli-modal)
# im1 = sitk.ReadImage(r"S:\datasets\s3_2\2d_highRes\slc_0_flexed_tex.tif")
# im2 = sitk.ReadImage(r"S:\datasets\s3_2\2d_highRes\4x10000__I_m-slc_13__sim-meansSq__Spc-2\elx_rslt---4x10000__I_m-slc_13__sim-meansSq__Spc-2.tif")
# # diff_im = sitk.SquaredDifference(im1, im2)
# diff_im = sitk.AbsoluteValueDifference(im1, im2)
#
# # im_statistics_fltr = sitk.StatisticsImageFilter()
# # im_statistics_fltr.Execute(diff_im)
# # MAD = im_statistics_fltr.GetMean()
# MAD = sitk.GetArrayFromImage(diff_im).mean()
#endregion

##############################################################

if __name__ == '__main__' :
    dataset_path = Path("S:/datasets/s3_1/")  # Sub_3 - s3_2/2d_highRes - Sub_7 -
    non_rigid_reg(dataset_path, 'R1', 'R4')
    # plot_convergence(r'S:\datasets\s1_3\weight_to_noWeight__tex__4x0.75k__sim-NMI__Spc-3.5__useRigPen__cropd')