import os
import subprocess
import sys
from pathlib import Path

import SimpleITK as sitk  # see doc at     https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html
import nibabel as nib

from collections import defaultdict

def callElastix(**kwargs):
    #region----> Important params
    similarity_metric = "AdvancedNormalizedCorrelation"                           # AdvancedNormalizedCorrelation - AdvancedMeanSquares (works well with masks) -  AdvancedMattesMutualInformation (better for vol) - NormalizedMutualInformation  -  "TransformRigidityPenalty"  - AdvancedKappaStatistic
    meansSq__useNormalization = "false"                      # def: false
    erodeMask = "false"         # [Par0017] If you use a mask, this option is important. If the mask serves as region of interest, set it to false. If the mask indicates which pixels are valid, then set it to true. If you do not use a mask, the option doesn't matter. .Better to use it for multi-res registration    (see   http://lists.bigr.nl/pipermail/elastix/2011-November/000627.html) & important if "RandomCoordinate" sampler is used with a fixed image mask in use as well
    # rgdtyWt_vec = 4*('0.02',) + (int(kwargs['n_res'])-4)*('0',)         # list multiplication (concatenation)
    #endregion

    #region--> Registration
    print(f" ######################  elastix ->  {kwargs['I_deformed_filename']}  ######################")
    cwd = os.getcwd()
    os.chdir(kwargs['dataset_path'])
    data_format = Path(kwargs["I_f_filename"]).suffix.split(".")[1]
    # sitk.ProcessObject_GlobalDefaultDebugOn()       # to see detailed debug info (e.g. image readers... etc.)

    elx = sitk.ElastixImageFilter()

    # elx.SetOutputDirectory(kwargs["I_deformed_filename"].split('.')[0])               # write all registration info there
    output_dir = Path(kwargs["I_deformed_filename"]).stem
    os.makedirs(output_dir, exist_ok=True)
    I_m = sitk.ReadImage(f'{kwargs["I_m_filename"]}')
    if data_format == "nii" and I_m.GetMetaData('bitpix') in ("8", "16"):                                        # as sitk might read 8-bit unsigned as 16-bit signed
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
    pMap['NumberOfResolutions'] = [kwargs["n_res"]]        # default: 4
    pMap['FinalGridSpacingInVoxels'] = [kwargs["spc"]]             # couldn't go lower than 2 for Sub_3 dataset. what is used in Staring 2007: 8.0; will be automatically approximated for each dimension during execution
    pMap['GridSpacingSchedule'] = [str(2 ** (int(kwargs["n_res"]) - 1 - i)) for i in range(int(kwargs["n_res"]))]  # generate using "list comprehension", default for 4 resolutions: ['2**3', '2**2', '2**1', '2**0']
    pMap['FixedImagePyramid'] = ['FixedSmoothingImagePyramid']  # no downsampling needed when a random sampler is used (what's used in Staring 2007 was 'RecursiveImagePyramid' but they use )
    pMap['MovingImagePyramid'] = ['MovingSmoothingImagePyramid']
    imPyrSchdl_lst = []
    for i in range(int(kwargs["n_res"])):
        for j in range(I_m.GetDimension()):  # just to repeat the factor for all dims per resolution
            imPyrSchdl_lst.append(str(2 ** (int(kwargs["n_res"]) - 1 - i)))
    pMap['ImagePyramidSchedule'] = imPyrSchdl_lst  # MUST be specified for all resolutions & DIMS in order to work correctly, otherwise the default schedule is created !!
    pMap['WritePyramidImagesAfterEachResolution'] = ['false']

    ## Metric0: similarity
    pMap['Metric'] = (similarity_metric,)
    pMap['UseNormalization'] = (meansSq__useNormalization,)
    if similarity_metric == "AdvancedMeanSquares" and data_format == "nii" and I_m.GetMetaData('bitpix') == "32":
            print("!!!!!! BETTER TO USE MI FOR TEXTURE DATA !!!!!!!!")
    pMap['NumberOfHistogramBins'] = ['32']
    pMap['UseFastAndLowMemoryVersion'] = ['true']          # def: true. false can take up HUGE RAM !!
    pMap['Metric0Weight'] = ['1']

    ## Metric1 (nonrigid): rigidity penalty [Staring 2007]
    if kwargs['reg_type'] == 'NON_RIGID' and eval(kwargs["use_rigidityPenalty"]):
        pMap['Metric'] = pMap['Metric'] + ('DistancePreservingRigidityPenalty',)
        pMap['Metric1Weight'] = rgdtyWt_vec
        pMap['SegmentedImageName'] = [kwargs['I_rigidity_coeff__filename']]
        # pMap['PenaltyGridSpacingInVoxels'] = ("4",)*I_m.GetDimension()                  # CONSTANT over all resolutions!! org: ("4","4","1") for 3d
        pMap['PenaltyGridSpacingInVoxels'] = ("4", "4", "1")

        # pMap['Metric'] = pMap['Metric'] + ('TransformRigidityPenalty',)
        # pMap['MovingRigidityImageName'] = [kwargs['I_rigidity_coeff__filename']]
        # pMap['Metric1Weight'] = rgdtyWt_vec
        # pMap['LinearityConditionWeight'] = ['100.0']  # originaly in Staring 2007: 100.0
        # pMap['OrthonormalityConditionWeight'] = ['1.0']  # originaly in Staring 2007: 1.0     (rigidity preservation - most dominant)
        # pMap['PropernessConditionWeight'] = ['2.0']  # originaly in Staring 2007: 2.0          (volume preservation)
        # pMap['DilateRigidityImages'] = ['false', 'false', 'true', 'true']       # def: true   (extend region of rigidity to force rigidity of the inner region)
        # pMap['DilationRadiusMultiplier'] = ['2.0']  # originaly in Staring 2007: 2.0 (def: 1x)    (multiples of the grid spacing of the B-spline transform (so, differs with the resolution))


    ## Metric2: Corresponding landmarks (MUST add it as the last metric)
    if eval(kwargs["use_landmarks"]):
        # elx.SetFixedPointSetFileName(kwargs["I_f_landmarks_filename"])
        # elx.SetMovingPointSetFileName(kwargs["I_m_landmarks_filename"])
        pMap['Metric'] = pMap['Metric'] + ('CorrespondingPointsEuclideanDistanceMetric',)
        metric_CorrLndmrks_wt = 0.01
        if kwargs['reg_type'] == 'NON_RIGID' and eval(kwargs["use_rigidityPenalty"]):
            pMap['Metric2Weight'] = [str(metric_CorrLndmrks_wt)]
        else:
            pMap['Metric1Weight'] = [str(metric_CorrLndmrks_wt)]

    ### Optimizer params
    pMap['Optimizer'] = ['AdaptiveStochasticGradientDescent']      # 16 optimizers avail.
    pMap['MaximumNumberOfIterations'] = [kwargs['n_itr']]          # reasonable range: 250 -> 2000 (500 is a good compromise)
    pMap['AutomaticParameterEstimation'] = ['true']
    pMap['ASGDParameterEstimationMethod'] = ['Original']       # Original || DisplacementDistribution-> more efficient estimation of intial step size (Qiao et al., 2016 in elastix doc)
    pMap['UseAdaptiveStepSizes'] = ['true']
    maximumStepLengthMultiplier = 1  # def: 1 (but best result with (spc = 4.0, rigidityPenaltyWtAtFullRes = 4.0) is at 100 ( R4 to R1 registration (bone mask)))
    meanVoxSpc = 1.25               # [TO DO] better to read directly from the image data !!
    pMap['MaximumStepLength'] = [str(meanVoxSpc*maximumStepLengthMultiplier)]                       # Default: mean voxel spacing of I_f & I_m (1.25 for Sub_3). It's the maximum voxel displacement between two iterations. The larger this parameter, the more aggressive the optimization.
    # pMap['ValueTolerance'] = ['1']                           # seems ineffective for 'AdaptiveStochasticGradientDescent'
    # pMap['GradientMagnitudeTolerance'] = ['0.000944']        # seems ineffective for 'AdaptiveStochasticGradientDescent'

    ### I_f sampler params
    pMap['ImageSampler'] = ['RandomCoordinate']        # {Random, Full, Grid, RandomCoordinate}
    numSaptialSamples = 5000
    pMap['NumberOfSpatialSamples'] = [str(numSaptialSamples*1)]          # def: 5000. Don't go lower than 2000
    pMap['NewSamplesEveryIteration'] = ['true']        # def: False. But highly recommended to use it specailly with random samplers.
    pMap['UseRandomSampleRegion'] = ['false']
    # pMap['SampleRegionSize'] = ['imSz_X_mm/3', 'imSz_Y_mm/3', 'imSz_Z_mm/3']         # if 'UseRandomSampleRegion' is used, this param is effective. VIP -> must specified in mm & for each dim (not each res !). Default: imSz_mm/3

    ### interpolators params
    intrpltr_BsplineOrdr = 5            # avail range: 0 -> 5
    pMap['FixedImageBSplineInterpolationOrder'] = [str(intrpltr_BsplineOrdr)]  # When using a RandomCoordinate sampler, the fixed image needs to be interpolated @ each iteration
    pMap['Interpolator'] = ['BSplineInterpolator']                 # use 'LinearInterpolator' for faster performance
    pMap['BSplineInterpolationOrder'] = [str(intrpltr_BsplineOrdr)]
    if "mask" in kwargs['I_f_filename']:
        pMap['FinalBSplineInterpolationOrder'] = ['0']             # A MUST when deforming a "binary" image (i.e. segmentation)
    else:
        pMap['FinalBSplineInterpolationOrder'] = [str(intrpltr_BsplineOrdr)]
        pMap["ResultImagePixelType"] = ["float"]
    pMap['ResampleInterpolator'] = ['FinalBSplineInterpolator']           # To generate the final result, i.e. the deformed result of the registration

    ## Result-related params
    pMap['ResultImageFormat'] = (data_format,)
    pMap['WriteTransformParametersEachResolution'] = ['true']
    pMap['WriteResultImageAfterEachResolution'] = ['true']
    pMap['WriteResultImageAfterEachIteration'] = ['false']
    pMap['WriteTransformParametersEachIteration'] = ['false']


    elx.SetParameterMap(pMap)
    # elx.SetLogToConsole(True)  # to see output of each iteration
    # elx.SetLogToFile(True)
    elx.WriteParameterFile(elx.GetParameterMap()[0], f'{output_dir}/pMap0.txt')
    elx.PrintParameterMap()  # will be read by elastix.exe for the actual registration  ;)

    try:
        subprocess.run(["elastix.exe",
                        "-f", kwargs['I_f_filename'],
                        "-m", kwargs['I_m_filename'],
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
    #endregion

##################################################################################################################################

if __name__ == '__main__':
    if callElastix(dataset_path = sys.argv[1],
                   I_f_filename = sys.argv[2],
                   I_m_filename = sys.argv[3],
                   I_f_mask_filename = sys.argv[4],
                   I_m_mask_filename = sys.argv[5],
                   use_I_m_mask = sys.argv[6],
                   rigid_alignment_transform__filename = sys.argv[7],
                   I_rigidity_coeff__filename = sys.argv[8],
                   reg_type = sys.argv[9],
                   n_itr = sys.argv[10],
                   n_res = sys.argv[11],
                   use_rigidityPenalty = sys.argv[12],
                   use_landmarks = sys.argv[13],
                   I_f_landmarks_filename = sys.argv[14],
                   I_m_landmarks_filename = sys.argv[15],
                   I_deformed_filename = sys.argv[16],
                   spc = sys.argv[17]):
        exit(0)
    else:
        exit(-1)