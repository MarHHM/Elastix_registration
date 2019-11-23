import os
import sys
from pathlib import Path

import SimpleITK as sitk  # see doc at     https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html
import nibabel as nib

from collections import defaultdict

def callElastix(**kwargs):
    print(f" ######################  elastix ->  {kwargs['I_deformed_filename']}  ######################")

    ## important params
    finalGridSpacingInVoxels = 4.0                  # def: 4.0
    maximumStepLengthMultiplier = 100.0             # def: 1 (but best result with (finalGridSpacingInVoxels = 4.0, rigidityPenaltyWtAtFullRes = 4.0) is at 100 ( R4 to R1 registration (bone mask)))

    ## TESTING - currently inactive!!
    # rigidityPenaltyWtAtFullRes = 1000.0                # def: 4.0 (used in Staring 2007 (alpha in eq. 1) -> 4.0 & 8.0 )
    test = ('2', '0.1', '0.1', '0.1')

    ####
    cwd = os.getcwd()
    os.chdir(kwargs['dataset_path'])
    # sitk.ProcessObject_GlobalDefaultDebugOn()       # to see detailed debug info (e.g. image readers... etc.)

    Elastix = sitk.ElastixImageFilter()
    Elastix.SetOutputDirectory(kwargs["I_deformed_filename"].split('.')[0])               # write all registration info there
    os.makedirs(Elastix.GetOutputDirectory(), exist_ok=True)
    I_f = sitk.ReadImage(kwargs['I_f_filename'])
    I_m = sitk.ReadImage(kwargs['I_m_filename'])
    Elastix.SetFixedImage(I_f)
    Elastix.SetMovingImage(I_m)
    Elastix.SetFixedMask(sitk.ReadImage(kwargs['I_f_mask_filename']))       # MUST be specified to avoid taking into account bgd & out-of-view noise during registration -  in the documentation, they say i usually only need to provide fixed image mask (moving image mask needed only when noisy features are close to the edge of the ROI (see 5.5 in manual))
    if eval(kwargs['use_I_m_mask']):
        Elastix.SetMovingMask(sitk.ReadImage(kwargs['I_m_mask_filename']))


    ### Setting pMap ###
    pMap = defaultdict(list)
    if kwargs['reg_type'] == 'RIGID':
        pMap = sitk.GetDefaultParameterMap('rigid')             # 'translation' - 'rigid' - 'affine' - 'bspline' (defaults)
        pMap['AutomaticScalesEstimation'] = ['true']
    elif kwargs['reg_type'] == 'NON_RIGID':
        pMap = sitk.ReadParameterFile('C:/Users/bzfmuham/OneDrive/Knee-Kinematics/Code_and_utils/'
                                      'Elastix_registration/Par0004.bs_base.NRP08.All.txt')
        Elastix.SetInitialTransformParameterFileName(kwargs['rigid_alignment_transform__filename'])
        pMap['BSplineTransformSplineOrder'] = ['3']

    # Input-related params
    pMap['FixedImageDimension'] = [str(I_f.GetDimension())]
    pMap['MovingImageDimension'] = [str(I_m.GetDimension())]
    pMap['DefaultPixelValue'] = [str(sitk.GetArrayFromImage(Elastix.GetMovingImage(0)).min())]  # sets pixel values outside the moving image grid (at interpolation) -> set it to <= the min in your dataset (i.e. bgd)

    ## Registration approach
    pMap['Registration'] = ['MultiMetricMultiResolutionRegistration']
    pMap['UseDirectionCosines'] = ['true']                 # def: true. Highly recommended, as it takes into account the "image-to-world" affine transform that comes from the scanner.

    ## Multi-resolution settings (b-spline grid & image pyramid)
    pMap['NumberOfResolutions'] = [kwargs["n_res"]]        # default: 4
    pMap['FinalGridSpacingInVoxels'] = [str(finalGridSpacingInVoxels)]             # couldn't go lower than 2 for Sub_3 dataset. what is used in Staring 2007: 8.0; will be automatically approximated for each dimension during execution
    pMap['GridSpacingSchedule'] = [str(2 ** (int(kwargs["n_res"]) - 1 - i)) for i in range(int(kwargs["n_res"]))]  # generate using "list comprehension", default for 4 resolutions: ['2**3', '2**2', '2**1', '2**0']
    pMap['FixedImagePyramid'] = ['FixedSmoothingImagePyramid']  # no downsampling needed when a random sampler is used (what's used in Staring 2007 was 'RecursiveImagePyramid' but they use )
    pMap['MovingImagePyramid'] = ['MovingSmoothingImagePyramid']
    imPyrSchdl_lst = []
    for i in range(int(kwargs["n_res"])):
        for j in range(I_m.GetDimension()):  # just to repeat the factor for all dims per resolution
            imPyrSchdl_lst.append(str(2 ** (int(kwargs["n_res"]) - 1 - i)))
    pMap['ImagePyramidSchedule'] = imPyrSchdl_lst  # MUST be specified for all resolutions & DIMS in order to work correctly, otherwise the default schedule is created !!
    pMap['ErodeMask'] = ['true']  # better to use it for multi-res registration    (see   http://lists.bigr.nl/pipermail/elastix/2011-November/000627.html)
    pMap['WritePyramidImagesAfterEachResolution'] = ['true']

    ## Metric0: similarity
    pMap['Metric'] = ['AdvancedMattesMutualInformation']
    pMap['NumberOfHistogramBins'] = ['32']
    pMap['UseFastAndLowMemoryVersion'] = ['false']          # def: true. false can take up HUGE RAM !!
    # pMap['Metric'] = ['AdvancedKappaStatistic']              # good as a similariy cost for registering binary images (i.e. masks)
    pMap['Metric0Weight'] = ['1']

    ## Metric1 (nonrigid): rigidity penalty [Staring 2007]
    if kwargs['reg_type'] == 'NON_RIGID' and eval(kwargs["use_rigidityPenalty"]):
        pMap['Metric'] = pMap['Metric'] + ('TransformRigidityPenalty',)
        pMap['MovingRigidityImageName'] = [kwargs['I_m_rigidityCoeffIm_filename']]
        ### TEST
        rigidityPenaltyWts = ['0.1'] * int(kwargs["n_res"])                   # def: 0.1 for all resolutions except last one (higher wt: 4) (e.g. ('0.1', '0.1','0.1', '4') for 4 res)
        # rigidityPenaltyWts[-1] = str(rigidityPenaltyWtAtFullRes)
        # pMap['Metric1Weight'] = tuple(rigidityPenaltyWts)                     # only last res gets the high weight
        # pMap['Metric1Weight'] = [str(rigidityPenaltyWtAtFullRes)]               # all resolutions get the same weight !!
        pMap['Metric1Weight'] = test

        pMap['LinearityConditionWeight'] = ['100.0']  # originaly in Staring 2007: 100.0
        pMap['OrthonormalityConditionWeight'] = ['1.0']  # originaly in Staring 2007: 1.0     (rigidity preservation - most dominant)
        pMap['PropernessConditionWeight'] = ['2.0']  # originaly in Staring 2007: 2.0          (volume preservation)
        pMap['DilateRigidityImages'] = ['false', 'false', 'true', 'true']       # def: true   (extend region of rigidity to force rigidity of the inner region)
        pMap['DilationRadiusMultiplier'] = ['2.0']  # originaly in Staring 2007: 2.0 (def: 1x)    (multiples of the grid spacing of the B-spline transform (so, differs with the resolution))


    ## Metric2: Corresponding landmarks (MUST add it as the last metric)
    if eval(kwargs["use_landmarks"]):
        Elastix.SetFixedPointSetFileName(kwargs["I_f_landmarks_filename"])
        Elastix.SetMovingPointSetFileName(kwargs["I_m_landmarks_filename"])
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
    meanVoxSpc = 1.25
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
    intrpltr_BsplineOrdr = 5  # avail range: 0 -> 5
    pMap['FixedImageBSplineInterpolationOrder'] = [str(intrpltr_BsplineOrdr)]  # When using a RandomCoordinate sampler, the fixed image needs to be interpolated @ each iteration
    pMap['Interpolator'] = ['BSplineInterpolator']                 # use 'LinearInterpolator' for faster performance
    pMap['BSplineInterpolationOrder'] = [str(intrpltr_BsplineOrdr)]
    if I_m.GetMetaData('bitpix') == '8':
        pMap['FinalBSplineInterpolationOrder'] = ['0']             # A MUST when deforming a "binary" image (i.e. segmentation)
        # pMap['FixedImagePixelType'] = ['unsigned char']            # [BUG] ineffective !! use sitk casting on the result instead !
        # pMap['MovingImagePixelType'] = ['unsigned char']
        # pMap['ResultImagePixelType'] = ['unsigned char']
    else:
        pMap['FinalBSplineInterpolationOrder'] = [str(intrpltr_BsplineOrdr)]
    pMap['ResampleInterpolator'] = ['FinalBSplineInterpolator']           # To generate the final result, i.e. the deformed result of the registration

    ## Result-related params
    pMap['ResultImageFormat'] = ['nii']
    pMap['WriteTransformParametersEachResolution'] = ['true']
    pMap['WriteResultImageAfterEachResolution'] = ['true']
    pMap['WriteResultImageAfterEachIteration'] = ['false']


    Elastix.SetParameterMap(pMap)

    Elastix.SetLogToConsole(True)  # to see output of each iteration
    Elastix.SetLogToFile(True)
    Elastix.WriteParameterFile(Elastix.GetParameterMap()[0], f'{Elastix.GetOutputDirectory()}/pMap0.txt')
    Elastix.PrintParameterMap()  # for confirmation

    try:
        Elastix.Execute()
    except:
        return False


    # clip the result to the range of I_m (to clean interpolation noise at edges)
    resultImage = Elastix.GetResultImage()
    if I_m.GetMetaData('bitpix') == '8':
        resultImage = sitk.Cast(resultImage, sitk.sitkUInt8)
    resultImage_arr = sitk.GetArrayFromImage(resultImage)
    I_m_arr = sitk.GetArrayFromImage(I_m)
    resultImage_arr = resultImage_arr.clip(min=I_m_arr.min(), max=I_m_arr.max())

    # write result using nibabel (not sitk.WriteImage() as it screws the header (sform_code & qform_code & quaternion))
    I_f_read_sitk_write_nib = nib.Nifti1Image(resultImage_arr.swapaxes(0, 2),        # swapping the local im coords (aka. lattice) due to diff bet nibabel & itk image axes
                                              nib.load(kwargs['I_f_filename']).affine)                                    # use the image resulting from Elastix registration with the affine transfrom coming from the original data (e.g. fixed im)
    try:
        I_f_read_sitk_write_nib.to_filename(f'{Elastix.GetOutputDirectory()}/{kwargs["I_deformed_filename"]}')
        # sitk.WriteImage(resultImage, f'{Elastix.kwargs[GetOutputDirectory()}/{"I_deformed_filename"]}')
    except:
        I_f_read_sitk_write_nib.to_filename(f'{Elastix.GetOutputDirectory()}/__TEMP_NAME__{kwargs["n_res"]}x{kwargs["n_itr"]}.nii')

    os.chdir(cwd)               # back to the working directory before entering this method
    return True



##################################################################################################################################


if __name__ == '__main__':
    if callElastix(dataset_path = sys.argv[1],
                   I_f_filename = sys.argv[2],
                   I_m_filename = sys.argv[3],
                   I_f_mask_filename = sys.argv[4],
                   I_m_mask_filename = sys.argv[5],
                   use_I_m_mask = sys.argv[6],
                   rigid_alignment_transform__filename = sys.argv[7],
                   I_m_rigidityCoeffIm_filename = sys.argv[8],
                   reg_type = sys.argv[9],
                   n_itr = sys.argv[10],
                   n_res = sys.argv[11],
                   use_rigidityPenalty = sys.argv[12],
                   use_landmarks = sys.argv[13],
                   I_f_landmarks_filename = sys.argv[14],
                   I_m_landmarks_filename = sys.argv[15],
                   I_deformed_filename = sys.argv[16]):
        exit(0)
    else:
        exit(-1)