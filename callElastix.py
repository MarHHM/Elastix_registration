import os
import sys
from pathlib import Path

import SimpleITK as sitk  # see doc at     https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html
import nibabel as nib


def callElastix(**kwargs):

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

    #### see doc of params in  http://elastix.isi.uu.nl/doxygen/parameter.html, or list of important params at https://simpleelastix.readthedocs.io/ParameterMaps.html#important-parameters
    rigid_pMap = sitk.GetDefaultParameterMap('rigid')   # 'translation' - 'rigid' - 'affine' - 'bspline' (defaults)

    rigid_pMap['FixedImageDimension'] = [str(I_f.GetDimension())]
    rigid_pMap['MovingImageDimension'] = [str(I_m.GetDimension())]
    rigid_pMap['Registration'] = ['MultiMetricMultiResolutionRegistration']
    rigid_pMap['AutomaticScalesEstimation'] = ['true']
    rigid_pMap['MaximumNumberOfIterations'] = [kwargs['n_itr']]
    rigid_pMap['ErodeMask'] = ['true']
    rigid_pMap['ImageSampler'] = ['RandomCoordinate']  # {Random, Full, Grid, RandomCoordinate}
    rigid_pMap['Interpolator'] = ['BSplineInterpolator']  # use 'LinearInterpolator' for faster performance
    rigid_pMap['DefaultPixelValue'] = [str(sitk.GetArrayFromImage(Elastix.GetMovingImage(0)).min())]  # sets pixel values outside the moving image grid (at interpolation) -> set it to <= the min in your dataset (i.e. bgd)

    ### interpolators
    intrpltr_BsplineOrdr = 5                # avail range: 0 -> 5
    rigid_pMap['FixedImageBSplineInterpolationOrder'] = [str(intrpltr_BsplineOrdr)]  # When using a RandomCoordinate sampler, the fixed image needs to be interpolated @ each iteration
    rigid_pMap['Interpolator'] = ['BSplineInterpolator']  # use 'LinearInterpolator' for faster performance
    rigid_pMap['BSplineInterpolationOrder'] = [str(intrpltr_BsplineOrdr)]
    rigid_pMap['ResampleInterpolator'] = ['FinalBSplineInterpolator']           # To generate the final result, i.e. the deformed result of the registration
    if I_m.GetMetaData('bitpix') == '8':
        rigid_pMap['FinalBSplineInterpolationOrder'] = ['0']  # A MUST when deforming a "binary" image (i.e. segmentation)
    else:
        rigid_pMap['FinalBSplineInterpolationOrder'] = [str(intrpltr_BsplineOrdr)]

    rigid_pMap['Optimizer'] = ['AdaptiveStochasticGradientDescent']  # 11 optimizers avail: 'StandardGradientDescent', 'AdaptiveStochasticGradientDescent', CMAEvolutionStrategy, ConjugateGradient, ConjugateGradientFRPR, FiniteDifferenceGradientDescent, FullSearch, QuasiNewtonLBFGS, RegularStepGradientDescent, RSGDEachParameterApart, SimultaneousPerturbation
    rigid_pMap['AutomaticParameterEstimation'] = ['true']  # default: true
    rigid_pMap['NumberOfResolutions'] = [kwargs["n_res"]]
    rigid_pMap['ResultImageFormat'] = ['nii']
    rigid_pMap['WriteTransformParametersEachResolution'] = ['true']
    rigid_pMap['WriteResultImageAfterEachResolution'] = ['true']



    if kwargs['reg_type'] == 'NON_RIGID':
        nonRigid_pMap = sitk.ReadParameterFile('C:/Users/bzfmuham/OneDrive/Knee-Kinematics/Code_and_utils/'
                                               'Elastix_registration/Par0004.bs_base.NRP08.All.txt')
        Elastix.SetInitialTransformParameterFileName(kwargs['rigid_alignment_transform__filename'])

        nonRigid_pMap['BSplineTransformSplineOrder'] = ['3']

        nonRigid_pMap['FixedImageDimension'] = [str(I_f.GetDimension())]
        nonRigid_pMap['MovingImageDimension'] = [str(I_m.GetDimension())]

        ## Registration approach
        nonRigid_pMap['Registration'] = ['MultiMetricMultiResolutionRegistration']
        nonRigid_pMap['UseDirectionCosines'] = ['true']                 # def: true. Highly recommended, as it takes into account the "image-to-world" affine transform that comes from the scanner.

        ## Multi-resolution settings (b-spline grid & image pyramid)
        nonRigid_pMap['NumberOfResolutions'] = [kwargs["n_res"]]        # default: 4
        nonRigid_pMap['FinalGridSpacingInVoxels'] = ['4.0']             # couldn't go lower than 2 for Sub_3 dataset. what is used in Staring 2007: 8.0; will be automatically approximated for each dimension during execution
        nonRigid_pMap['GridSpacingSchedule'] = [str(2 ** (int(kwargs["n_res"]) - 1 - i)) for i in range(int(kwargs["n_res"]))]  # generate using "list comprehension", default for 4 resolutions: ['2**3', '2**2', '2**1', '2**0']
        nonRigid_pMap['FixedImagePyramid'] = ['FixedSmoothingImagePyramid']  # no downsampling needed when a random sampler is used (what's used in Staring 2007 was 'RecursiveImagePyramid' but they use )
        nonRigid_pMap['MovingImagePyramid'] = ['MovingSmoothingImagePyramid']
        imPyrSchdl_lst = []
        for i in range(int(kwargs["n_res"])):
            for j in range(I_m.GetDimension()):  # just to repeat the factor for all dims per resolution
                imPyrSchdl_lst.append(str(2 ** (int(kwargs["n_res"]) - 1 - i)))
        nonRigid_pMap['ImagePyramidSchedule'] = imPyrSchdl_lst  # MUST be specified for all resolutions & DIMS in order to work correctly, otherwise the default schedule is created !!
        nonRigid_pMap['WritePyramidImagesAfterEachResolution'] = ['false']

        ## Metric0: similarity
        nonRigid_pMap['Metric'] = ['AdvancedMattesMutualInformation']
        nonRigid_pMap['NumberOfHistogramBins'] = ['32']
        nonRigid_pMap['UseFastAndLowMemoryVersion'] = ['false']          # def: true. false can take up HUGE RAM !!

        # nonRigid_pMap['Metric'] = ['AdvancedKappaStatistic']              # good as a similariy cost for registering binary images (i.e. masks)
        nonRigid_pMap['Metric0Weight'] = ['1']
        if eval(kwargs["use_rigidityPenalty"]):
            nonRigid_pMap['Metric'] = nonRigid_pMap['Metric'] + ('TransformRigidityPenalty',)
            nonRigid_pMap['MovingRigidityImageName'] = [kwargs['I_m_rigidityCoeffIm_filename']]
            nonRigid_pMap['Metric1Weight'] = ['0.1', '0.1', '4', '4']  # ['0.1', '0.1', '0.1', '4']   ['0.5', '0.3', '1000', '1000']
            nonRigid_pMap['DilateRigidityImages'] = ['false', 'false', 'true', 'true', 'true']
            nonRigid_pMap['DilationRadiusMultiplier'] = ['2.0']  # originaly: 2.0
            nonRigid_pMap['OrthonormalityConditionWeight'] = ['1.0']  # originaly: 1.0     (rigidity preservation)
            nonRigid_pMap['PropernessConditionWeight'] = ['2.0']  # originaly: 2.0          (volume preservation)

        nonRigid_pMap['Optimizer'] = ['AdaptiveStochasticGradientDescent']      # 16 optimizers avail.
        nonRigid_pMap['MaximumNumberOfIterations'] = [kwargs['n_itr']]          # reasonable range: 250 -> 2000 (500 is a good compromise)
        nonRigid_pMap['AutomaticParameterEstimation'] = ['true']
        nonRigid_pMap['ASGDParameterEstimationMethod'] = ['Original']       # Original || DisplacementDistribution-> more efficient estimation of intial step size (Qiao et al., 2016 in elastix doc)
        nonRigid_pMap['UseAdaptiveStepSizes'] = ['true']
        # meanVoxSpc = 1.25
        # nonRigid_pMap['MaximumStepLength'] = [str(meanVoxSpc*175)]                       # Default: mean voxel spacing of I_f & I_m (1.25 for Sub_3). It's the maximum voxel displacement between two iterations. The larger this parameter, the more aggressive the optimization.
        # nonRigid_pMap['ValueTolerance'] = ['1']                           # seems ineffective for 'AdaptiveStochasticGradientDescent'
        # nonRigid_pMap['GradientMagnitudeTolerance'] = ['0.000944']        # seems ineffective for 'AdaptiveStochasticGradientDescent'
        nonRigid_pMap['DefaultPixelValue'] = [str(sitk.GetArrayFromImage(Elastix.GetMovingImage(0)).min())]  # sets pixel values outside the moving image grid (at interpolation) -> set it to <= the min in your dataset (i.e. bgd)
        nonRigid_pMap['ErodeMask'] = ['true']               # better to use it for multi-res registration    (see   http://lists.bigr.nl/pipermail/elastix/2011-November/000627.html)

        ### I_f sampler
        nonRigid_pMap['ImageSampler'] = ['RandomCoordinate']        # {Random, Full, Grid, RandomCoordinate}
        nonRigid_pMap['NumberOfSpatialSamples'] = ['5000']          # def: 5000. Don't go lower than 2000
        nonRigid_pMap['NewSamplesEveryIteration'] = ['true']        # def: False. But highly recommended to use it specailly with random samplers.
        nonRigid_pMap['UseRandomSampleRegion'] = ['false']
        # nonRigid_pMap['SampleRegionSize'] = ['imSz_X_mm/3', 'imSz_Y_mm/3', 'imSz_Z_mm/3']         # if 'UseRandomSampleRegion' is used, this param is effective. VIP -> must specified in mm & for each dim (not each res !). Default: imSz_mm/3


        ### interpolators
        nonRigid_pMap['FixedImageBSplineInterpolationOrder'] = [str(intrpltr_BsplineOrdr)]  # When using a RandomCoordinate sampler, the fixed image needs to be interpolated @ each iteration
        nonRigid_pMap['Interpolator'] = ['BSplineInterpolator']                 # use 'LinearInterpolator' for faster performance
        nonRigid_pMap['BSplineInterpolationOrder'] = [str(intrpltr_BsplineOrdr)]
        nonRigid_pMap['ResampleInterpolator'] = ['FinalBSplineInterpolator']           # To generate the final result, i.e. the deformed result of the registration
        if I_m.GetMetaData('bitpix') == '8':
            nonRigid_pMap['FinalBSplineInterpolationOrder'] = ['0']             # A MUST when deforming a "binary" image (i.e. segmentation)
            # nonRigid_pMap['FixedImagePixelType'] = ['unsigned char']            # [BUG] ineffective !! use sitk casting on the result instead !
            # nonRigid_pMap['MovingImagePixelType'] = ['unsigned char']
            # nonRigid_pMap['ResultImagePixelType'] = ['unsigned char']
        else:
            nonRigid_pMap['FinalBSplineInterpolationOrder'] = [str(intrpltr_BsplineOrdr)]
            # nonRigid_pMap['FixedImagePixelType'] = ['float']
            # nonRigid_pMap['MovingImagePixelType'] = ['float']
            # nonRigid_pMap['ResultImagePixelType'] = ['float']

        nonRigid_pMap['ResultImageFormat'] = ['nii']
        nonRigid_pMap['WriteTransformParametersEachResolution'] = ['true']
        nonRigid_pMap['WriteResultImageAfterEachResolution'] = ['true']
        nonRigid_pMap['WriteResultImageAfterEachIteration'] = ['false']

    if eval(kwargs["use_landmarks"]):         # must add it as the last metric
        Elastix.SetFixedPointSetFileName(kwargs["I_f_landmarks_filename"])
        Elastix.SetMovingPointSetFileName(kwargs["I_m_landmarks_filename"])
        rigid_pMap['Metric'] = rigid_pMap['Metric'] + ('CorrespondingPointsEuclideanDistanceMetric',)
        if kwargs['reg_type'] == 'NON_RIGID':
            nonRigid_pMap['Metric'] = nonRigid_pMap['Metric'] + ('CorrespondingPointsEuclideanDistanceMetric',)
            metric_CorrLndmrks_wt = 0.01
            if eval(kwargs["use_rigidityPenalty"]):
                nonRigid_pMap['Metric2Weight'] = [str(metric_CorrLndmrks_wt)]
            else:
                nonRigid_pMap['Metric1Weight'] = [str(metric_CorrLndmrks_wt)]

    if kwargs['reg_type'] == 'RIGID':
        Elastix.SetParameterMap(rigid_pMap)
    elif kwargs['reg_type'] == 'NON_RIGID':
        Elastix.SetParameterMap(nonRigid_pMap)
    # elifkwargs[ 'reg_type'] == 'PRE_ALLIGNED_NON_RIGID':
    #     Elastix.SetParameterMap(rigid_pMap)
    #     Elastix.AddParameterMap(nonRigid_pMap)
    #     Elastix.WriteParameterFile(Elastix.GetParameterMap()[0], f'{Elastix.GetOutputDirectory()}/pMap0.txt')
    #     Elastix.WriteParameterFile(Elastix.GetParameterMap()[1], f'{Elastix.GetOutputDirectory()}/pMap1.txt')

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
    I_f_read_sitk_write_nib = nib.Nifti1Image(resultImage_arr.swapaxes(0, 2),  # swapping due to diff bet nibabel & itk image axes
                                              nib.load(kwargs['I_f_filename']).affine)                                    # use the image resulting from Elastix registration with the affine transfrom coming from the original data (e.g. fixed im)
    try:
        I_f_read_sitk_write_nib.to_filename(f'{Elastix.GetOutputDirectory()}/{kwargs["I_deformed_filename"]}')
        # sitk.WriteImage(resultImage, f'{Elastix.kwargs[GetOutputDirectory()}/{"I_deformed_filename"]}')
    except:
        I_f_read_sitk_write_nib.to_filename(f'{Elastix.GetOutputDirectory()}/__TEMP_NAME__{kwargs["n_res"]}x{kwargs["n_itr"]}.nii')

    os.chdir(cwd)               # back to the working directory before entering this method
    return True
#################################################################
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