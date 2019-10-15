import os
import sys

import SimpleITK as sitk  # see doc at     https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html
import nibabel as nib


def callElastix(self, dataset_path, I_f_filename, I_m_filename, I_f_mask_filename, I_m_mask_filename, use_I_m_mask, rigid_alignment_transform__filename,
                I_m_rigidityCoeffIm_filename, reg_type, n_itr_rigid, n_itr_nonRigid, n_res, use_rigidityPenalty, use_landmarks, I_f_landmarks_filename,
                I_m_landmarks_filename, I_deformed_filename):

    cwd = os.getcwd()
    os.chdir(dataset_path)
    # sitk.ProcessObject_GlobalDefaultDebugOn()       # to see detailed debug info (e.g. image readers... etc.)
    Elastix = sitk.ElastixImageFilter()
    Elastix.SetOutputDirectory(I_deformed_filename.split('.')[0])               # write all registration info there
    os.makedirs(Elastix.GetOutputDirectory(), exist_ok=True)
    Elastix.SetFixedImage(sitk.ReadImage(I_f_filename))           # '72_Echo1.nii' 'DataMergeR4_t1_minus_t2.nii' 'DataMergeR4_echo1.nii'
    I_m = sitk.ReadImage(I_m_filename)
    Elastix.SetMovingImage(I_m)                               # 'DataMergeR3_t1_minus_t2.nii' 'DataMergeR3_echo1.nii'
    Elastix.SetFixedMask(sitk.ReadImage(I_f_mask_filename))       # MUST be specified to avoid taking into account bgd & out-of-view noise during registration -  in the documentation, they say i usually only need to provide fixed image mask (moving image mask needed only when noisy features are close to the edge of the ROI (see 5.5 in manual))
    if eval(use_I_m_mask):
        Elastix.SetMovingMask(sitk.ReadImage(I_m_mask_filename))

    #### see doc of params in  http://elastix.isi.uu.nl/doxygen/parameter.html, or list of important params at https://simpleelastix.readthedocs.io/ParameterMaps.html#important-parameters
    rigid_pMap = sitk.GetDefaultParameterMap('rigid')   # 'translation' - 'rigid' - 'affine' - 'bspline' (defaults)
    rigid_pMap['Registration'] = ['MultiMetricMultiResolutionRegistration']
    rigid_pMap['AutomaticScalesEstimation'] = ['true']
    rigid_pMap['MaximumNumberOfIterations'] = [str(n_itr_rigid)]
    rigid_pMap['ErodeMask'] = ['false']
    rigid_pMap['ImageSampler'] = ['RandomCoordinate']  # {Random, Full, Grid, RandomCoordinate}
    rigid_pMap['Interpolator'] = ['BSplineInterpolator']  # use 'LinearInterpolator' for faster performance
    rigid_pMap['DefaultPixelValue'] = [str(sitk.GetArrayFromImage(Elastix.GetMovingImage(0)).min())]  # sets pixel values outside the moving image grid (at interpolation) -> set it to <= the min in your dataset (i.e. bgd)
    rigid_pMap['ResampleInterpolator'] = ['FinalBSplineInterpolator']
    rigid_pMap['Optimizer'] = ['AdaptiveStochasticGradientDescent']  # 11 optimizers avail: 'StandardGradientDescent', 'AdaptiveStochasticGradientDescent', CMAEvolutionStrategy, ConjugateGradient, ConjugateGradientFRPR, FiniteDifferenceGradientDescent, FullSearch, QuasiNewtonLBFGS, RegularStepGradientDescent, RSGDEachParameterApart, SimultaneousPerturbation
    rigid_pMap['AutomaticParameterEstimation'] = ['true']  # default: true
    rigid_pMap['NumberOfResolutions'] = [str(n_res)]
    rigid_pMap['ResultImageFormat'] = ['nii']
    rigid_pMap['WriteTransformParametersEachResolution'] = ['true']
    rigid_pMap['WriteResultImageAfterEachResolution'] = ['true']



    if reg_type == 'NON_RIGID':
        nonRigid_pMap = sitk.ReadParameterFile('C:/Users/bzfmuham/OneDrive/Knee-Kinematics/elastix_4.9.0/params_from_database/Par0004/Table_1_2_3/Par0004.bs_base.NRP08.All.txt')
        Elastix.SetInitialTransformParameterFileName(rigid_alignment_transform__filename)
        # nonRigid_pMap['UseDirectionCosines'] = ['false']
        nonRigid_pMap['Registration'] = ['MultiMetricMultiResolutionRegistration']  # 'MultiResolutionRegistration'
        nonRigid_pMap['Metric'] = ['AdvancedMattesMutualInformation']
        nonRigid_pMap['NumberOfHistogramBins'] = ['32']
        nonRigid_pMap['Metric0Weight'] = ['1']
        if eval(use_rigidityPenalty):
            nonRigid_pMap['Metric'] = nonRigid_pMap['Metric'] + ('TransformRigidityPenalty',)
            nonRigid_pMap['MovingRigidityImageName'] = [I_m_rigidityCoeffIm_filename]
            nonRigid_pMap['Metric1Weight'] = ['0.1', '0.1', '0.1', '4']  # ['0.1', '0.1', '0.1', '4']   ['0.5', '0.3', '1000', '1000']
            nonRigid_pMap['DilateRigidityImages'] = ['false', 'false', 'true', 'true']
            nonRigid_pMap['DilationRadiusMultiplier'] = ['2.0']  # originaly: 2.0
            nonRigid_pMap['OrthonormalityConditionWeight'] = ['1.0']  # originaly: 1.0     (rigidity preservation)
            nonRigid_pMap['PropernessConditionWeight'] = ['2.0']  # originaly: 2.0          (volume preservation)

        nonRigid_pMap['MaximumNumberOfIterations'] = [str(n_itr_nonRigid)]  # reasonable range: 250 -> 2000 (500 is a good compromise)
        nonRigid_pMap['Optimizer'] = ['AdaptiveStochasticGradientDescent']  # 11 optimizers avail: 'StandardGradientDescent', 'AdaptiveStochasticGradientDescent', CMAEvolutionStrategy, ConjugateGradient, ConjugateGradientFRPR, FiniteDifferenceGradientDescent, FullSearch, QuasiNewtonLBFGS, RegularStepGradientDescent, RSGDEachParameterApart, SimultaneousPerturbation
        nonRigid_pMap['AutomaticParameterEstimation'] = ['true']
        # nonRigid_pMap['ValueTolerance'] = ['1']
        # nonRigid_pMap['GradientMagnitudeTolerance'] = ['0.000944']
        nonRigid_pMap['DefaultPixelValue'] = [str(sitk.GetArrayFromImage(Elastix.GetMovingImage(0)).min())]  # sets pixel values outside the moving image grid (at interpolation) -> set it to <= the min in your dataset (i.e. bgd)
        nonRigid_pMap['FixedImagePixelType'] = ['float']
        nonRigid_pMap['MovingImagePixelType'] = ['float']
        nonRigid_pMap['ErodeMask'] = ['false']
        nonRigid_pMap['ImageSampler'] = ['RandomCoordinate']  # {Random, Full, Grid, RandomCoordinate}
        nonRigid_pMap['FixedImageBSplineInterpolationOrder'] = ['3']  # When using a RandomCoordinate sampler, the fixed image needs to be interpolated @ each iterationnonRigid_pMap['Interpolator'] = ['BSplineInterpolator']  # use 'LinearInterpolator' for faster performance
        nonRigid_pMap['BSplineInterpolationOrder'] = ['3']
        nonRigid_pMap['ResampleInterpolator'] = ['FinalBSplineInterpolator']

        nonRigid_pMap['NumberOfResolutions'] = [str(n_res)]  # default: 4
        nonRigid_pMap['FinalGridSpacing'] = ['8']

        nonRigid_pMap['ResultImagePixelType'] = ['float']
        nonRigid_pMap['ResultImageFormat'] = ['nii']
        nonRigid_pMap['WriteTransformParametersEachResolution'] = ['true']
        nonRigid_pMap['WriteResultImageAfterEachResolution'] = ['true']

    if eval(use_landmarks):  # must add it as the last metric
        Elastix.SetFixedPointSetFileName(I_f_landmarks_filename)
        Elastix.SetMovingPointSetFileName(I_m_landmarks_filename)
        rigid_pMap['Metric'] = rigid_pMap['Metric'] + ('CorrespondingPointsEuclideanDistanceMetric',)
        if reg_type == 'NON_RIGID' or reg_type == 'PRE_ALLIGNED_NON_RIGID':
            nonRigid_pMap['Metric'] = nonRigid_pMap['Metric'] + ('CorrespondingPointsEuclideanDistanceMetric',)

    if reg_type == 'RIGID':
        Elastix.SetParameterMap(rigid_pMap)
        Elastix.WriteParameterFile(Elastix.GetParameterMap()[0], f'{Elastix.GetOutputDirectory()}/pMap0.txt')
    elif reg_type == 'NON_RIGID':
        Elastix.SetParameterMap(nonRigid_pMap)
        Elastix.WriteParameterFile(Elastix.GetParameterMap()[0], f'{Elastix.GetOutputDirectory()}/pMap0.txt')
    elif reg_type == 'PRE_ALLIGNED_NON_RIGID':
        Elastix.SetParameterMap(rigid_pMap)
        Elastix.AddParameterMap(nonRigid_pMap)
        Elastix.WriteParameterFile(Elastix.GetParameterMap()[0], f'{Elastix.GetOutputDirectory()}/pMap0.txt')
        Elastix.WriteParameterFile(Elastix.GetParameterMap()[1], f'{Elastix.GetOutputDirectory()}/pMap1.txt')

    Elastix.SetLogToConsole(True)  # to see output of each iteration
    Elastix.SetLogToFile(True)
    Elastix.PrintParameterMap()  # for confirmation
    Elastix.Execute()

    # clip the result to the range of I_m (to clean interpolation noise at edges)
    resultImage_arr = sitk.GetArrayFromImage(Elastix.GetResultImage())
    I_m_arr = sitk.GetArrayFromImage(I_m)
    resultImage_arr = resultImage_arr.clip(min=I_m_arr.min(), max=I_m_arr.max())

    # write result using nibabel (not sitk.WriteImage() as it screws the header (sform_code & qform_code & quaternion))
    I_f_read_sitk_write_nib = nib.Nifti1Image(resultImage_arr.swapaxes(0, 2),  # swapping due to diff bet nibabel & itk image axes
                                              nib.load(I_f_filename).affine)                                    # use the image resulting from Elastix registration with the affine transfrom coming from the original data (e.g. fixed im)
    I_f_read_sitk_write_nib.to_filename(f'{Elastix.GetOutputDirectory()}/{I_deformed_filename}')
    # sitk.WriteImage(resultImage, f'{Elastix.GetOutputDirectory()}/{I_deformed_filename}')

    os.chdir(cwd)               # back to the working directory before entering this method
#################################################################
if __name__ == '__main__':
    callElastix(*sys.argv)              # unpack sys.argv