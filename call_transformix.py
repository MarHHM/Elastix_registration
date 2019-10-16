import os
import sys
from pathlib import Path

import SimpleITK as sitk  # see doc at     https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html
import nibabel as nib


def call_transformix(dataset_path, im_to_deform_filename, pMap_filename, output_filename):
    print('\n###################### TRANSFORMIX ######################\n')
    cwd = os.getcwd()
    os.chdir(dataset_path)

    Transformix = sitk.TransformixImageFilter()
    pMap__dir = str(pMap_filename).split('\\')[0]
    Transformix.SetOutputDirectory(f"{pMap__dir}/{output_filename.stem}")
    # Transformix.SetOutputDirectory(f"./{output_filename.stem}")
    os.makedirs(Transformix.GetOutputDirectory(), exist_ok=True)
    im_to_deform = sitk.ReadImage(im_to_deform_filename.name)
    Transformix.SetMovingImage(im_to_deform)
    pMap = sitk.ReadParameterFile(f"{dataset_path}/{pMap_filename}")
    if im_to_deform.GetMetaData('bitpix') == '8' :
        pMap['FinalBSplineInterpolationOrder'] = ['0']          # A MUST when deforming a "binary" image (i.e. segmentation) (see doc)
        # pMap_filename['ResultImagePixelType'] = ["unsigend short"]       # not effective ! use sitk casting on the result instead !
    pMap['DefaultPixelValue'] = [str(sitk.GetArrayFromImage(Transformix.GetMovingImage()).min())]                    # sets pixel values outside the moving image grid (at interpolation) -> set it to <= the min in your dataset (i.e. bgd)
    pMap['ResultImageFormat'] = ['nii']

    # workaround to solve this issue      <https://groups.google.com/forum/#!category-topic/elastix-imageregistration/simpleelastix/TlAbmFE8TPw>
    init_transform_param_filename = pMap['InitialTransformParametersFileName'][0]
    if  init_transform_param_filename != 'NoInitialTransform' :
        pMap['InitialTransformParametersFileName'] = ['NoInitialTransform']
        pMap_init = sitk.ReadParameterFile(init_transform_param_filename)
        Transformix.SetTransformParameterMap(pMap_init)
    Transformix.AddTransformParameterMap(pMap)

    Transformix.ComputeDeformationFieldOn()
    Transformix.ComputeSpatialJacobianOff()                # takes time
    Transformix.ComputeDeterminantOfSpatialJacobianOn()

    Transformix.SetLogToConsole(True)
    Transformix.SetLogToFile(True)
    # Transformix.PrintParameterMap()     # for confirmation

    Transformix.Execute()       # automatically writes the resulting "spatialJacobian" & "FullSpatialJacobian"

    resultImage = Transformix.GetResultImage()
    if im_to_deform.GetMetaData('bitpix') == '8':
        resultImage = sitk.Cast(resultImage, sitk.sitkUInt8)
    ## write result using nibabel (not sitk.WriteImage() as it screws the header (sform_code & qform_code & quaternion))
    I_f_read_sitk_write_nib = nib.Nifti1Image(sitk.GetArrayFromImage(resultImage).swapaxes(0, 2),
                                              nib.load(im_to_deform_filename.name).affine)                    # use the image resulting from Elastix registration with the affine transfrom coming from the original data (e.g. fixed im)
    I_f_read_sitk_write_nib.to_filename(f'{Transformix.GetOutputDirectory()}/{output_filename.name}')
    # sitk.WriteImage(Transformix.GetResultImage(), f'{Transformix.GetOutputDirectory()}/{output_filename}')


    ### TESTING reading deformation field
    # deformation_fld = Transformix.GetDeformationField()
    # q = deformation_fld.GetDimension()
    # i = deformation_fld.GetNumberOfPixels()
    # pix = deformation_fld.GetPixel((1,4,5))         # doesn't work !
    # sitk.WriteImage(deformation_fld, 'deformation_fld.mhd')     # can't write it as .nii !, & as .mhd -> empty !!
    #
    # # t = Transformix.GetTransformParameter()
    # # d = sitk.TransformToDisplacementField(transformParameterMap[0])
    #
    # # hh = sitk.GetArrayFromImage(resultImage)
    # # gg = sitk.GetArrayFromImage(deformation_fld)
    # # mag_deformation_fld = sitk.VectorMagnitude(deformation_fld)
    # # sitk.WriteImage(mag_deformation_fld, 'mag_deformation_fld.nii')

    os.chdir(cwd)               # back to the working directory before entering this method
###################################################################################################
if __name__ == '__main__':          # in case "callTransformix" is called directly by the interpreter
	callTransformix(*sys.argv)