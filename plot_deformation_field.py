import SimpleITK as sitk
import numpy as np

import matplotlib
matplotlib.use('qt5agg')        # backend can affect the speed of plotting - GTK should be the fastest but still need to install it for windows
from matplotlib import pyplot as plt

import mayavi           # good for 3d visualization

import tifffile

#####################################################

pp = sitk.ReadImage(r'S:\datasets\s3_2\2d_highRes\4x10000__I_m-slc_13__sim-MI__Spc-4\deformationField.tif')
det = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(pp))
vectors = sitk.GetArrayFromImage(pp)

x, y = det.shape
X, Y = np.meshgrid(np.arange(0, x, 1), np.arange(0, y, 1))
U = vectors[:, :, 0]
V = vectors[:, :, 1]

print(det.shape)
print(vectors.shape)
print(U.shape)
print(V.shape)

plt.imshow(det, cmap='jet')
plt.colorbar()
# plt.quiver(X, Y, U, V)                # if it's very slow, change the matplotlib backend to GTK before importing pyplot
plt.quiver(Y, X, V, U)