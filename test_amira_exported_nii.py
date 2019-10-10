import numpy as np

p = np.array([4, 7, 34, 1])
affine_fromAmira = np.array([1.25, 0, 0, 0, 0, 0, 1.25, 0, 0, -1.25, 0, 0, 59.661, -43.8194, 21.2107, 1]).reshape(4,4).transpose()
affine_fromNibabel = nib.load('S:\MR_datasets_local\Sub_3-2018-09-11\R1_t1-minus-t2.nii').affine

print('affine_fromAmira: \n', affine_fromAmira)
print('affine_fromNibabel: \n', affine_fromNibabel)

# both should be the same but reflected around z (i.e. negated in x & y)
print('affine_fromAmira @ p : \n', affine_fromAmira @ p)
print('affine_fromNibabel @ p : \n', affine_fromNibabel @ p)
