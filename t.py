import SimpleITK as sitk
import cv2
import numpy as np
import tensorflow as tf
from load_from_mhd import read_sequence_with_be
ed_dir = "./CAMUS/val/patient0423/patient0423_2CH_ED.mhd"
es_dir = "./CAMUS/val/patient0423/patient0423_2CH_ES.mhd"
seq_dir = "./CAMUS/val/patient0423/patient0423_2CH_sequence.mhd"

a = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[0, 2, 2], [2, 2, 2], [2, 2, 2]], [[3, 3, 3], [3, 3, 3], [3, 3, 3]]])
b = a[::-1]
print(b)



"""
for i in range(len(seq)):
	img = cv2.resize(seq[i, :, :], (512, 512))
	cv2.imshow("1", img)
	cv2.waitKey(300)

cv2.destroyAllWindows()
"""