import numpy as np
import cv2
import os
import glob


def visualize(img, mask, index, save_path):
	"""
	:param img: with shape [height, width]
	:param label: with shape [height, width]
	:return: image + mask with shape [height, width, 3]
	"""
	assert img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]
	height = img.shape[0]
	width = img.shape[1]

	img = img.astype(np.uint8)
	mask = mask.astype(np.uint8)

	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

	# -------------------create colormap------------------- #
	colormap = np.zeros((256, 3), dtype=np.uint8)
	ind = np.arange(256, dtype=np.uint8)
	for shift in reversed(range(8)):
		for channel in range(3):
			colormap[:, channel] |= ((ind >> channel) & 1) << shift
		ind >>= 3
	# ------------------------------------------------------ #

	assert mask.ndim == 2
	seg_image = colormap[mask]

	result = cv2.addWeighted(img, 1, seg_image, 0.5, 0)
	path = os.path.join(save_path, "results_" + str(index) + ".jpg")
	cv2.imwrite(path, result)


def confuse_mask_image(image_path, mask_path, mode, save_path):
	img_files = os.listdir(image_path)
	mask_files = os.listdir(mask_path)
	assert len(img_files) == len(mask_files)
	for i in range(len(img_files)):
		img_pattern = os.path.join(image_path, "image" + str(i+1) + ".csv")
		if mode == "label":
			mask_pattern = os.path.join(mask_path, "label" + str(i+1) + ".csv")
		else:
			mask_pattern = os.path.join(mask_path, "seg" + str(i + 1) + ".csv")
		img_dirs = glob.glob(img_pattern)[0]
		mask_dirs = glob.glob(mask_pattern)[0]
		img = np.loadtxt(img_dirs, delimiter=",")
		mask = np.loadtxt(mask_dirs, delimiter=",")
		visualize(img, mask, i+1, save_path)


if __name__ == "__main__":
	image_path = './produce/image'
	label_path = './produce/label'
	seg_path = './produce/segmentation'
	save_path = './img_mask/label'
	confuse_mask_image(image_path, label_path, 'label', save_path)

	"""
	img = np.loadtxt(image_path, delimiter=",")
	label = np.loadtxt(label_path, delimiter=",")
	seg = np.loadtxt(seg_path, delimiter=",")
	visualize(img, label)
	"""