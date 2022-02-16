import os
import SimpleITK as sitk
import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
from Configuration import normalize
import tensorflow as tf


def load_mhd(path):
	image = sitk.ReadImage(path)
	image = sitk.GetArrayFromImage(image)
	return image


def read_mhd(root, mode, phase, height, width):
	assert phase == "ED" or phase == "ES" or phase == "E?"  # "E?" means ED and ES
	assert mode == "training" or mode == "testing" or mode == "val"

	image_list = []
	n_image_list = []
	label_list = []

	# mhd contains information and refer to raw images
	root = os.path.join(root, mode)
	patient_files = os.listdir(root)
	for patient_file in patient_files:
		# files in the filefolder
		temp_root = os.path.join(root, patient_file)
		img_pattern = os.path.join(temp_root, "*_*_" + phase + ".mhd")
		label_pattern = os.path.join(temp_root, "*_*_" + phase + "_gt.mhd")
		img_dirs = glob.glob(img_pattern)
		label_dirs = glob.glob(label_pattern)
		if len(img_dirs) == len(label_dirs):
			for img_dir, label_dir in zip(img_dirs, label_dirs):
				print(img_dir)
				print(label_dir)
				image = sitk.ReadImage(img_dir)
				label = sitk.ReadImage(label_dir)
				image = sitk.GetArrayFromImage(image)
				label = sitk.GetArrayFromImage(label)
				assert image.shape[0] == 1 and label.shape[0] == 1
				image = image[0]
				label = label[0]
				image = cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)
				label = cv.resize(label, (width, height), interpolation=cv.INTER_NEAREST)
				n_image = normalize(image)
				image = np.reshape(image, [height, width, 1])
				n_image = np.reshape(n_image, [height, width, 1])
				label = np.reshape(label, [height, width, 1])
				image_list.append(image)
				n_image_list.append(n_image)
				label_list.append(label)
		else:
			for img_dir in img_dirs:
				image = sitk.ReadImage(img_dir)
				image = sitk.GetArrayFromImage(image)
				assert image.shape[0] == 1
				image = image[0]
				image = cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)
				n_image = normalize(image)
				image = np.reshape(image, [height, width, 1])
				n_image = np.reshape(n_image, [height, width, 1])
				image_list.append(image)
				n_image_list.append(n_image)

	image_list = np.asarray(image_list)
	n_image_list = np.asarray(n_image_list)
	label_list = np.asarray(label_list)

	if mode == 'training':
		shuffle_index = np.random.permutation(len(image_list))
		image_list = image_list[shuffle_index]
		n_image_list = n_image_list[shuffle_index]
		label_list = label_list[shuffle_index]

	return image_list, n_image_list, label_list


def read_mhd_S(root, mode, phase, height, width, structure):
	assert phase == "ED" or phase == "ES" or phase == "E?"  # "E?" means ED and ES
	assert mode == "training" or mode == "testing" or mode == "val"

	image_list = []
	n_image_list = []
	label_list = []

	# mhd contains information and refer to raw images
	root = os.path.join(root, mode)
	patient_files = os.listdir(root)
	for patient_file in patient_files:
		# files in the filefolder
		temp_root = os.path.join(root, patient_file)
		img_pattern = os.path.join(temp_root, "*_" + structure + "_" + phase + ".mhd")
		label_pattern = os.path.join(temp_root, "*_" + structure + "_" + phase + "_gt.mhd")
		img_dirs = glob.glob(img_pattern)
		label_dirs = glob.glob(label_pattern)
		if len(img_dirs) == len(label_dirs):
			for img_dir, label_dir in zip(img_dirs, label_dirs):
				print(img_dir)
				print(label_dir)
				image = sitk.ReadImage(img_dir)
				label = sitk.ReadImage(label_dir)
				image = sitk.GetArrayFromImage(image)
				label = sitk.GetArrayFromImage(label)
				assert image.shape[0] == 1 and label.shape[0] == 1
				image = image[0]
				label = label[0]
				image = cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)
				label = cv.resize(label, (width, height), interpolation=cv.INTER_NEAREST)
				n_image = normalize(image)
				image = np.reshape(image, [height, width, 1])
				n_image = np.reshape(n_image, [height, width, 1])
				label = np.reshape(label, [height, width, 1])
				image_list.append(image)
				n_image_list.append(n_image)
				label_list.append(label)
		else:
			for img_dir in img_dirs:
				image = sitk.ReadImage(img_dir)
				image = sitk.GetArrayFromImage(image)
				assert image.shape[0] == 1
				image = image[0]
				image = cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)
				n_image = normalize(image)
				image = np.reshape(image, [height, width, 1])
				n_image = np.reshape(n_image, [height, width, 1])
				image_list.append(image)
				n_image_list.append(n_image)

	image_list = np.asarray(image_list)
	n_image_list = np.asarray(n_image_list)
	label_list = np.asarray(label_list)

	if mode == 'training':
		shuffle_index = np.random.permutation(len(image_list))
		image_list = image_list[shuffle_index]
		n_image_list = n_image_list[shuffle_index]
		label_list = label_list[shuffle_index]

	return image_list, n_image_list, label_list


def read_sequence_mhd(root, mode, height, width):
	assert mode == "training" or mode == "testing"

	image_list = []
	n_image_list = []

	# mhd contains information and refer to raw images
	root = os.path.join(root, mode)
	patient_files = os.listdir(root)
	for patient_file in patient_files:
		# files in the filefolder
		temp_root = os.path.join(root, patient_file)
		img_pattern = os.path.join(temp_root, "*_*_sequence.mhd")
		img_dirs = glob.glob(img_pattern)

		for img_dir in img_dirs:
			print(img_dir)
			images = sitk.ReadImage(img_dir)
			images = sitk.GetArrayFromImage(images)
			for i in range(0, len(images), 10):  # sample unlabeled every 10 frames
				image = images[i]
				image = cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)
				n_image = normalize(image)
				image = np.reshape(image, [height, width, 1])
				n_image = np.reshape(n_image, [height, width, 1])
				image_list.append(image)
				n_image_list.append(n_image)

	image_list = np.asarray(image_list)
	n_image_list = np.asarray(n_image_list)

	if mode == 'training':
		shuffle_index = np.random.permutation(len(image_list))
		image_list = image_list[shuffle_index]
		n_image_list = n_image_list[shuffle_index]

	return image_list, n_image_list


def read_sequence_with_be(root, structure, height, width):
	assert structure == "2CH" or structure == "4CH"

	seq_list = []
	n_seq_list = []

	seq_pattern = os.path.join(root, "*_" + str(structure) + "_sequence.mhd")
	ed_pattern = os.path.join(root, "*_" + str(structure) + "_ED.mhd")
	ed_label_pattern = os.path.join(root, "*_" + str(structure) + "_ED_gt.mhd")
	es_pattern = os.path.join(root, "*_" + str(structure) + "_ES.mhd")
	es_label_pattern = os.path.join(root, "*_" + str(structure) + "_ES_gt.mhd")
	seq_dir = glob.glob(seq_pattern)[0]
	ed_dir = glob.glob(ed_pattern)[0]
	ed_label_dir = glob.glob(ed_label_pattern)[0]
	es_dir = glob.glob(es_pattern)[0]
	es_label_dir = glob.glob(es_label_pattern)[0]

	seq = load_mhd(seq_dir)
	ed = load_mhd(ed_dir)[0]
	ed_label = load_mhd(ed_label_dir)[0]
	es = load_mhd(es_dir)[0]
	es_label = load_mhd(es_label_dir)[0]

	if (ed == seq[0]).all() and (es == seq[-1]).all():
		# the begin of sequence is ED and the end of sequence is ES
		for i in range(len(seq)):
			img = seq[i]
			img = cv.resize(img, (width, height), interpolation=cv.INTER_LINEAR)
			n_img = normalize(img)
			img = np.reshape(img, [height, width, 1])
			n_img = np.reshape(n_img, [height, width, 1])
			seq_list.append(img)
			n_seq_list.append(n_img)

	seq_list = np.asarray(seq_list)
	n_seq_list = np.asarray(n_seq_list)
	ed_label = cv.resize(ed_label, (width, height), interpolation=cv.INTER_NEAREST)
	ed_label = np.reshape(ed_label, [1, height, width, 1])
	es_label = cv.resize(es_label, (width, height), interpolation=cv.INTER_NEAREST)
	es_label = np.reshape(es_label, [1, height, width, 1])
	return seq_list, n_seq_list, ed_label, es_label


def plot_in_fig(image, label, num_column=4):
	assert num_column % 2 == 0
	num_slices = len(image) * 2
	num_row = (num_slices + num_column - 1) // num_column
	f, plots = plt.subplots(num_row, num_column)
	for i in range(0, num_row * num_column, 2):
		plot_img = plots[i % num_column] if num_row == 1 else plots[i // num_column, i % num_column]
		plot_label = plots[i % num_column + 1] if num_row == 1 else plots[i // num_column, i % num_column + 1]

		if i < num_slices:
			plot_img.imshow(image[i//2, :, :, 0], cmap='gray')
			plot_label.imshow(label[i//2, :, :, 0], cmap='gray')
		else:
			plot_img.axis('off')
			plot_label.axis('off')


if __name__ == "__main__":
	# root = "./CAMUS/"
	# im, _, la = read_mhd(root, "training", "ES", 512, 512)
	# im & la represent the images and labels respectively
	# im with shape [num_img, height, width] (grey scale, range from 0~255)
	# la with shape [num_img, height, width] (range from 0~3)
	# im, _ = read_sequence_mhd(root, "training", 512, 512)
	root = "./CAMUS/val/patient0037/"
	train_path = "./CAMUS/training"
	train_path_list = []
	train_patient_files = os.listdir(train_path)
	for file in train_patient_files:
		path = os.path.join(train_path, file)
		train_path_list.append(path)
	for path in train_path_list:
		seq, ed_label, es_label = read_sequence_with_be(path, "2CH", 512, 512)
	"""
	print(seq.shape)
	print(seq[0].shape)
	print(ed_label.shape)
	print(es_label.shape)
	plt.subplot(2, 2, 1)
	plt.imshow(seq[0, :, :, 0])
	plt.subplot(2, 2, 2)
	plt.imshow(ed_label[0, :, :, 0])
	plt.subplot(2, 2, 3)
	plt.imshow(seq[-1, :, :, 0])
	plt.subplot(2, 2, 4)
	plt.imshow(es_label[0, :, :, 0])
	plt.show()
	# plot_in_fig(im[0:10], im[0:10])
	# plt.show()
	"""
	"""
	assert im.shape[0] == la.shape[0]
	img_placeholder = tf.placeholder(im.dtype, im.shape)
	label_placeholder = tf.placeholder(la.dtype, la.shape)
	dataset = tf.data.Dataset.from_tensor_slices({"image": img_placeholder, "label": label_placeholder}).batch(4).repeat()
	iterator = dataset.make_initializable_iterator()
	data = iterator.get_next()
	
	with tf.Session() as sess:
		sess.run(iterator.initializer, feed_dict={
			img_placeholder: im,
			label_placeholder: la
		})
		check = sess.run(data)
		print(check['image'].shape)
		print(check['label'].shape)
		for i in range(5):
			assemble = sess.run(data)
			if (assemble['image'] == check['image']).all() and (assemble['label'] == check['label']).all():
				print("loop over")
			plot_in_fig(assemble['image'], assemble['label'])
			# plot_in_fig(im[i*4+4:i*4+8], la[i*4+4:i*4+8])
		plt.show()
		"""