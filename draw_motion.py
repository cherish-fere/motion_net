import tensorflow as tf
from motion_seg_model import motion_net
from Configuration import warp_mse_loss
from load_from_mhd import read_sequence_with_be
from matplotlib import pyplot as plt
import cv2
import imageio
import numpy as np


def visualize(img, mask):
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

	return result


with tf.variable_scope("input_container"):
	# target: end of one motion; source: begin of one motion
	source = tf.placeholder(dtype=tf.float32, shape=[5, 256, 256, 1])
	target = tf.placeholder(dtype=tf.float32, shape=[5, 256, 256, 1])
	training = tf.placeholder(dtype=tf.bool)

with tf.variable_scope("motion", reuse=tf.AUTO_REUSE):
	flow_logit, warp_logit = motion_net(target_images=target, source_images=source, training=training)

loss_logit = warp_mse_loss(warp_images=warp_logit, target_images=target, is_regularize=False)

trained_model_path = "./model_weights/motion ed/model.ckpt-70"

saver = tf.train.Saver(tf.global_variables())
global_init = tf.global_variables_initializer()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session() as sess:
	sess.run(global_init)
	saver.restore(sess, trained_model_path)
	path = "./CAMUS/training/patient0004/"
	sequence, _, edl, eds = read_sequence_with_be(root=path, structure='2CH', height=256, width=256)
	sequence_ed = sequence[0:6]
	feed_dict = {source: sequence_ed[0:5], target: sequence_ed[1:], training: False}
	flow, warp, loss = sess.run([flow_logit, warp_logit, loss_logit], feed_dict=feed_dict)
	print("Loss: %.4f" % loss)

	# image part
	fig = plt.figure('momtion')
	for i in range(5):
		plt.subplot(4, 5, i+1)
		plt.imshow(sequence_ed[0:5][i, :, :, 0], cmap='gray')
		plt.title("Source Images %d" % i)
		plt.subplot(4, 5, 5 + i + 1)
		plt.imshow(warp[i, :, :, 0], cmap='gray')
		plt.title("Warp Images %d" % i)
		plt.subplot(4, 5, 10 + i + 1)
		plt.imshow(sequence_ed[1:][i, :, :, 0], cmap='gray')
		plt.title("Target Images %d" % i)
		plt.subplot(4, 5, 15 + i + 1)
		plt.imshow(sequence_ed[1:][i, :, :, 0] - warp[i, :, :, 0], cmap='gray')
		plt.title("Difference %d" % i)
	fig.tight_layout()
	fig.subplots_adjust(wspace=0.5, hspace=0.5)

	# label part
	plt.figure("label")
	produce_label = tf.convert_to_tensor(edl, dtype=tf.float32)
	plt.subplot(2, 6, 1)
	plt.imshow(edl[0, :, :, 0])
	plt.title("warp label 0")
	mask = visualize(sequence_ed[0, :, :, 0], edl[0, :, :, 0])
	plt.subplot(2, 6, 7)
	plt.imshow(mask)
	plt.title("mask 0")
	pls = []
	for i in range(5):
		produce_label = tf.contrib.image.dense_image_warp(image=produce_label, flow=flow[i])
		pl = sess.run(produce_label)
		pl = np.round(pl)
		pl = pl.astype(np.int16)
		pl = pl[0, :, :, 0]
		pls.append(pl)
		mask = visualize(sequence_ed[i+1, :, :, 0], pl)

		plt.subplot(2, 6, i+2)
		plt.imshow(pl)
		plt.title("warp label %d" % (i+1))
		plt.subplot(2, 6, 6+i+2)
		plt.imshow(mask)
		plt.title("mask %d" % (i+1))

	plt.show()
	print((warp == sequence_ed[1:]).all())
	print((sequence_ed[0:5][1:] == sequence_ed[1:][:-1]).all())

	warps = []
	targets = []
	for i in range(5):
		img = np.round(warp[i]).astype(np.uint8)
		warps.append(img)
		targets.append(sequence_ed[1:][i])

	imageio.mimsave('warp.gif', warps, 'GIF', duration=0.2)
	imageio.mimsave('target.gif', targets, 'GIF', duration=0.2)
	imageio.mimsave('pl.gif', pls, 'GIF', duration=0.2)