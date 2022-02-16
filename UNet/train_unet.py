import tensorflow as tf
from Configuration import softmax_loss, accuracy, mean_iou, predict_seg_map, mean_dice
from unet_model import unet, attention_unet
from load_from_mhd import read_mhd
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os


def train(path, phase, batch_size, height, width, channels, num_classes, epoches, init_lr):

	# --------------get data iterator of training set and validation set-------------- #
	train_img, train_n_img, train_label = read_mhd(path, "training", phase, height, width)
	val_img, val_n_img, val_label = read_mhd(path, "val", phase, height, width)

	num_train_data = len(train_img)  # used for calculating training steps
	num_val_data = len(val_img)

	train_img_ph = tf.placeholder(tf.float32, train_img.shape)
	train_n_img_ph = tf.placeholder(tf.float32, train_n_img.shape)
	train_label_ph = tf.placeholder(tf.int64, train_label.shape)
	val_img_ph = tf.placeholder(tf.float32, val_img.shape)
	val_n_img_ph = tf.placeholder(tf.float32, val_n_img.shape)
	val_label_ph = tf.placeholder(tf.int64, val_label.shape)

	train_dataset = tf.data.Dataset.from_tensor_slices(
		{"image": train_img_ph, "n_image": train_n_img_ph, "label": train_label_ph}).batch(batch_size).repeat()
	val_dataset = tf.data.Dataset.from_tensor_slices(
		{"image": val_img_ph, "n_image": val_n_img_ph, "label": val_label_ph}).batch(batch_size).repeat()

	train_iterator = train_dataset.make_initializable_iterator()
	val_iterator = val_dataset.make_initializable_iterator()

	train_data = train_iterator.get_next()
	val_data = val_iterator.get_next()
	# -------------------------------------------------------------------------------- #

	with tf.variable_scope("input_container"):
		x = tf.placeholder(dtype=tf.float32, shape=[None, height, width, channels])
		y = tf.placeholder(dtype=tf.int64, shape=[None, height, width, 1])  # non-one-hot
		is_training = tf.placeholder(dtype=tf.bool)
		lr = tf.placeholder(dtype=tf.float32)

	logit = attention_unet(x=x, num_classes=num_classes, is_training= is_training)
	loss_logit = softmax_loss(logits=logit, labels=y, is_regularize=True)
	acc_logit = accuracy(logits=logit, labels=y)
	miou_logit, miou_updates = mean_iou(logits=logit, labels=y)
	dice_logit = mean_dice(logits=logit, labels=y)
	seg_logit = predict_seg_map(logits=logit)
	print(seg_logit)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(loss_logit)

	loss_summary = tf.summary.scalar('Loss', loss_logit)
	acc_summary = tf.summary.scalar('ACC', acc_logit)
	miou_summary = tf.summary.scalar('M-IoU', miou_logit)

	saver = tf.train.Saver(tf.global_variables())
	global_init = tf.global_variables_initializer()
	local_init = tf.local_variables_initializer()
	config = tf.ConfigProto(allow_soft_placement=True)

	model_path = "./model_weights"
	train_log_path = "./training_log"
	val_log_path = "./val_log"

	with tf.Session(config=config) as sess:
		sess.run(train_iterator.initializer, feed_dict={train_img_ph: train_img, train_n_img_ph: train_n_img, train_label_ph: train_label})
		sess.run(val_iterator.initializer, feed_dict={val_img_ph: val_img, val_n_img_ph: val_n_img, val_label_ph: val_label})
		sess.run(global_init)

		# -------------------------------log------------------------------- #
		merge = tf.summary.merge([loss_summary, acc_summary, miou_summary])
		train_writer = tf.summary.FileWriter(train_log_path, sess.graph)
		val_writer = tf.summary.FileWriter(val_log_path, sess.graph)

		assert num_train_data % batch_size == 0  # whether can program overview data set in an epoch
		assert num_val_data % batch_size == 0
		train_steps = num_train_data // batch_size
		val_steps = num_val_data // batch_size

		for epoch in range(epoches):

			if epoch < 5:
				learning_rate = init_lr * (10 ** -epoch)
			else:
				learning_rate = init_lr * (10 ** -5)

			for step in range(train_steps):
				train_data_batch = sess.run(train_data)  # contains images and labels
				feed = {
					x: train_data_batch['n_image'],
					y: train_data_batch['label'],
					is_training: True,
					lr: learning_rate
				}
				sess.run(local_init)

				_, _ = sess.run([optimizer, miou_updates], feed_dict=feed)
				loss, acc, miou, dice, summary = sess.run([loss_logit, acc_logit, miou_logit, dice_logit, merge], feed_dict=feed)

				print(
					"** Training ** Epoch: %d  Step: %d  Loss: %.4f  Acc: %.2f  M-IOU: %.2f  Dice: %.2f"
					% (epoch + 1, step + 1, loss, acc, miou, dice)
				)

				train_writer.add_summary(summary, epoch * train_steps + step + 1)
			
				val_data_batch = sess.run(val_data)
				feed = {
					x: val_data_batch['n_image'],
					y: val_data_batch['label'],
					is_training: False
				}
				sess.run(local_init)

				_ = sess.run(miou_updates, feed_dict=feed)
				loss, acc, miou, dice, summary = sess.run([loss_logit, acc_logit, miou_logit, dice_logit, merge], feed_dict=feed)

				print(
					"**    Val   ** Epoch: %d  Step: %d  Loss: %.4f  Acc: %.2f  M-IOU: %.2f  Dice: %.2f"
					% (epoch + 1, step + 1, loss, acc, miou, dice)
				)

				val_writer.add_summary(summary, epoch * train_steps + step + 1)

			save_model_path = os.path.join(model_path, 'model.ckpt')
			saver.save(sess, save_model_path, global_step=epoch)

		count = 1
		for step in range(val_steps):
			# produce segmentation map
			val_data_batch = sess.run(val_data)
			feed = {x: val_data_batch['n_image'], y: val_data_batch['label'], is_training: False}
			seg_batches = sess.run(seg_logit, feed_dict=feed)
			nums = val_data_batch['image'].shape[0]
			for i in range(nums):
				image = val_data_batch['image'][i, :, :, 0]
				label = val_data_batch['label'][i, :, :, 0]
				seg = seg_batches[i]
				img_file_name = "./produce/image/image" + str(count) + ".csv"
				label_file_name = "./produce/label/label" + str(count) + ".csv"
				seg_file_name = "./produce/segmentation/seg" + str(count) + ".csv"
				np.savetxt(img_file_name, image, delimiter=',')
				np.savetxt(label_file_name, label, delimiter=',')
				np.savetxt(seg_file_name, seg, delimiter=',')
				count += 1


if __name__ == '__main__':
	path = "./CAMUS/"
	phase = "ED"
	batch_size = 1
	width = height = 512  # height and width should be 2^n
	channels = 1
	num_classes = 4
	epoches = 35
	init_lr = 1e-4
	with tf.device('/cpu:0'):
		train(path, phase, batch_size, height, width, channels, num_classes, epoches, init_lr)
