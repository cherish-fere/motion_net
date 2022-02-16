from Configuration import *
from motion_seg_model import motion_net
from load_from_mhd import read_sequence_with_be
import os
import numpy as np


def train(train_path, val_path, width, height, channels, num_classes, epochs, lr, phase):
	assert phase == "ED" or phase == "ES"
	train_path_list = []
	train_patient_files = os.listdir(train_path)
	for file in train_patient_files:
		path = os.path.join(train_path, file)
		train_path_list.append(path)

	val_path_list = []
	val_patient_files = os.listdir(val_path)
	for file in val_patient_files:
		path = os.path.join(val_path, file)
		val_path_list.append(path)

	with tf.variable_scope("input_container"):
		# target: end of one motion; source: begin of one motion
		source = tf.placeholder(dtype=tf.float32, shape=[5, height, width, channels])
		target = tf.placeholder(dtype=tf.float32, shape=[5, height, width, channels])
		training = tf.placeholder(dtype=tf.bool)

	with tf.variable_scope("motion", reuse=tf.AUTO_REUSE):
		flow_logit, warp_logit = motion_net(target_images=target, source_images=source, training=training)

	loss_logit = warp_mse_loss(warp_images=warp_logit, target_images=target, is_regularize=False)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(loss_logit)

	loss_summary = tf.summary.scalar('Loss', loss_logit)
	saver = tf.train.Saver(tf.global_variables())
	global_init = tf.global_variables_initializer()
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True

	model_path = "./model_weights"
	train_log_path = "./training_log"

	with tf.Session(config=config) as sess:
		sess.run(global_init)
		merge = tf.summary.merge([loss_summary])
		ed_writer = tf.summary.FileWriter(train_log_path, sess.graph)
		es_writer = tf.summary.FileWriter(train_log_path, sess.graph)
		count = 0
		for epoch in range(epochs):
			step = 0
			for path in train_path_list:
				sequence_2ch, _, _, _ = read_sequence_with_be(root=path, structure='2CH', height=height, width=width)
				sequence_4ch, _, _, _ = read_sequence_with_be(root=path, structure='4CH', height=height, width=width)

				if len(sequence_2ch) == 0 or len(sequence_4ch) == 0:
					continue

				if phase == "ES":
					sequence_2ch = sequence_2ch[::-1]
					sequence_4ch = sequence_4ch[::-1]

				sequence_2ch = sequence_2ch[0:6]
				sequence_4ch = sequence_4ch[0:6]

				# ------------------------- 2ch ------------------------- #
				# if ed, source is [0:5] target is [1:]
				feed_dict = {
					source: sequence_2ch[0:5],
					target: sequence_2ch[1:],
					training: True
				}
				_, loss, summary = sess.run([optimizer, loss_logit, merge], feed_dict=feed_dict)
				print("** Training 2CH ** : Epoch: %d Step: %d Loss: %f" % (epoch, step, loss))
				ed_writer.add_summary(summary, count)
				count += 1

				# ------------------------- 4ch ------------------------- #
				feed_dict = {
					source: sequence_4ch[0:5],
					target: sequence_4ch[1:],
					training: True
				}
				_, loss, summary = sess.run([optimizer, loss_logit, merge], feed_dict=feed_dict)
				print("** Training 4CH ** : Epoch: %d Step: %d Loss: %f" % (epoch, step, loss))
				es_writer.add_summary(summary, count)
				count += 1
				step += 1

			step = 0
			for path in val_path_list:
				sequence_2ch, _, _, _ = read_sequence_with_be(root=path, structure='2CH', height=height, width=width)
				sequence_4ch, _, _, _ = read_sequence_with_be(root=path, structure='4CH', height=height, width=width)

				if len(sequence_2ch) == 0 or len(sequence_4ch) == 0:
					continue

				if phase == "ES":
					sequence_2ch = sequence_2ch[::-1]
					sequence_4ch = sequence_4ch[::-1]

				sequence_2ch = sequence_2ch[0:6]
				sequence_4ch = sequence_4ch[0:6]

				# ------------------------- 2ch ------------------------- #
				feed_dict = {
					source: sequence_2ch[0:5],
					target: sequence_2ch[1:],
					training: False
				}
				loss = sess.run(loss_logit, feed_dict=feed_dict)
				print("**   Val    2CH ** : Epoch: %d Step: %d Loss: %f" % (epoch, step, loss))

				# ------------------------- 4ch ------------------------- #
				feed_dict = {
					source: sequence_4ch[0:5],
					target: sequence_4ch[1:],
					training: False
				}
				loss = sess.run(loss_logit, feed_dict=feed_dict)
				print("**   Val    4CH ** : Epoch: %d Step: %d Loss: %f" % (epoch, step, loss))
				step += 1

			save_model_path = os.path.join(model_path, 'model.ckpt')
			saver.save(sess, save_model_path, global_step=epoch)


if __name__ == "__main__":
	train_path = "./CAMUS/training"
	val_path = "./CAMUS/val"
	width = height = 256  # height and width should be 2^n
	channels = 1
	num_classes = 4
	epochs = 100
	lr = 1e-5
	phase = "ED"
	with tf.device('/gpu:0'):
		train(train_path, val_path, width, height, channels, num_classes, epochs, lr, phase)