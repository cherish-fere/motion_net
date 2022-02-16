from Configuration import *
from motion_seg_model import motion_net
from load_from_mhd import read_sequence_with_be
from FCN_model import inference
import os
import matplotlib.pyplot as plt


def label_motion(label, flow):
	label_list = [label]
	flow_times = flow.shape[0]
	new_label = tf.cast(label, tf.float32)
	for i in range(flow_times):
		new_label = tf.contrib.image.dense_image_warp(image=new_label, flow=flow[i])
		produce_label = tf.math.round(new_label)
		produce_label = tf.cast(produce_label, dtype=tf.int64)
		label_list.append(produce_label)
	label_batch = tf.concat(label_list, axis=0)
	return label_batch


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
		x = tf.placeholder(dtype=tf.float32, shape=[6, height, width, channels])
		y = tf.placeholder(dtype=tf.int64, shape=[1, height, width, 1])
		source = tf.placeholder(dtype=tf.float32, shape=[5, height, width, channels])
		target = tf.placeholder(dtype=tf.float32, shape=[5, height, width, channels])
		prob = tf.placeholder(dtype=tf.float32)

	with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
		seg_logit = inference(image=x, num_classes=num_classes, keep_prob=prob)

	with tf.variable_scope("motion", reuse=tf.AUTO_REUSE):
		flow_logit, _ = motion_net(target_images=target, source_images=source, training=False)

	label_batch = label_motion(label=y, flow=flow_logit)  # utilize flow to produce label for unlabeled data

	trained_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='seg')
	print(trained_variable)
	
	# separate ground truth from predict label
	b, h, w, c = seg_logit.shape
	seg_logit_first = tf.reshape(seg_logit[0], [1, h, w, c])
	label_first = tf.reshape(label_batch[0], [1, h, w, 1])
	seg_logit_other = seg_logit[1:]
	label_other = label_batch[1:]
	"""
	# Debug label batch
	variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['seg', 'input_container'])
	restore = tf.train.Saver(variables_to_restore)
	pretrained_weight = "./model_weights/motion tanh/model.ckpt-77"

	global_init = tf.global_variables_initializer()
	config = tf.ConfigProto(allow_soft_placement=True)
	# config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:
		sess.run(global_init)
		restore.restore(sess, pretrained_weight)
		path = train_path_list[0]
		sequence, edl, esl = read_sequence_with_be(root=path, structure='2CH', height=height, width=width)
		if phase == "ES":
			sequence = sequence[::-1]
			l = esl
		else:
			l = edl
		assert len(sequence) != 0
		sequence = sequence[0:6]
		feed = {
			source: sequence[0:5],
			target: sequence[1:],
			y: l
		}

		labels, tst = sess.run([label_batch, label_first], feed_dict=feed)
		#for i in range(len(labels)):
			#plt.subplot(1, len(labels), i+1)
			#plt.imshow(labels[i][:, :, 0])

		plt.plot()
		plt.imshow(tst[0, :, :, 0])
		plt.show()
	"""

	main_loss_logit = softmax_loss(logits=seg_logit_first, labels=label_first, is_regularize=False)
	associate_loss_logit = softmax_loss(logits=seg_logit_other, labels=label_other, is_regularize=False)
	loss_logit = 0.8 * main_loss_logit + 0.2 * associate_loss_logit

	optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(loss_logit, var_list=trained_variable)

	acc_logit = accuracy(logits=seg_logit_first, labels=label_first)
	dice_logit = mean_dice(logits=seg_logit_first, labels=label_first)

	loss_summary = tf.summary.scalar('Loss', loss_logit)
	train_log_path = "./training_log"

	variables_to_save = tf.contrib.framework.get_variables_to_restore(include=['seg'])
	saver = tf.train.Saver(variables_to_save)
	save_model_path = "./model_weights"

	variables_to_restore = tf.contrib.framework.get_variables_to_restore(include=['motion'])
	restore = tf.train.Saver(variables_to_restore)
	pretrained_weight = "./model_weights/motion tanh/model.ckpt-77"

	global_init = tf.global_variables_initializer()
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:
		sess.run(global_init)
		restore.restore(sess, pretrained_weight)
		merge = tf.summary.merge([loss_summary])
		writer = tf.summary.FileWriter(train_log_path, sess.graph)
		count = 0
		for epoch in range(epochs):
			step = 0
			for path in train_path_list:
				# ------------------------------------ 2CH ------------------------------------ #
				sequence, _, edl, esl = read_sequence_with_be(root=path, structure='2CH', height=height, width=width)
				if len(sequence) == 0:
					continue
				if phase == "ES":
					sequence = sequence[::-1]
					label = esl
				else:
					label = edl
				sequence = sequence[0:6]

				feed = {source: sequence[0:5], target: sequence[1:], x: sequence, y: label, prob: 0.85}

				sess.run(optimizer, feed_dict=feed)
				t_loss, m_loss, a_loss, acc, dice, summary = sess.run(
					[loss_logit, main_loss_logit, associate_loss_logit, acc_logit, dice_logit, merge], feed_dict=feed)

				print(
					"** Training 2CH ** Epoch %d Step %d Loss: %.2f Main Loss: %.2f Associate Loss: %.2f"
					% (epoch, step, t_loss, m_loss, a_loss))
				print("** Training 2CH ** Epoch %d Step %d Acc: %.2f Dice: %.2f" % (epoch, step, acc, dice))
				writer.add_summary(summary, count)

				# ------------------------------------ 4CH ------------------------------------ #
				sequence, _, edl, esl = read_sequence_with_be(root=path, structure='4CH', height=height, width=width)
				if len(sequence) == 0:
					continue
				if phase == "ES":
					sequence = sequence[::-1]
					label = esl
				else:
					label = edl
				sequence = sequence[0:6]

				feed = {source: sequence[0:5], target: sequence[1:], x: sequence, y: label, prob: 0.85}

				sess.run(optimizer, feed_dict=feed)
				t_loss, m_loss, a_loss, acc, dice, summary = sess.run(
					[loss_logit, main_loss_logit, associate_loss_logit, acc_logit, dice_logit, merge], feed_dict=feed)

				print(
					"** Training 4CH ** Epoch %d Step %d Loss: %.2f Main Loss: %.2f Associate Loss: %.2f"
					% (epoch, step, t_loss, m_loss, a_loss))
				print("** Training 4CH ** Epoch %d Step %d Acc: %.2f Dice: %.2f" % (epoch, step, acc, dice))

				step += 1

			step = 0
			for path in val_path_list:
				# ------------------------------------ 2CH ------------------------------------ #
				sequence, _, edl, esl = read_sequence_with_be(root=path, structure='2CH', height=height, width=width)
				if len(sequence) == 0:
					continue
				if phase == "ES":
					sequence = sequence[::-1]
					label = esl
				else:
					label = edl
				sequence = sequence[0:6]

				feed = {source: sequence[0:5], target: sequence[1:], x: sequence, y: label, prob: 1}

				t_loss, m_loss, a_loss, acc, dice = sess.run(
					[loss_logit, main_loss_logit, associate_loss_logit, acc_logit, dice_logit], feed_dict=feed)

				print(
					"** Training 2CH ** Epoch %d Step %d Loss: %.2f Main Loss: %.2f Associate Loss: %.2f"
					% (epoch, step, t_loss, m_loss, a_loss))
				print("** Training 2CH ** Epoch %d Step %d Acc: %.2f Dice: %.2f" % (epoch, step, acc, dice))

				# ------------------------------------ 4CH ------------------------------------ #
				sequence, _, edl, esl = read_sequence_with_be(root=path, structure='4CH', height=height, width=width)
				if len(sequence) == 0:
					continue
				if phase == "ES":
					sequence = sequence[::-1]
					label = esl
				else:
					label = edl
				sequence = sequence[0:6]

				feed = {source: sequence[0:5], target: sequence[1:], x: sequence, y: label, prob: 1}

				t_loss, m_loss, a_loss, acc, dice = sess.run(
					[loss_logit, main_loss_logit, associate_loss_logit, acc_logit, dice_logit], feed_dict=feed)

				print(
					"** Training 4CH ** Epoch %d Step %d Loss: %.2f Main Loss: %.2f Associate Loss: %.2f"
					% (epoch, step, t_loss, m_loss, a_loss))
				print("** Training 4CH ** Epoch %d Step %d Acc: %.2f Dice: %.2f" % (epoch, step, acc, dice))
				step += 1

			save_model_path = os.path.join(save_model_path, 'model.ckpt')
			saver.save(sess, save_model_path, global_step=epoch)


if __name__ == "__main__":
	train_path = "./CAMUS/training"
	val_path = "./CAMUS/val"
	width = height = 256  # height and width should be 2^n
	channels = 1
	num_classes = 4
	epochs = 15
	lr = 1e-5
	phase = "ES"
	with tf.device('/cpu:0'):
		train(train_path, val_path, width, height, channels, num_classes, epochs, lr, phase)