import tensorflow as tf
from Configuration import softmax_loss, accuracy, mean_iou, mean_dice, mse_loss
from FCN.FCN_model import multi_fcn
from load_from_mhd import read_mhd
import os


def train(path, phase, batch_size, height, width, channels, num_classes, epochs, init_lr):
	# --------------get data iterator of training set and validation set-------------- #
	train_img, train_n_img, train_label = read_mhd(path, "training", phase, height, width)
	val_img, val_n_img, val_label = read_mhd(path, "val", phase, height, width)
	# seq_img, seq_n_img = read_sequence_mhd(path, "training", height, width)

	num_train_data = len(train_img)  # used for calculating training steps

	print(train_img.shape)
	print(val_img.shape)
	print(seq_img.shape)

	train_img_ph = tf.placeholder(tf.float32, train_img.shape)
	train_n_img_ph = tf.placeholder(tf.float32, train_n_img.shape)
	train_label_ph = tf.placeholder(tf.int64, train_label.shape)
	val_img_ph = tf.placeholder(tf.float32, val_img.shape)
	val_n_img_ph = tf.placeholder(tf.float32, val_n_img.shape)
	val_label_ph = tf.placeholder(tf.int64, val_label.shape)
	seq_img_ph = tf.placeholder(tf.float32, seq_img.shape)
	seq_n_img_ph = tf.placeholder(tf.float32, seq_n_img.shape)

	train_dataset = tf.data.Dataset.from_tensor_slices(
		{"image": train_img_ph, "n_image": train_n_img_ph, "label": train_label_ph}).batch(batch_size).repeat()
	val_dataset = tf.data.Dataset.from_tensor_slices(
		{"image": val_img_ph, "n_image": val_n_img_ph, "label": val_label_ph}).batch(batch_size).repeat()
	seq_dataset = tf.data.Dataset.from_tensor_slices(
		{"image": seq_img_ph, "n_image": seq_n_img_ph}).batch(batch_size).repeat()

	train_iterator = train_dataset.make_initializable_iterator()
	val_iterator = val_dataset.make_initializable_iterator()
	seq_iterator = seq_dataset.make_initializable_iterator()

	train_data = train_iterator.get_next()
	val_data = val_iterator.get_next()
	seq_data = seq_iterator.get_next()
	# -------------------------------------------------------------------------------- #

	with tf.variable_scope("input_container"):
		xl = tf.placeholder(dtype=tf.float32, shape=[batch_size, height, width, channels])
		# xu = tf.placeholder(dtype=tf.float32, shape=[batch_size, height, width, channels])
		y = tf.placeholder(dtype=tf.int64, shape=[batch_size, height, width, 1])  # non-one-hot
		prob = tf.placeholder(tf.float32)

	with tf.variable_scope("", reuse=tf.AUTO_REUSE):
		seg_logit, rec_logit, seg_var, rec_var = multi_fcn(xl, num_classes, prob)
		# rec_seg_logit, rec_logit, _, rec_var = multi_fcn(xu, num_classes, prob)

	# segmentation loss
	seg_loss_logit = softmax_loss(logits=seg_logit, labels=y, is_regularize=False)
	# reconstruction loss
	rec_loss_logit = mse_loss(labels=y, rec_logits=rec_logit, images=xl, is_regularize=False)
	alpha = 0.5
	sum_loss_logit = (1 - alpha) * seg_loss_logit + alpha * rec_loss_logit

	acc_logit = accuracy(logits=seg_logit, labels=y)
	miou_logit, miou_updates = mean_iou(logits=seg_logit, labels=y)
	dice_logit = mean_dice(logits=seg_logit, labels=y)

	optimizer = tf.train.AdamOptimizer(learning_rate=init_lr, beta1=0.5).minimize(sum_loss_logit)

	seg_loss_summary = tf.summary.scalar('Loss_Seg', seg_loss_logit)
	rec_loss_summary = tf.summary.scalar('Loss_Rec', rec_loss_logit)
	acc_summary = tf.summary.scalar('ACC', acc_logit)
	miou_summary = tf.summary.scalar('M-IoU', miou_logit)
	dice_summary = tf.summary.scalar('Dice', dice_logit)

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
		sess.run(seq_iterator.initializer, feed_dict={seq_img_ph: seq_img, seq_n_img_ph: seq_n_img})
		sess.run(global_init)

		# -------------------------------log------------------------------- #
		merge = tf.summary.merge([seg_loss_summary, rec_loss_summary, acc_summary, miou_summary, dice_summary])
		train_writer = tf.summary.FileWriter(train_log_path, sess.graph)
		val_writer = tf.summary.FileWriter(val_log_path, sess.graph)

		assert num_train_data % batch_size == 0  # whether can program overview data set in an epoch
		train_steps = num_train_data // batch_size

		for epoch in range(epochs):

			for step in range(train_steps):
				# -------------------------Segmentation Part------------------------- #
				train_data_batch = sess.run(train_data)  # contains images and labels
				seq_data_batch = sess.run(seq_data)
				feed = {
					xl: train_data_batch['n_image'],
					xu: seq_data_batch['n_image'],
					y: train_data_batch['label'],
					prob: 0.85}
				sess.run(local_init)

				_, _ = sess.run([optimizer, miou_updates], feed_dict=feed)
				seg_loss, rec_loss, acc, miou, dice, summary = sess.run(
					[seg_loss_logit, rec_loss_logit, acc_logit, miou_logit, dice_logit, merge], feed_dict=feed)

				print(
					"** Training Seg Part ** Epoch: %d  Step: %d  Seg_Loss: %.4f  Rec_Loss: %.4f  Acc: %.2f  M-IOU: %.2f  Dice: %.2f"
					% (epoch + 1, step + 1, seg_loss, rec_loss, acc, miou, dice)
				)

				train_writer.add_summary(summary, epoch * train_steps + step + 1)

				# -------------------------Validation Part------------------------- #
				val_data_batch = sess.run(val_data)
				feed = {
					xl: val_data_batch['n_image'],
					xu: val_data_batch['n_image'],
					y: val_data_batch['label'],
					prob: 1
				}
				sess.run(local_init)

				_ = sess.run(miou_updates, feed_dict=feed)
				seg_loss, rec_loss, acc, miou, dice, summary = sess.run(
					[seg_loss_logit, rec_loss_logit, acc_logit, miou_logit, dice_logit, merge], feed_dict=feed)

				print(
					"**       Val        ** Epoch: %d  Step: %d  Seg_Loss: %.4f  Rec_Loss: %.4f  Acc: %.2f  M-IOU: %.2f  Dice: %.2f"
					% (epoch + 1, step + 1, seg_loss, rec_loss, acc, miou, dice)
				)

				val_writer.add_summary(summary, epoch * train_steps + step + 1)

			save_model_path = os.path.join(model_path, 'model.ckpt')
			saver.save(sess, save_model_path, global_step=epoch)


if __name__ == '__main__':
	path = "./CAMUS/"
	phase = "ED"
	batch_size = 1
	width = height = 512  # height and width should be 2^n
	channels = 1
	num_classes = 4
	epochs = 15
	init_lr = 1e-5
	with tf.device('/cpu:0'):
		train(path, phase, batch_size, height, width, channels, num_classes, epochs, init_lr)
