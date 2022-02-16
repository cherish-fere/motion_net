import tensorflow as tf
from Configuration import softmax_loss, accuracy, mean_iou, predict_seg_map, mean_dice
from FCN_model import inference, inference_attention
from unet_model import attention_unet, unet
from load_from_mhd import read_mhd
import cv2
from scipy.spatial.distance import directed_hausdorff
import numpy as np


def m_test(path, phase, height, width, channels, num_classes):
	val_img, val_n_img, val_label = read_mhd(path, "val", phase, height, width)
	num_data = len(val_img)

	print(val_img.shape)
	print(val_label.shape)

	val_img_ph = tf.placeholder(tf.float32, val_img.shape)
	val_n_img_ph = tf.placeholder(tf.float32, val_n_img.shape)
	val_label_ph = tf.placeholder(tf.int64, val_label.shape)

	val_dataset = tf.data.Dataset.from_tensor_slices(
		{"image": val_img_ph, "n_image": val_n_img_ph, "label": val_label_ph}).batch(1).repeat(1)
	val_iterator = val_dataset.make_initializable_iterator()
	val_data = val_iterator.get_next()

	with tf.variable_scope("input_container"):
		x = tf.placeholder(dtype=tf.float32, shape=[1, height, width, channels])
		y = tf.placeholder(dtype=tf.int64, shape=[1, height, width, 1])  # non-one-hot
		prob = tf.placeholder(tf.float32)
	with tf.variable_scope("", reuse=tf.AUTO_REUSE):
		logit = inference_attention(x, num_classes, prob)
	# logit = attention_unet(x, num_classes, False)

	loss_logit = softmax_loss(logits=logit, labels=y, is_regularize=False)
	acc_logit = accuracy(logits=logit, labels=y)
	miou_logit, miou_updates = mean_iou(logits=logit, labels=y)
	dice_logit = mean_dice(logits=logit, labels=y)

	predict_logit = predict_seg_map(logits=logit)
	sparse_predict = tf.one_hot(predict_logit, depth=num_classes)
	label = tf.squeeze(y, axis=3)
	sparse_label = tf.one_hot(label, depth=num_classes)
	assert sparse_predict.shape == sparse_label.shape

	# trained model path
	trained_model_path = "./model_weights/fcn-a ed/model.ckpt-14"

	saver = tf.train.Saver(tf.global_variables())
	global_init = tf.global_variables_initializer()
	local_init = tf.local_variables_initializer()

	config = tf.ConfigProto(allow_soft_placement=True)

	with tf.Session(config=config) as sess:
		sess.run(val_iterator.initializer, feed_dict={val_img_ph: val_img, val_n_img_ph: val_n_img, val_label_ph: val_label})
		sess.run(global_init)
		saver.restore(sess, trained_model_path)
		print('Model Load Successful')
		dice_sum = 0
		haus_sum = 0
		acc_sum = 0
		for i in range(num_data):
			val_data_batch = sess.run(val_data)
			feed = {x: val_data_batch['n_image'], y: val_data_batch['label'], prob: 1}
			sess.run(local_init)
			_ = sess.run(miou_updates, feed_dict=feed)
			loss, acc, miou, dice, seg, lp, lt = sess.run([loss_logit, acc_logit, miou_logit, dice_logit, predict_logit, sparse_predict, sparse_label], feed_dict=feed)
			print("**    Val   ** Loss: %.4f  Acc: %.2f  M-IOU: %.2f  Dice: %.2f" % (loss, acc, miou, dice))

			sub_haus = 0
			count = 0
			for c in range(1, num_classes):
				predict = lp[0, :, :, c].astype(np.uint8)
				gt = lt[0, :, :, c].astype(np.uint8)
				predict_contours = cv2.findContours(predict, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_KCOS)[1]
				gt_contours = cv2.findContours(gt, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_KCOS)[1]
				if len(predict_contours) == 0:
					print("empty")
					continue
				u = predict_contours[0]
				for i in range(len(predict_contours)):
					if len(predict_contours[i]) > len(u):
						u = predict_contours[i]
				v = gt_contours[0]
				for i in range(len(gt_contours)):
					if len(gt_contours[i]) > len(v):
						v = gt_contours[i]
				u = np.reshape(u, [-1, 2])
				v = np.reshape(v, [-1, 2])
				haus1 = directed_hausdorff(u, v)[0]
				haus2 = directed_hausdorff(v, u)[0]
				haus = max(haus1, haus2)
				sub_haus += haus
				count += 1

			sub_haus = sub_haus / count
			dice_sum += dice
			haus_sum += sub_haus
			acc_sum += acc

		dice_mean = dice_sum / num_data
		haus_mean = haus_sum / num_data
		acc_sum = acc_sum / num_data
		print("Dice scores: " + str(dice_mean))
		print("Hausdorff distance: " + str(haus_mean))
		print("Accuracy: " + str(acc_sum))


if __name__ == "__main__":
	path = "./CAMUS/"
	phase = "ED"
	width = height = 256  # height and width should be 2^n
	channels = 1
	num_classes = 4
	with tf.device('/gpu:0'):
		m_test(path, phase, height, width, channels, num_classes)
