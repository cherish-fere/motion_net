import tensorflow as tf
from load_from_mhd import read_mhd, read_mhd_S
from FCN_model import inference
from Configuration import *
import numpy as np
from visualize_mask import visualize

def tst():
	height = width = 256
	channels = 1
	num_classes = 4

	path = "./CAMUS/"
	save_path = './img_mask/'
	val_img, val_n_img, val_label = read_mhd_S(path, "val", "ED", height, width, "4CH")
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

	with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
		logit = inference(image=x, num_classes=num_classes, keep_prob=1)

	seg_logit = predict_seg_map(logits=logit)
	dice_logit = mean_dice(logits=logit, labels=y)

	variables_to_restore = tf.contrib.framework.get_variables_to_restore(include=['seg'])
	restore = tf.train.Saver(variables_to_restore)
	# restore = tf.train.Saver(tf.global_variables())
	pretrained_weight = "./model_weights/motion seg es/model.ckpt-10"
	# pretrained_weight = "./model_weights/fcn/es/model.ckpt-14"

	global_init = tf.global_variables_initializer()
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:
		sess.run(global_init)
		sess.run(val_iterator.initializer, feed_dict={val_img_ph: val_img, val_n_img_ph: val_n_img, val_label_ph: val_label})
		restore.restore(sess, pretrained_weight)
		sum_dice = 0
		for i in range(num_data):
			data = sess.run(val_data)
			image = data["image"].astype(np.int8)
			label = data["label"].astype(np.int8)
			feed = {x: data["n_image"], y: data["label"]}
			seg, dice = sess.run([seg_logit, dice_logit], feed_dict=feed)
			print("Validation data %d  Dice: %.2f" % (i, dice))
			sum_dice += dice
			seg = np.reshape(seg, [height, width])
			image = np.reshape(image, [height, width])
			label = np.reshape(label, [height, width])
			visualize(image, seg, i, save_path)

		print("Mean Dice: %f" % (sum_dice/num_data))


if __name__ == "__main__":
	with tf.device('/cpu:0'):
		tst()