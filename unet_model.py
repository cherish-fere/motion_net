import tensorflow as tf
from Configuration import *
from load_from_mhd import read_mhd


def unet(x, num_classes, is_training):
	"""
	:param x: input tensor with shape [batch_size, height, width, channels]
	:param num_classes: number of classes
	:param is_training: bool value
	:return: predict map with shape [batch_size, height, width, num_classes]
	"""
	with tf.variable_scope("down_block_1"):
		conv1_1 = conv_bn_relu(inputs=x, filters=64, kernel=[3, 3], layer_name="conv1_1", is_training=is_training)
		conv1_2 = conv_bn_relu(inputs=conv1_1, filters=64, kernel=[3, 3], layer_name="conv1_2", is_training=is_training)
		pool1 = max_pooling(conv1_2)

	with tf.variable_scope("down_block_2"):
		conv2_1 = conv_bn_relu(inputs=pool1, filters=128, kernel=[3, 3], layer_name="conv2_1", is_training=is_training)
		conv2_2 = conv_bn_relu(inputs=conv2_1, filters=128, kernel=[3, 3], layer_name="conv2_2", is_training=is_training)
		pool2 = max_pooling(conv2_2)

	with tf.variable_scope("down_block_3"):
		conv3_1 = conv_bn_relu(inputs=pool2, filters=256, kernel=[3, 3], layer_name="conv3_1", is_training=is_training)
		conv3_2 = conv_bn_relu(inputs=conv3_1, filters=256, kernel=[3, 3], layer_name="conv3_2", is_training=is_training)
		pool3 = max_pooling(conv3_2)

	with tf.variable_scope("down_block_4"):
		conv4_1 = conv_bn_relu(inputs=pool3, filters=512, kernel=[3, 3], layer_name="conv4_1", is_training=is_training)
		conv4_2 = conv_bn_relu(inputs=conv4_1, filters=512, kernel=[3, 3], layer_name="conv4_2", is_training=is_training)
		pool4 = max_pooling(conv4_2)

	with tf.variable_scope("bottom_5"):
		conv5_1 = conv_bn_relu(inputs=pool4, filters=1024, kernel=[3, 3], layer_name="conv5_1", is_training=is_training)
		conv5_2 = conv_bn_relu(inputs=conv5_1, filters=1024, kernel=[3, 3], layer_name="conv5_2", is_training=is_training)
		conv5_transpose = conv_transpose_bn_relu(
			inputs=conv5_2, filters=512, kernel=[2, 2], layer_name="conv_transpose5", is_training=is_training)

	with tf.variable_scope("up_block_6"):
		cat6 = tf.concat([conv4_2, conv5_transpose], axis=3)
		conv6_1 = conv_bn_relu(inputs=cat6, filters=512, kernel=[3, 3], layer_name="conv6_1", is_training=is_training)
		conv6_2 = conv_bn_relu(inputs=conv6_1, filters=512, kernel=[3, 3], layer_name="conv6_2", is_training=is_training)
		conv6_transpose = conv_transpose_bn_relu(
			inputs=conv6_2, filters=256, kernel=[2, 2], layer_name="conv_transpose6", is_training=is_training)

	with tf.variable_scope("up_block_7"):
		cat7 = tf.concat([conv3_2, conv6_transpose], axis=3)
		conv7_1 = conv_bn_relu(inputs=cat7, filters=256, kernel=[3, 3], layer_name="conv7_1", is_training=is_training)
		conv7_2 = conv_bn_relu(inputs=conv7_1, filters=256, kernel=[3, 3], layer_name="conv7_2", is_training=is_training)
		conv7_transpose = conv_transpose_bn_relu(
			inputs=conv7_2, filters=128, kernel=[2, 2], layer_name="conv_transpose7", is_training=is_training)

	with tf.variable_scope("up_block_8"):
		cat8 = tf.concat([conv2_2, conv7_transpose], axis=3)
		conv8_1 = conv_bn_relu(inputs=cat8, filters=128, kernel=[3, 3], layer_name="conv8_1", is_training=is_training)
		conv8_2 = conv_bn_relu(inputs=conv8_1, filters=128, kernel=[3, 3], layer_name="conv8_2", is_training=is_training)
		conv8_transpose = conv_transpose_bn_relu(
			inputs=conv8_2, filters=64, kernel=[2, 2], layer_name="conv_transpose8", is_training=is_training)

	with tf.variable_scope("up_block_9"):
		cat9 = tf.concat([conv1_2, conv8_transpose], axis=3)
		conv9_1 = conv_bn_relu(inputs=cat9, filters=64, kernel=[3, 3], layer_name="conv9_1", is_training=is_training)
		conv9_2 = conv_bn_relu(inputs=conv9_1, filters=64, kernel=[3, 3], layer_name="conv9_2", is_training=is_training)

	with tf.variable_scope("score"):
		scores = conv_layer(inputs=conv9_2, filters=num_classes, kernel=[1, 1], layer_name="conv_scores")

	return scores


def attention_unet(x, num_classes, is_training):
	"""
	:param x: input tensor with shape [batch_size, height, width, channels]
	:param num_classes: number of classes
	:param is_training: bool value
	:return: predict map with shape [batch_size, height, width, num_classes]
	"""
	with tf.variable_scope("down_block_1"):
		conv1_1 = conv_bn_relu(inputs=x, filters=64, kernel=[3, 3], layer_name="conv1_1", is_training=is_training)
		conv1_2 = conv_bn_relu(inputs=conv1_1, filters=64, kernel=[3, 3], layer_name="conv1_2", is_training=is_training)
		pool1 = max_pooling(conv1_2)

	with tf.variable_scope("down_block_2"):
		conv2_1 = conv_bn_relu(inputs=pool1, filters=128, kernel=[3, 3], layer_name="conv2_1", is_training=is_training)
		conv2_2 = conv_bn_relu(inputs=conv2_1, filters=128, kernel=[3, 3], layer_name="conv2_2", is_training=is_training)
		pool2 = max_pooling(conv2_2)

	with tf.variable_scope("down_block_3"):
		conv3_1 = conv_bn_relu(inputs=pool2, filters=256, kernel=[3, 3], layer_name="conv3_1", is_training=is_training)
		conv3_2 = conv_bn_relu(inputs=conv3_1, filters=256, kernel=[3, 3], layer_name="conv3_2", is_training=is_training)
		pool3 = max_pooling(conv3_2)

	with tf.variable_scope("down_block_4"):
		conv4_1 = conv_bn_relu(inputs=pool3, filters=512, kernel=[3, 3], layer_name="conv4_1", is_training=is_training)
		conv4_2 = conv_bn_relu(inputs=conv4_1, filters=512, kernel=[3, 3], layer_name="conv4_2", is_training=is_training)
		pool4 = max_pooling(conv4_2)

	with tf.variable_scope("bottom_5"):
		conv5_1 = conv_bn_relu(inputs=pool4, filters=1024, kernel=[3, 3], layer_name="conv5_1", is_training=is_training)
		conv5_2 = conv_bn_relu(inputs=conv5_1, filters=1024, kernel=[3, 3], layer_name="conv5_2", is_training=is_training)
		conv5_transpose = conv_transpose_bn_relu(
			inputs=conv5_2, filters=512, kernel=[2, 2], layer_name="conv_transpose5", is_training=is_training)

	with tf.variable_scope("up_block_6"):
		attention6 = attention_gate(conv4_2, conv5_2)
		cat6 = tf.concat([attention6, conv5_transpose], axis=3)
		conv6_1 = conv_bn_relu(inputs=cat6, filters=512, kernel=[3, 3], layer_name="conv6_1", is_training=is_training)
		conv6_2 = conv_bn_relu(inputs=conv6_1, filters=512, kernel=[3, 3], layer_name="conv6_2", is_training=is_training)
		conv6_transpose = conv_transpose_bn_relu(
			inputs=conv6_2, filters=256, kernel=[2, 2], layer_name="conv_transpose6", is_training=is_training)

	with tf.variable_scope("up_block_7"):
		attention7 = attention_gate(conv3_2, conv6_2)
		cat7 = tf.concat([attention7, conv6_transpose], axis=3)
		conv7_1 = conv_bn_relu(inputs=cat7, filters=256, kernel=[3, 3], layer_name="conv7_1", is_training=is_training)
		conv7_2 = conv_bn_relu(inputs=conv7_1, filters=256, kernel=[3, 3], layer_name="conv7_2", is_training=is_training)
		conv7_transpose = conv_transpose_bn_relu(
			inputs=conv7_2, filters=128, kernel=[2, 2], layer_name="conv_transpose7", is_training=is_training)

	with tf.variable_scope("up_block_8"):
		attention8 = attention_gate(conv2_2, conv7_2)
		cat8 = tf.concat([attention8, conv7_transpose], axis=3)
		conv8_1 = conv_bn_relu(inputs=cat8, filters=128, kernel=[3, 3], layer_name="conv8_1", is_training=is_training)
		conv8_2 = conv_bn_relu(inputs=conv8_1, filters=128, kernel=[3, 3], layer_name="conv8_2", is_training=is_training)
		conv8_transpose = conv_transpose_bn_relu(
			inputs=conv8_2, filters=64, kernel=[2, 2], layer_name="conv_transpose8", is_training=is_training)

	with tf.variable_scope("up_block_9"):
		cat9 = tf.concat([conv1_2, conv8_transpose], axis=3)
		conv9_1 = conv_bn_relu(inputs=cat9, filters=64, kernel=[3, 3], layer_name="conv9_1", is_training=is_training)
		conv9_2 = conv_bn_relu(inputs=conv9_1, filters=64, kernel=[3, 3], layer_name="conv9_2", is_training=is_training)

	with tf.variable_scope("score"):
		scores = conv_layer(inputs=conv9_2, filters=num_classes, kernel=[1, 1], layer_name="conv_scores")

	return scores


if __name__ == "__main__":
	root = "./CAMUS/"
	im, la = read_mhd(root, "val", "ED", 512, 512)  # only 2^n is allowed
	print(im.shape)
	print(la.shape)
	img = tf.convert_to_tensor(im[0:5, :, :, :], dtype=tf.float32)
	label = tf.convert_to_tensor(la[0:5, :, :, :], dtype=tf.int64)
	logit = attention_unet(img, 4, True)
	print(logit)
	loss = softmax_loss(logits=logit, labels=label, is_regularize=True)
	print(loss)
	acc = accuracy(logits=logit, labels=label)
	print(acc)
	miou_confmat = mean_iou(logits=logit, labels=label)
	print(miou_confmat)