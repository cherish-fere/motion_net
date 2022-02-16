from Configuration import *
import tensorflow as tf
from load_from_mhd import read_sequence_with_be


def vgg16(input, training):
	"""
	:param input: with shape [batch_size, height, width, channel]
	:param training: bool, indicates whether is training
	:return: dictionary contains output of every layer
	"""
	net = {}
	with tf.variable_scope("Conv1"):
		current = conv_bn_relu(inputs=input, filters=64, kernel=[3, 3], layer_name="Conv1_1", is_training=training)
		print("Conv1_1: " + str(current.shape))
		current = conv_bn_relu(inputs=current, filters=64, kernel=[3, 3], layer_name="Conv1_2", is_training=training)
		print("Conv1_2: " + str(current.shape))
		net['Conv1'] = current
		current = max_pooling(current)
		print("Conv1_maxpool: " + str(current.shape))

	with tf.variable_scope("Conv2"):
		current = conv_bn_relu(inputs=current, filters=128, kernel=[3, 3], layer_name="Conv2_1", is_training=training)
		print("Conv2_1: " + str(current.shape))
		current = conv_bn_relu(inputs=current, filters=128, kernel=[3, 3], layer_name="Conv2_2", is_training=training)
		print("Conv2_2: " + str(current.shape))
		net['Conv2'] = current
		current = max_pooling(current)
		print("Conv2_maxpool: " + str(current.shape))

	with tf.variable_scope("Conv3"):
		current = conv_bn_relu(inputs=current, filters=256, kernel=[3, 3], layer_name="Conv3_1", is_training=training)
		print("Conv3_1: " + str(current.shape))
		current = conv_bn_relu(inputs=current, filters=256, kernel=[3, 3], layer_name="Conv3_2", is_training=training)
		print("Conv3_2: " + str(current.shape))
		current = conv_bn_relu(inputs=current, filters=256, kernel=[3, 3], layer_name="Conv3_3", is_training=training)
		print("Conv3_3: " + str(current.shape))
		net['Conv3'] = current
		current = max_pooling(current)
		print("Conv3_maxpool: " + str(current.shape))

	with tf.variable_scope("Conv4"):
		current = conv_bn_relu(inputs=current, filters=512, kernel=[3, 3], layer_name="Conv4_1", is_training=training)
		print("Conv4_1: " + str(current.shape))
		current = conv_bn_relu(inputs=current, filters=512, kernel=[3, 3], layer_name="Conv4_2", is_training=training)
		print("Conv4_2: " + str(current.shape))
		current = conv_bn_relu(inputs=current, filters=512, kernel=[3, 3], layer_name="Conv4_3", is_training=training)
		print("Conv4_3: " + str(current.shape))
		net['Conv4'] = current
		current = max_pooling(current)
		print("Conv4_maxpool: " + str(current.shape))

	with tf.variable_scope("Conv5"):
		current = conv_bn_relu(inputs=current, filters=512, kernel=[3, 3], layer_name="Conv5_1", is_training=training)
		print("Conv5_1: " + str(current.shape))
		current = conv_bn_relu(inputs=current, filters=512, kernel=[3, 3], layer_name="Conv5_2", is_training=training)
		print("Conv5_2: " + str(current.shape))
		current = conv_bn_relu(inputs=current, filters=512, kernel=[3, 3], layer_name="Conv5_3", is_training=training)
		print("Conv5_3: " + str(current.shape))
		net['Conv5'] = current

	return net


def motion_net(target_images, source_images, training):
	"""
	calculating the motion between two conjunction frame in sequence with shape [batch_size, height, width, channel]
	:param target_images: end of one motion, sequence[1:], with shape [batch_size-1, height, width, channels]
	:param source_images: begin of one motion, sequence[0:-1], with shape [batch_size-1, height, width, channels]
	:param training: bool
	:return: a motion features with shape [batch_size-1, height, width, 2], indicates motion in x, y dimension
	"""

	origin_shape = target_images.shape
	origin_width = origin_shape[1]

	with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
		target_features = vgg16(target_images, training)
		source_features = vgg16(source_images, training)

	blocks = ["Conv1", "Conv2", "Conv3", "Conv4", "Conv5"]

	with tf.variable_scope("concat"):
		concats = {}
		for i in blocks:
			target_feature = target_features[i]
			source_feature = source_features[i]
			concat = tf.concat([target_feature, source_feature], axis=3)
			print("Concat " + i + str(concat.shape))
			concats[i] = concat

	with tf.variable_scope("upsampling"):
		upsamples = []
		for i in blocks:
			shape = concats[i].shape
			width = shape[1]
			scale_factor = int(origin_width//width)
			with tf.variable_scope(i):
				if scale_factor == 1:
					# equals to origin input width and height
					upsample = conv_layer(inputs=concats[i], filters=64, kernel=[1, 1], stride=1, layer_name="conv")
				else:
					upsample = conv_transpose_layer(inputs=concats[i], filters=64, kernel=[3, 3], stride=scale_factor, layer_name="conv_t")
				upsample = relu(upsample)
				upsamples.append(upsample)
				print("Upsample " + i + str(upsample.shape))
		fuse = tf.concat(upsamples, axis=3)
		print("Fuse0 " + str(fuse.shape))

	with tf.variable_scope("output"):
		fuse = conv_layer(inputs=fuse, filters=64, kernel=[1, 1], stride=1, layer_name="fuse1")
		print("Fuse1 " + str(fuse.shape))
		flow = conv_layer(inputs=fuse, filters=2, kernel=[1, 1], stride=1, layer_name="fuse2")
		flow = 3 * tf.tanh(flow)
		print("Flow " + str(flow.shape))

	with tf.variable_scope("warp"):
		output = tf.contrib.image.dense_image_warp(image=source_images, flow=flow)
		print("Output" + str(output.shape))

	return flow, output


if __name__ == "__main__":
	root = "./CAMUS/val/patient0423/"
	seq, ed_label, es_label = read_sequence_with_be(root, "2CH", 512, 512)
	x = tf.zeros([10, 512, 512, 1], dtype=tf.float32)
	flow, warp = motion_net(x, x, True)
	loss = warp_mse_loss(warp, x, False)
	print(loss)