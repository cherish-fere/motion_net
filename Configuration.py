import tensorflow as tf
import numpy as np
import TensorflowUtils as utils


def conv_layer(inputs, filters, kernel, stride=1, layer_name="conv"):
	with tf.variable_scope(layer_name):
		regularizer = tf.contrib.layers.l2_regularizer(0.5)
		output = tf.layers.conv2d(
			inputs=inputs, filters=filters, kernel_size=kernel,
			strides=stride, padding='SAME', kernel_regularizer=regularizer)
		return output


def conv_transpose_layer(inputs, filters, kernel, stride=2, layer_name="conv_transpose"):
	with tf.variable_scope(layer_name):
		regularizer = tf.contrib.layers.l2_regularizer(0.5)
		output = tf.layers.conv2d_transpose(
			inputs=inputs, filters=filters, kernel_size=kernel,
			strides=stride, padding='SAME', kernel_regularizer=regularizer)
		return output


def batch_normalization(inputs, is_training, layer_name="BN"):
	with tf.variable_scope(layer_name):
		output = tf.layers.batch_normalization(inputs, training=is_training, trainable=True)
		return output


def relu(x):
	return tf.nn.relu(x)


def max_pooling(x, pool_size=(2, 2), stride=2, padding='SAME'):
	return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def conv_bn_relu(inputs, filters, kernel, layer_name, is_training):
	with tf.variable_scope(layer_name):
		conv = conv_layer(inputs, filters, kernel)
		bn = batch_normalization(conv, is_training)
		outputs = relu(bn)
		return outputs


def conv_transpose_bn_relu(inputs, filters, kernel, layer_name, is_training):
	with tf.variable_scope(layer_name):
		conv_transpose = conv_transpose_layer(inputs, filters, kernel)
		bn = batch_normalization(conv_transpose, is_training)
		outputs = relu(bn)
		return outputs


def attention_gate(features, signal):
	"""
	:param features: features with shape [batch_size, height, width, channels]
	:param signal: signal (in the decoder part) with shape [batch_size, height/2, width/2, channels*2]
	:return: attention features (attention map * features, element wise multiplication)
	"""
	assert (features.shape[1] // 2) == signal.shape[1] and (features.shape[2] // 2) == signal.shape[2]
	assert (features.shape[3] * 2) == signal.shape[3]

	filter_int = features.shape[3] // 2
	with tf.variable_scope("attention_gate"):
		g = conv_layer(inputs=signal, filters=filter_int, kernel=[1, 1], stride=1, layer_name='Wg')
		w = conv_layer(inputs=features, filters=filter_int, kernel=[1, 1], stride=2, layer_name='Wx')
		g_add_w = tf.add(g, w)
		sigma1 = tf.nn.relu(g_add_w)
		theta = conv_layer(inputs=sigma1, filters=1, kernel=[1, 1], stride=1, layer_name='theta')
		# theta = tf.squeeze(theta)
		sigma2 = tf.nn.sigmoid(theta)
		attention_map = tf.image.resize_bilinear(sigma2, [features.shape[1], features.shape[2]])
		attention_features = features * attention_map
		assert attention_features.shape[1:] == features.shape[1:]
		return attention_features


def channel_attention_module(features):
	"""
	:param features: features with shape [batch_size, height, width, channels]
	:return: channel attention with shape [batch_size, 1, 1, channels]
	"""
	shape = features.shape
	in_channel = shape[-1].value
	decay = 8
	assert in_channel % decay == 0
	avg_pool = tf.reduce_mean(features, axis=[1, 2])
	max_pool = tf.reduce_max(features, axis=[1, 2])

	initial_1 = tf.truncated_normal([in_channel, in_channel // decay], stddev=0.02)
	weight_1 = tf.get_variable(name="MLP_1", initializer=initial_1)
	fc_1_avg = tf.matmul(avg_pool, weight_1, name="fc_1_avg")
	fc_1_max = tf.matmul(max_pool, weight_1, name="fc_1_max")

	initial_2 = tf.truncated_normal([in_channel // decay, in_channel], stddev=0.02)
	weight_2 = tf.get_variable(name="MLP_2", initializer=initial_2)
	fc_2_avg = tf.matmul(fc_1_avg, weight_2, name="fc_2_avg")
	fc_2_max = tf.matmul(fc_1_max, weight_2, name="fc_2_max")

	sum_channel_weights = fc_2_avg + fc_2_max
	sum_channel_weights = tf.sigmoid(sum_channel_weights)
	sum_channel_weights = tf.reshape(sum_channel_weights, [shape[0], 1, 1, in_channel])

	return sum_channel_weights


def spatial_attention_module(features):
	"""
	:param features: features with shape [batch_size, height, width, channels]
	:return: spatial attention with shape [batch_size, height, width, 1]
	"""
	avg_pool = tf.reduce_mean(features, axis=[3])
	max_pool = tf.reduce_max(features, axis=[3])
	sum_pool = tf.stack([avg_pool, max_pool], axis=3)

	spatial_conv = tf.layers.conv2d(
		inputs=sum_pool, filters=1, kernel_size=7,
		strides=1, padding='SAME', name="spatial_conv",
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
	spatial_conv = tf.sigmoid(spatial_conv)
	return spatial_conv


def attention_module(features):
	"""
	:param features: features with shape [batch_size, height, width, channels]
	:return: new features with the same shape
	"""
	channel_weights = channel_attention_module(features)
	spatial_weights = spatial_attention_module(features)
	features = features * channel_weights
	features = features * spatial_weights
	return features


def normalize(img):
	mean = np.mean(img)
	var = np.mean(np.square(img - mean))
	img = (img - mean) / np.sqrt(var)
	return img


def calculate_cam(features, weights, labels):
	"""
	:param features: should be with shape [batchsize x height x width x channels]
	:param weights: weights of FC, should be with shape [channels x num_classes]
	:param labels: one_hot labels, should be with shape [batchsize x num_classes]
	:return: CAM map with shape [batchsize x height x width]
	"""

	batchsize = features.shape[0]
	height = features.shape[1]
	width = features.shape[2]
	# initialize CAM_map
	cam_map = np.zeros([batchsize, height, width])

	# select corresponding weights of ground truth
	# one-hot -> figure
	labels = np.argmax(labels, axis=1)  # [batchsize, ]
	for i in range(batchsize):
		f = features[i]
		w = weights[:, labels[i]]
		cam = np.dot(f, w)

		cam = np.maximum(cam, 0)  # passing through relu
		cam = cam / np.max(cam)

		cam_map[i] = cam

	return cam_map


def softmax_loss(logits, labels, is_regularize):
	"""
	:param logits: output of u-net, normally with shape [batch_size, height, width, num_classes]
	:param labels: labels with shape [batch_size, height, width, 1], values of each pixels (0~3) represent a class
	:param is_regularize: bool value
	:return: tensor loss
	"""
	"""
	num_classes = logits.shape[-1]  # channel last
	logits = tf.reshape(logits, [-1, num_classes])  # shape -> [batch_size*height*width, num_classes]
	labels = tf.reshape(labels, [-1, ])  # shape -> [batch_size*height*width, ] (value 0~3, refer to classes)
	# assert logits.shape[0] == labels.shape[0]
	"""
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(labels, axis=3), logits=logits)
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	if is_regularize:
		l2_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		loss = tf.add_n([cross_entropy_mean] + l2_losses)
	else:
		loss = cross_entropy_mean
	return loss


def rec_mse_loss(labels, rec_logits, images, is_regularize):
	"""
	:param labels: normally with shape [batch_size, height, width, 1]
	:param rec_logits: output of reconstruction, normally with shape [batch_size, height, width, num_classes]
	:param images: original images used for training
	:param is_regularize: bool
	:return: loss value in type of float
	"""
	num_classes = rec_logits.shape[-1]
	batch_size = rec_logits.shape[0]
	# seg_index = predict_seg_map(seg_logits)
	seg_image, weights = image_segment(images, labels, num_classes)
	assert seg_image.shape == rec_logits.shape and weights.shape == [rec_logits.shape[0], num_classes]
	losses = 0
	for i in range(num_classes):
		for j in range(batch_size):
			rec = rec_logits[j, :, :, i]
			seg = seg_image[j, :, :, i]
			weight = weights[j, i]
			loss = tf.losses.mean_squared_error(labels=seg, predictions=rec, weights=weight)
			losses += loss

	if is_regularize:
		l2_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		losses = tf.add_n([losses] + l2_losses)

	return losses


def warp_mse_loss(warp_images, target_images, is_regularize):
	"""
	Computing mse loss between warp images and target images
	:param warp_images: warp from source images, with shape [batch_size, height, width, channels]
	:param target_images: target images with shape [batch_size, height, width, channels]
	:param is_regularize: bool
	:return: float
	"""
	losses = tf.losses.mean_squared_error(labels=target_images, predictions=warp_images)

	if is_regularize:
		l2_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		losses = tf.add_n([losses] + l2_losses)

	return losses


def accuracy(logits, labels):
	"""
	:param logits: output of u-net, normally with shape [batch_size, height, width, num_classes]
	:param labels: labels with shape [batch,size, height, width, 1], values of each pixels (0~3) represent a class
	:return: accuracy tensor
	"""
	logits = tf.nn.softmax(logits)
	num_classes = logits.shape[-1]
	logits = tf.reshape(logits, [-1, num_classes])
	labels = tf.reshape(labels, [-1, 1])
	predict = tf.reshape(tf.argmax(logits, axis=1), [-1, 1])
	correct_predict = tf.equal(predict, labels)
	acc = tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))
	return acc


def mean_iou(logits, labels):
	"""
		:param logits: output of u-net, normally with shape [batch_size, height, width, num_classes]
		:param labels: labels with shape [batch,size, height, width, 1], values of each pixels (0~3) represent a class
		:return: result tensor, contains (<mean_iou>, <confusion_matrix>)
	"""
	num_classes = logits.shape[-1]
	logits = tf.nn.softmax(logits)
	logits = tf.reshape(logits, [-1, num_classes])
	labels = tf.reshape(labels, [-1, ])
	predict = tf.reshape(tf.argmax(logits, axis=1), [-1, ])
	result = tf.metrics.mean_iou(labels, predict, num_classes)
	return result


def mean_dice(logits, labels):
	"""
	:param logits: output of u-net, normally with shape [batch_size, height, width, num_classes]
	:param labels: labels with shape [batch,size, height, width, 1], values of each pixels (0~3) represent a class
	:return: result tensor (0~1)
	"""
	smooth = 1e-5
	num_classes = logits.shape[-1]
	logits = tf.nn.softmax(logits)
	logits = tf.reshape(logits, [-1, num_classes])
	labels = tf.reshape(labels, [-1, ])
	predicts = tf.reshape(tf.argmax(logits, axis=1), [-1, ])

	predict_one_hot = tf.one_hot(predicts, depth=num_classes)
	label_one_hot = tf.one_hot(labels, depth=num_classes)

	assert predict_one_hot.shape[1] == num_classes and label_one_hot.shape[1] == num_classes

	dice_sum = []
	for i in range(num_classes):
		predict_sample = predict_one_hot[:, i]
		label_sample = label_one_hot[:, i]
		inse = tf.reduce_sum(predict_sample * label_sample)

		l = tf.reduce_sum(predict_sample * predict_sample)
		r = tf.reduce_sum(label_sample * label_sample)

		dice = (2 * inse + smooth) / (l + r + smooth)
		dice_sum.append(dice)

	return tf.reduce_mean(dice_sum)


def predict_seg_map(logits):
	"""
	:param logits: output of u-net, normally with shape [batch_size, height, width, num_classes]
	:return: segmentation map with shape [batch_size, height, width]
	"""
	scores = tf.nn.softmax(logits)
	segmentation_map = tf.argmax(scores, axis=3)
	return segmentation_map


def image_segment(images, labels, num_classes):
	"""
	This function is used for the reconstruction of images in classes, return the image in each classes as labels
	:param images: original images with shape [batch_size, height, width, 1] (only gray scale)
	:param labels: labels with shape [batch_size, height, width, 1] or [batch_size, height, width]
	:param num_classes: integer
	:return: segmentation of image in shape [batch_size, height, width, num_classes]
	"""
	sparse_labels = tf.one_hot(labels, depth=num_classes)  # [batch_size, height, width, 1, num_classes]
	if sparse_labels.shape[3] == 1:
		sparse_labels = tf.squeeze(sparse_labels, axis=3)   # [batch_size, height, width, num_classes]

	# calculate weights of each classes
	n = int(sparse_labels.shape[1] * sparse_labels.shape[2])
	n = tf.convert_to_tensor(n, dtype=tf.float32)
	weights = tf.reduce_sum(sparse_labels, axis=[1, 2])  # [batch_size, num_classes]
	weights = tf.divide(weights, n)

	sparse_labels = tf.cast(sparse_labels, dtype=tf.float32)
	images = tf.cast(images, dtype=tf.float32)
	assert images.shape[-1] == 1 and sparse_labels.shape[-1] == num_classes and len(sparse_labels.shape) == 4
	seg_images = images * sparse_labels
	return seg_images, weights


