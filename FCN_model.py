import TensorflowUtils as utils
import numpy as np
import tensorflow as tf
from load_from_mhd import read_mhd
from Configuration import attention_module


def vgg_net(weights, image):
	# reconstruct VGG-19 without fc layers
	layers = (
		'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2',
		'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4',
		'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4',
		'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
	)

	net = {}
	current = image
	for i, name in enumerate(layers):
		kind = name[:4]
		if kind == 'conv':
			kernels, bias = weights[i][0][0][0][0]
			# matconvnet: weights are [width, height, in_channels, out_channels]
			# tensorflow: weights are [height, width, in_channels, out_channels]
			kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
			bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
			current = utils.conv2d_basic(current, kernels, bias)
			print("当前形状：", np.shape(current))
		elif kind == 'relu':
			current = tf.nn.relu(current, name=name)
		elif kind == 'pool':
			current = utils.avg_pool_2x2(current)
			print("当前形状：", np.shape(current))
		net[name] = current
	return net


def vgg_net_attention(weights, image):
	# reconstruct VGG-19 without fc layers
	layers = (
		'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2',
		'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4',
		'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4',
		'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
	)

	net = {}
	current = image
	for i, name in enumerate(layers):
		kind = name[:4]
		if kind == 'conv':
			kernels, bias = weights[i][0][0][0][0]
			# matconvnet: weights are [width, height, in_channels, out_channels]
			# tensorflow: weights are [height, width, in_channels, out_channels]
			kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
			bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
			current = utils.conv2d_basic(current, kernels, bias)
			print("Conv：", np.shape(current))
			net[name] = current
		elif kind == 'relu':
			current = tf.nn.relu(current, name=name)
			net[name] = current
		elif kind == 'pool':
			scope_name = "Conv_Attention_" + name
			with tf.variable_scope(scope_name):
				current = attention_module(current)
				print("Attention：", np.shape(current))
				net[scope_name] = current
			current = utils.avg_pool_2x2(current)
			print("AvgPool：", np.shape(current))
			net[name] = current

	return net


# FCN的网络结构定义，网络中用到的参数是迁移VGG训练好的参数
def inference(image, num_classes, keep_prob):
	"""
	Semantic segmentation network definition
	:param image: input image. Should have values in range 0-255
	:param num_classes: number of classes
	:param keep_prob: dropout rate
	:return:
	"""
	image = tf.image.grayscale_to_rgb(image)
	print(image.shape)
	# load weights
	model_data = utils.get_model_data('D:/university/python/cardiac_seg/imagenet-vgg-verydeep-19.mat')
	weights = np.squeeze(model_data['layers'])

	"""
	# image preprocessor
	mean = model_data['normalization'][0][0][0]
	mean_pixel = np.mean(mean, axis=(0, 1))
	processed_image = utils.process_image(image, mean_pixel)
	print("预处理后的图像:", np.shape(processed_image))
	"""

	with tf.variable_scope("inference"):
		# establish original VGG-19

		print("Construct VGG-19：")
		image_net = vgg_net(weights, image)

		# add 1 pooling layer and 3 conv layer after VGG-19
		conv_final_layer = image_net["conv5_3"]
		print("features from VGG：", np.shape(conv_final_layer))

		pool5 = utils.max_pool_2x2(conv_final_layer)

		print("pool5：", np.shape(pool5))

		W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
		b6 = utils.bias_variable([4096], name="b6")
		conv6 = utils.conv2d_basic(pool5, W6, b6)
		relu6 = tf.nn.relu(conv6, name="relu6")
		relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

		print("conv6:", np.shape(relu_dropout6))

		W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
		b7 = utils.bias_variable([4096], name="b7")
		conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
		relu7 = tf.nn.relu(conv7, name="relu7")
		relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

		print("conv7:", np.shape(relu_dropout7))

		W8 = utils.weight_variable([1, 1, 4096, num_classes], name="W8")
		b8 = utils.bias_variable([num_classes], name="b8")
		conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)

		print("conv8:", np.shape(conv8))
		# annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

		# upsampling (deconv) part
		# pool4 + 2 x conv8
		deconv_shape1 = image_net["pool4"].get_shape()
		W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, num_classes], name="W_t1")
		b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
		conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
		fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

		print("pool4 and de_conv8 ==> fuse1:", np.shape(fuse_1))

		# pool3 + 2 x fuse_11
		deconv_shape2 = image_net["pool3"].get_shape()
		W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
		b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
		conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
		fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

		print("pool3 and deconv_fuse1 ==> fuse2:", np.shape(fuse_2))

		shape = tf.shape(image)
		deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], num_classes])
		W_t3 = utils.weight_variable([16, 16, num_classes, deconv_shape2[3].value], name="W_t3")
		b_t3 = utils.bias_variable([num_classes], name="b_t3")
		conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

		print("conv_t3:", np.shape(conv_t3))
	
	return conv_t3


def multi_fcn(image, num_classes, keep_prob):
	"""
	A multitask FCN model in which an extra 8x upsampling block that is identical to FCN was add to reconstruct unlabeled
	images in each classes.
	:param image: input images in shape[batch_size, height, width, channels]
	:param num_classes: number of classes, integer number
	:param keep_prob: probability of dropout layer
	:return: prediction scores in shape [batch_size, height, width, num_classes]
	"""
	image = tf.image.grayscale_to_rgb(image)
	print(image.shape)
	# load weights
	model_data = utils.get_model_data('./imagenet-vgg-verydeep-19.mat')
	weights = np.squeeze(model_data['layers'])

	with tf.variable_scope("encoder"):
		# construct original VGG-19
		print("Construct VGG-19：")
		vgg = vgg_net(weights, image)

		conv_final_layer = vgg["conv5_3"]
		print("features from VGG：", np.shape(conv_final_layer))
		# add 1 pooling layer and 3 conv layer after VGG-19
		pool5 = utils.max_pool_2x2(conv_final_layer)
		print("pool5:", np.shape(pool5))

	with tf.variable_scope("seg_decoder"):
		W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
		b6 = utils.bias_variable([4096], name="b6")
		conv6 = utils.conv2d_basic(pool5, W6, b6)
		relu6 = tf.nn.relu(conv6, name="relu6")
		relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

		print("conv6:", np.shape(relu_dropout6))

		W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
		b7 = utils.bias_variable([4096], name="b7")
		conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
		relu7 = tf.nn.relu(conv7, name="relu7")
		relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

		print("conv7:", np.shape(relu_dropout7))

		W8 = utils.weight_variable([1, 1, 4096, num_classes], name="W8")
		b8 = utils.bias_variable([num_classes], name="b8")
		conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)

		print("conv8:", np.shape(conv8))
		# annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

		# upsampling (deconv) part
		# pool4 + 2 x conv8
		deconv_shape1 = vgg["pool4"].get_shape()
		W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, num_classes], name="W_t1")
		b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
		conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(vgg["pool4"]))
		fuse_1 = tf.add(conv_t1, vgg["pool4"], name="fuse_1")

		print("pool4 and de_conv8 ==> fuse1:", np.shape(fuse_1))

		# pool3 + 2 x fuse_11
		deconv_shape2 = vgg["pool3"].get_shape()
		W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
		b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
		conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(vgg["pool3"]))
		fuse_2 = tf.add(conv_t2, vgg["pool3"], name="fuse_2")

		print("pool3 and deconv_fuse1 ==> fuse2:", np.shape(fuse_2))

		shape = tf.shape(image)
		deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], num_classes])
		W_t3 = utils.weight_variable([16, 16, num_classes, deconv_shape2[3].value], name="W_t3")
		b_t3 = utils.bias_variable([num_classes], name="b_t3")
		conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

		print("conv_t3:", np.shape(conv_t3))

	with tf.variable_scope("rec_decoder"):
		# add 1 pooling layer and 3 conv layer after VGG-19
		rec_W6 = utils.weight_variable([7, 7, 512, 4096], name="rec_W6")
		rec_b6 = utils.bias_variable([4096], name="rec_b6")
		rec_conv6 = utils.conv2d_basic(pool5, rec_W6, rec_b6)
		rec_relu6 = tf.nn.relu(rec_conv6, name="rec_relu6")
		rec_relu_dropout6 = tf.nn.dropout(rec_relu6, keep_prob=keep_prob)

		print("rec_conv6:", np.shape(rec_relu_dropout6))

		rec_W7 = utils.weight_variable([1, 1, 4096, 4096], name="rec_W7")
		rec_b7 = utils.bias_variable([4096], name="rec_b7")
		rec_conv7 = utils.conv2d_basic(rec_relu_dropout6, rec_W7, rec_b7)
		rec_relu7 = tf.nn.relu(rec_conv7, name="rec_relu7")
		rec_relu_dropout7 = tf.nn.dropout(rec_relu7, keep_prob=keep_prob)

		print("rec_conv7:", np.shape(rec_relu_dropout7))

		rec_W8 = utils.weight_variable([1, 1, 4096, num_classes], name="rec_W8")
		rec_b8 = utils.bias_variable([num_classes], name="rec_b8")
		rec_conv8 = utils.conv2d_basic(rec_relu_dropout7, rec_W8, rec_b8)

		print("rec_conv8:", np.shape(rec_conv8))

		# upsampling (deconv) part
		# pool4 + 2 x conv8
		deconv_shape1 = vgg["pool4"].get_shape()
		rec_W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, num_classes], name="rec_W_t1")
		rec_b_t1 = utils.bias_variable([deconv_shape1[3].value], name="rec_b_t1")
		rec_conv_t1 = utils.conv2d_transpose_strided(rec_conv8, rec_W_t1, rec_b_t1, output_shape=tf.shape(vgg["pool4"]))
		rec_fuse_1 = tf.add(rec_conv_t1, vgg["pool4"], name="rec_fuse_1")

		print("pool4 and de_conv8 ==> fuse1:", np.shape(rec_fuse_1))

		# pool3 + 2 x fuse_11
		deconv_shape2 = vgg["pool3"].get_shape()
		rec_W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="rec_W_t2")
		rec_b_t2 = utils.bias_variable([deconv_shape2[3].value], name="rec_b_t2")
		rec_conv_t2 = utils.conv2d_transpose_strided(rec_fuse_1, rec_W_t2, rec_b_t2, output_shape=tf.shape(vgg["pool3"]))
		rec_fuse_2 = tf.add(rec_conv_t2, vgg["pool3"], name="rec_fuse_2")

		print("pool3 and deconv_fuse1 ==> fuse2:", np.shape(rec_fuse_2))

		shape = tf.shape(image)
		deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], num_classes])
		rec_W_t3 = utils.weight_variable([16, 16, num_classes, deconv_shape2[3].value], name="rec_W_t3")
		rec_b_t3 = utils.bias_variable([num_classes], name="rec_b_t3")
		rec_conv_t3 = utils.conv2d_transpose_strided(rec_fuse_2, rec_W_t3, rec_b_t3, output_shape=deconv_shape3, stride=8)

		print("conv_t3:", np.shape(rec_conv_t3))

	encoder_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")
	seg_decoder_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="seg_decoder")
	rec_decoder_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="rec_decoder")

	seg_variable = encoder_variable + seg_decoder_variable
	rec_variable = encoder_variable + rec_decoder_variable

	return conv_t3, rec_conv_t3, seg_variable, rec_variable


def multi_fcn_joint(image, num_classes, keep_prob, mode):
	"""
	A multitask FCN model in which an extra 8x upsampling block that is identical to FCN was add to reconstruct unlabeled
	images in each classes.
	:param image: input images in shape[batch_size, height, width, channels]
	:param num_classes: number of classes, integer number
	:param keep_prob: probability of dropout layer
	:return: prediction scores in shape [batch_size, height, width, num_classes]
	"""
	assert mode == "seg" or mode == "rec"
	image = tf.image.grayscale_to_rgb(image)
	print(image.shape)
	# load weights
	model_data = utils.get_model_data('./imagenet-vgg-verydeep-19.mat')
	weights = np.squeeze(model_data['layers'])

	with tf.variable_scope("encoder"):
		# construct original VGG-19
		print("Construct VGG-19：")
		vgg = vgg_net(weights, image)

		conv_final_layer = vgg["conv5_3"]
		print("features from VGG：", np.shape(conv_final_layer))
		# add 1 pooling layer and 3 conv layer after VGG-19
		pool5 = utils.max_pool_2x2(conv_final_layer)
		print("pool5:", np.shape(pool5))
	encoder_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")

	if mode == "seg":
		with tf.variable_scope("seg_decoder"):
			W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
			b6 = utils.bias_variable([4096], name="b6")
			conv6 = utils.conv2d_basic(pool5, W6, b6)
			relu6 = tf.nn.relu(conv6, name="relu6")
			relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

			print("conv6:", np.shape(relu_dropout6))

			W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
			b7 = utils.bias_variable([4096], name="b7")
			conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
			relu7 = tf.nn.relu(conv7, name="relu7")
			relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

			print("conv7:", np.shape(relu_dropout7))

			W8 = utils.weight_variable([1, 1, 4096, num_classes], name="W8")
			b8 = utils.bias_variable([num_classes], name="b8")
			conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)

			print("conv8:", np.shape(conv8))

			# upsampling (deconv) part
			# pool4 + 2 x conv8
			deconv_shape1 = vgg["pool4"].get_shape()
			W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, num_classes], name="W_t1")
			b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
			conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(vgg["pool4"]))
			fuse_1 = tf.add(conv_t1, vgg["pool4"], name="fuse_1")

			print("pool4 and de_conv8 ==> fuse1:", np.shape(fuse_1))

			# pool3 + 2 x fuse_11
			deconv_shape2 = vgg["pool3"].get_shape()
			W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
			b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
			conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(vgg["pool3"]))
			fuse_2 = tf.add(conv_t2, vgg["pool3"], name="fuse_2")

			print("pool3 and deconv_fuse1 ==> fuse2:", np.shape(fuse_2))

			shape = tf.shape(image)
			deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], num_classes])
			W_t3 = utils.weight_variable([16, 16, num_classes, deconv_shape2[3].value], name="W_t3")
			b_t3 = utils.bias_variable([num_classes], name="b_t3")
			conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

			print("conv_t3:", np.shape(conv_t3))

		seg_decoder_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="seg_decoder")
		variable = encoder_variable + seg_decoder_variable

	elif mode == "rec":
		with tf.variable_scope("rec_decoder"):
			# add 1 pooling layer and 3 conv layer after VGG-19
			W6 = utils.weight_variable([7, 7, 512, 4096], name="rec_W6")
			b6 = utils.bias_variable([4096], name="rec_b6")
			conv6 = utils.conv2d_basic(pool5, W6, b6)
			relu6 = tf.nn.relu(conv6, name="rec_relu6")
			relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

			print("rec_conv6:", np.shape(relu_dropout6))

			W7 = utils.weight_variable([1, 1, 4096, 4096], name="rec_W7")
			b7 = utils.bias_variable([4096], name="rec_b7")
			conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
			relu7 = tf.nn.relu(conv7, name="rec_relu7")
			relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

			print("rec_conv7:", np.shape(relu_dropout7))

			W8 = utils.weight_variable([1, 1, 4096, num_classes], name="rec_W8")
			b8 = utils.bias_variable([num_classes], name="rec_b8")
			conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)

			print("rec_conv8:", np.shape(conv8))

			# upsampling (deconv) part
			# pool4 + 2 x conv8
			deconv_shape1 = vgg["pool4"].get_shape()
			W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, num_classes], name="rec_W_t1")
			b_t1 = utils.bias_variable([deconv_shape1[3].value], name="rec_b_t1")
			conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(vgg["pool4"]))
			fuse_1 = tf.add(conv_t1, vgg["pool4"], name="rec_fuse_1")

			print("pool4 and de_conv8 ==> fuse1:", np.shape(fuse_1))

			# pool3 + 2 x fuse_11
			deconv_shape2 = vgg["pool3"].get_shape()
			W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="rec_W_t2")
			b_t2 = utils.bias_variable([deconv_shape2[3].value], name="rec_b_t2")
			conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(vgg["pool3"]))
			fuse_2 = tf.add(conv_t2, vgg["pool3"], name="rec_fuse_2")

			print("pool3 and deconv_fuse1 ==> fuse2:", np.shape(fuse_2))

			shape = tf.shape(image)
			deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], num_classes])
			W_t3 = utils.weight_variable([16, 16, num_classes, deconv_shape2[3].value], name="rec_W_t3")
			b_t3 = utils.bias_variable([num_classes], name="rec_b_t3")
			conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

			print("conv_t3:", np.shape(conv_t3))

			encoder_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")

		rec_decoder_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="rec_decoder")
		variable = encoder_variable + rec_decoder_variable

	return conv_t3, variable


def inference_attention(image, num_classes, keep_prob):
	"""
	Semantic segmentation network definition
	:param image: input image. Should have values in range 0-255
	:param num_classes: number of classes
	:param keep_prob: dropout rate
	:return:
	"""
	image = tf.image.grayscale_to_rgb(image)
	print(image.shape)
	# load weights
	model_data = utils.get_model_data('./imagenet-vgg-verydeep-19.mat')
	weights = np.squeeze(model_data['layers'])

	"""
	# image preprocessor
	mean = model_data['normalization'][0][0][0]
	mean_pixel = np.mean(mean, axis=(0, 1))
	processed_image = utils.process_image(image, mean_pixel)
	print("预处理后的图像:", np.shape(processed_image))
	"""

	with tf.variable_scope("inference"):
		# establish original VGG-19

		print("Construct VGG-19：")
		image_net = vgg_net_attention(weights, image)
		print(image_net)

		# add 1 pooling layer and 3 conv layer after VGG-19
		conv_final_layer = image_net["conv5_3"]
		print("features from VGG：", np.shape(conv_final_layer))

		pool5 = utils.max_pool_2x2(conv_final_layer)

		print("pool5：", np.shape(pool5))

		W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
		b6 = utils.bias_variable([4096], name="b6")
		conv6 = utils.conv2d_basic(pool5, W6, b6)
		relu6 = tf.nn.relu(conv6, name="relu6")
		relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

		print("conv6:", np.shape(relu_dropout6))

		W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
		b7 = utils.bias_variable([4096], name="b7")
		conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
		relu7 = tf.nn.relu(conv7, name="relu7")
		relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

		print("conv7:", np.shape(relu_dropout7))

		W8 = utils.weight_variable([1, 1, 4096, num_classes], name="W8")
		b8 = utils.bias_variable([num_classes], name="b8")
		conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)

		print("conv8:", np.shape(conv8))
		# annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

		# upsampling (deconv) part
		# pool4 + 2 x conv8
		deconv_shape1 = image_net["pool4"].get_shape()
		W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, num_classes], name="W_t1")
		b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
		conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
		fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

		print("pool4 and de_conv8 ==> fuse1:", np.shape(fuse_1))

		# pool3 + 2 x fuse_11
		deconv_shape2 = image_net["pool3"].get_shape()
		W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
		b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
		conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
		fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

		print("pool3 and deconv_fuse1 ==> fuse2:", np.shape(fuse_2))

		shape = tf.shape(image)
		deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], num_classes])
		W_t3 = utils.weight_variable([16, 16, num_classes, deconv_shape2[3].value], name="W_t3")
		b_t3 = utils.bias_variable([num_classes], name="b_t3")
		conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

		print("conv_t3:", np.shape(conv_t3))

	return conv_t3


if __name__ == '__main__':
	root = "D:/university/python/cardiac_seg/CAMUS/"
	im, n_im, la = read_mhd(root, "val", "ED", 512, 512)  # only 2^n is allowed
	print(im.shape)
	print(n_im.shape)
	print(la.shape)
	img = tf.convert_to_tensor(n_im[0:5, :, :, :], dtype=tf.float32)
	label = tf.convert_to_tensor(la[0:5, :, :, :], dtype=tf.int64)
	x = tf.placeholder(dtype=tf.float32, shape=[5, 512, 512, 1])
	y = tf.placeholder(dtype=tf.int64, shape=[5, 512, 512, 1])
	logit = inference(x, 4, 1)
	# loss_logit = softmax_loss(logits=x, labels=y, is_regularize=False)