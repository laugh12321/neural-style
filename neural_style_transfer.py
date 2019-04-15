import time
import argparse
import numpy as np
from scipy.misc import imsave
from keras import backend as K
from keras.applications import vgg19
from scipy.optimize import fmin_l_bfgs_b
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageEnhance, ImageFilter


# 定义初始变量
parser = argparse.ArgumentParser(description='基于 Keras 的图像风格迁移')
parser.add_argument('style_reference_image_path', metavar='ref', nargs='+', type=str,
                    help='目标风格图片的位置.')
parser.add_argument('target_image_path', , metavar='base', type=str,
                    help='基准图片的位置.')
parser.add_argument('--iterations', dest='iterations', default=20, type=int,
                    help='迭代次数')				
#parser.add_argument("--image_size", dest="img_size", default=400, type=int,
                    help='生成图像的尺寸')

# 获取参数
args = parser.parse_args()
target_image_path = args.target_image_path
style_reference_image_path = args.style_reference_image_path
iterations = args.iterations
#image_size = args.image_size
img_width, img_height = load_img(target_image_path).size

# 辅助函数
def preprocess_image(image_path):
	'''
	预处理图片，包括变形到(1，width, height)形状，数据归一到0-1之间
	:param image: 输入一张图片
	:return: 预处理好的图片
	'''
	img = load_img(image_path, target_size=(img_height, img_width))
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0) # (width, height)->(1，width, height)
	img = vgg19.preprocess_input(img) # 0-255 -> 0-1.0
	return img

def deprocess_img(x):
	'''
    将0-1之间的数据变成图片的形式返回
    :param x: 数据在0-1之间的矩阵
    :return: 图片，数据都在0-255之间
	'''
	if K.image_data_format() == 'channels_first':
		x = x.reshape((3, img_height, img_width))
		x = x.transpose((1, 2, 0))
	else:
		x = x.reshape((img_height, img_width, 3))
    
	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68
	x = x[:, :, ::-1]  # 将图像由 BGR 格式转换为 RBG 格式
	x = np.clip(x, 0, 255).astype('uint8')
	return x
	
# 图像增强
def save_img(fname, image, image_enhance=True):  
	image = Image.fromarray(image)
	if image_enhance:
		# 亮度增强
		enh_bri = ImageEnhance.Brightness(image)
		brightness = 1.2
		image = enh_bri.enhance(brightness)

		# 色度增强
		enh_col = ImageEnhance.Color(image)
		color = 1.2
		image = enh_col.enhance(color)

		# 锐度增强
		enh_sha = ImageEnhance.Sharpness(image)
		sharpness = 1.2
		image = enh_sha.enhance(sharpness)
	imsave(fname, image)
	return

# 内容损失
def content_loss(base, combination):
	return K.sum(K.square(combination - base))

# Gram矩阵
def gram_matrix(x):
	assert K.ndim(x) == 3
	if K.image_data_format() == 'channels_first':
		features = K.batch_flatten(x)
	else:
		features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
	gram = K.dot(features, K.transpose(features))
	return gram
  
# 风格损失 - 是风格图片与结果图片的Gram矩阵之差，并对所有元素求和
def style_loss(style, combination):
	assert K.ndim(style) == 3
	assert K.ndim(combination) == 3
	S = gram_matrix(style)
	C = gram_matrix(combination)
	channels = 3
	size = img_height * img_width
	return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# 总变差损失
def total_variation_loss(x):
	assert K.ndim(x) == 4
	if K.image_data_format() == 'channels_first':
		a = K.square(x[:, :, :img_height - 1, :img_width - 1] - x[:, :, 1:,a :img_width - 1])
		b = K.square(x[:, :, :img_height - 1, :img_width - 1] - x[:, :, :img_height - 1, 1:])
	else:
		a = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1, :img_width - 1, :])
		b = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
	return K.sum(K.pow(a + b, 1.25))
  
def eval_loss_and_grads(x):
	if K.image_data_format() == 'channels_first':
		x = x.reshape((1, 3, img_height, img_width))
	else:
		x = x.reshape((1, img_height, img_width, 3))
	outs = f_outputs([x])
	loss_value = outs[0]
	if len(outs[1:]) == 1:
		grad_values = outs[1].flatten().astype('float64')
	else:
		grad_values = np.array(outs[1:]).flatten().astype('float64')
	return loss_value, grad_values

# 设置梯度下降过程
class Evaluator(object):
	def __init__(self):
		self.loss_value = None
		self.grads_values = None
    
	def loss(self, x):
		assert self.loss_value is None
		loss_value, grad_values = eval_loss_and_grads(x)
		self.loss_value = loss_value
		self.grad_values = grad_values
		return self.loss_value

	def grads(self, x):
		assert self.loss_value is not None
		grad_values = np.copy(self.grad_values)
		self.loss_value = None
		self.grad_values = None
		return grad_values
	
# 得到需要处理的数据，处理为keras的变量（tensor），处理为一个(3, width, height, 3)的矩阵
# 分别是基准图片，风格图片，结果图片
target_image = K.variable(preprocess_image(target_image_path))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))

# 用于保存生成图像
if K.image_data_format() == 'channels_first':
  combination_image = K.placeholder((1, 3, img_height, img_width))
else:
  combination_image = K.placeholder((1, img_height, img_width, 3))  

# 加载预训练的 VGG19 网络

# 将三张图像合并为一个批量
input_tensor = K.concatenate([target_image,
                              style_reference_image,
                              combination_image], axis=0)

# 利用三张图像组成的批量作为输入构建 VGG19 网络
# 加载模型将使用预训练的 ImageNet 权重
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet',
                    include_top=False)

print('Model loaded.')

# 定义需要最小话的最终损失

# 将层的名称映射为激活张量的字典
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
# 用于内容损失的层
content_layer = 'block5_conv2'
# 用于风格损失的层
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

# 损失分量的加权平均值所使用的权重
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

# 添加内容损失
loss = K.variable(0.) # 在定义损失时将所有分量添加到这个标量变量中
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features, combination_features)

# 添加每个目标层的风格损失分量
for layer_name in style_layers:
	layer_features = outputs_dict[layer_name]
	style_reference_features = layer_features[1, :, :, :]
	combination_features = layer_features[2, :, :, :]
	sl = style_loss(style_reference_features, combination_features)
	loss += (style_weight / len(style_layers)) * sl
# 添加总变量损失
loss += total_variation_weight * total_variation_loss(combination_image)


# 获取损失相对于生成图像的梯度
grads = K.gradients(loss, combination_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
  outputs += grads
else:
  outputs.append(grads)
  
# 用于获取当前损失值和当前梯度值的函数
f_outputs = K.function([combination_image], outputs)

evaluator = Evaluator()
result_prefix = 'result'
x = preprocess_image(target_image_path)  # 目标图像
x = x.flatten()  # 将图像展平，scipy.optimize.fmin_l_bfgs_b 只能处理展平的向量
for i in range(1, iterations+1):
	print('Start of iteration ', i)
	start_time = time.time()
	# 对生成图像的像素运行 L-BFGS 最优化
	# 以将神经风格损失最小化
	x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
	print('Current loss value: ', min_val)
	# 保存当前的生成图像
	if i == iterations or i % 5 == 0:
		img = deprocess_img(x.copy())
		fname = result_prefix + '_%d.png' % i
		save_img(fname, img)
		print('Image saved as ', fname)
	end_time = time.time()
	print('Iteration %d completed in %ds' % (i, end_time - start_time))
