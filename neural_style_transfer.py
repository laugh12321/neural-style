import time
import argparse
import numpy as np
from keras import backend as K
from keras.applications import vgg19
from scipy.optimize import fmin_l_bfgs_b
from keras.preprocessing.image import load_img, save_img, img_to_array

# 定义初始变量
parser = argparse.ArgumentParser(description='基于 Keras 的图像风格迁移')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='基准图片的位置')
parser.add_argument('style_reference_image_path', metavar='ref', type=str,
                    help='目标风格图片的位置')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='迭代次数')
parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')
parser.add_argument("--image_size", type=int, default=400, required=False,
                    help='Minimum image size')
					
# 获取参数
args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
result_prefix = args.result_prefix
iterations = args.iter
img_nrows = args.image_size

# 损失分量的加权平均值所使用的权重
total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

# # 生成图像的尺寸
width, height = load_img(base_image_path).size

img_ncols = int(width * img_nrows / height)

# 函数的作用是:打开、调整图片的大小和格式，使之成为合适的张量
def preprocess_image(image_path):
	'''
    预处理图片，包括变形到(1，width, height)形状，数据归一到0-1之间
    :param image: 输入一张图片
    :return: 预处理好的图片
    '''
	img = load_img(image_path, target_size=(img_nrows, img_ncols))
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = vgg19.preprocess_input(img)
	return img

# util function to convert a tensor into a valid image
def deprocess_image(x):
    '''
    将0-1之间的数据变成图片的形式返回
    :param x: 数据在0-1之间的矩阵
    :return: 图片，数据都在0-255之间
    '''
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8') # 以防溢出255范围
    return x

# get tensor representations of our images
base_image = K.variable(preprocess_image(base_image_path))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))

# # 用于保存生成图像
if K.image_data_format() == 'channels_first':
    combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
else:
    combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

# 将三张图像合并为一个批量
input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image], axis=0)

# 利用三张图像组成的批量作为输入构建 VGG19 网络
# 加载模型将使用预训练的 ImageNet 权重
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)
print('Model loaded.')

# 将层的名称映射为激活张量的字典
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
# 用于内容损失的层
content_layer = 'block5_conv2'
# 用于风格损失的层
feature_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
				
# compute the neural style loss
# first we need to define 4 util functions

# the gram matrix of an image tensor (feature-wise outer product)

# # Gram矩阵
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image

# 风格损失
# 风格图片与结果图片的Gram矩阵之差，并对所有元素求和
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image

# 内容损失
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent

# 总变差损失
def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# 添加内容损失
loss = K.variable(0.0) # 在定义损失时将所有分量添加到这个标量变量中
layer_features = outputs_dict[content_layer]
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features,
                                      combination_features)
									  
# 添加每个目标层的风格损失分量
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl
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

# 输入x，输出对应于x的梯度和loss
def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x]) # 输入x，得到输出
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.


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

evaluator = Evaluator()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
x = preprocess_image(base_image_path) # 目标图像
x = x.flatten()  # 将图像展平，scipy.optimize.fmin_l_bfgs_b 只能处理展平的向量

for i in range(1, iterations+1):
    print('Start of iteration', i)
    start_time = time.time()
	# 对生成图像的像素运行 L-BFGS 最优化
	# 以将神经风格损失最小化
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # 保存当前的生成图像
    img = deprocess_image(x.copy())
    fname = result_prefix + '_at_iteration_%d.png' % i
    save_img(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))