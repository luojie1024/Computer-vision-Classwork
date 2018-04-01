# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math

'''
# @Time    : 18-3-31 下午2:16
# @Author  : 罗杰                  
# @ID      : F1710w0249
# @File    : assignments.py
# @Desc    : 计算机视觉作业01
'''


# 卷积
def imgConvolve(image, kernel):
    '''
    :param image: 图片矩阵
    :param kernel: 滤波窗口
    :return:卷积后的矩阵
    '''
    img_h = int(image.shape[0])
    img_w = int(image.shape[1])
    kernel_h = int(kernel.shape[0])
    kernel_w = int(kernel.shape[1])
    # padding
    padding_h = int((kernel_h - 1) / 2)
    padding_w = int((kernel_w - 1) / 2)

    convolve_h = int(img_h + 2 * padding_h)
    convolve_W = int(img_w + 2 * padding_w)

    # 分配空间
    img_padding = np.zeros((convolve_h, convolve_W))
    # 中心填充图片
    img_padding[padding_h:padding_h + img_h, padding_w:padding_w + img_w] = image[:, :]
    # 卷积结果
    image_convolve = np.zeros(image.shape)
    # 卷积
    for i in range(padding_h, padding_h + img_h):
        for j in range(padding_w, padding_w + img_w):
            image_convolve[i - padding_h][j - padding_w] = int(
                np.sum(img_padding[i - padding_h:i + padding_h + 1, j - padding_w:j + padding_w + 1] * kernel))

    return image_convolve


# 均值滤波
def imgAverageFilter(image, kernel):
    '''
    :param image: 图片矩阵
    :param kernel: 滤波窗口
    :return:均值滤波后的矩阵
    '''
    return imgConvolve(image, kernel) * (1.0 / kernel.size)


# 高斯滤波
def imgGaussian(sigma):
    '''
    :param sigma: σ标准差
    :return: 高斯滤波器的模板
    '''
    img_h = img_w = 2 * sigma + 1
    gaussian_mat = np.zeros((img_h, img_w))
    for x in range(-sigma, sigma + 1):
        for y in range(-sigma, sigma + 1):
            gaussian_mat[x + sigma][y + sigma] = np.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2))
    return gaussian_mat


# Sobel Edge
def sobelEdge(image, sobel):
    '''
    :param image: 图片矩阵
    :param sobel: 滤波窗口
    :return:Sobel处理后的矩阵
    '''
    return imgConvolve(image, sobel)


# Prewitt Edge
def prewittEdge(image, prewitt_x, prewitt_y):
    '''
    :param image: 图片矩阵
    :param prewitt_x: 竖直方向
    :param prewitt_y:  水平方向
    :return:处理后的矩阵
    '''
    img_X = imgConvolve(image, prewitt_x)
    img_Y = imgConvolve(image, prewitt_y)

    img_prediction = np.zeros(img_X.shape)
    for i in range(img_prediction.shape[0]):
        for j in range(img_prediction.shape[1]):
            img_prediction[i][j] = max(img_X[i][j], img_Y[i][j])
    return img_prediction


######################常量################################
# 滤波3x3
kernel_3x3 = np.ones((3, 3))
# 滤波5x5
kernel_5x5 = np.ones((5, 5))

# sobel 算子
sobel_1 = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_2 = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
# prewitt 算子
prewitt_1 = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

prewitt_2 = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]])

# ######################均值滤波################################
# 读图片
image = cv2.imread('balloonGrayNoisy.jpg', cv2.IMREAD_GRAYSCALE)
# 均值滤波
img_k3 = imgAverageFilter(image, kernel_3x3)

# 写图片
cv2.imwrite('average_3x3.jpg', img_k3)
# 均值滤波
img_k5 = imgAverageFilter(image, kernel_5x5)
# 写图片
cv2.imwrite('average_5x5.jpg', img_k5)

######################高斯滤波################################
image = cv2.imread('balloonGrayNoisy.jpg', cv2.IMREAD_GRAYSCALE)
img_gaus1 = imgAverageFilter(image, imgGaussian(1))
cv2.imwrite('gaussian1.jpg', img_gaus1)
img_gaus2 = imgAverageFilter(image, imgGaussian(2))
cv2.imwrite('gaussian2.jpg', img_gaus2)
img_gaus3 = imgAverageFilter(image, imgGaussian(3))
cv2.imwrite('gaussian3.jpg', img_gaus3)


######################Sobel算子################################
image=cv2.imread('buildingGray.jpg',cv2.IMREAD_GRAYSCALE)
img_spbel1 = sobelEdge(image, sobel_1)
cv2.imwrite('sobel1.jpg',img_spbel1)
img_spbel2 = sobelEdge(image, sobel_2)
cv2.imwrite('sobel2.jpg',img_spbel2)

######################prewitt算子################################
img_prewitt1 = prewittEdge(image, prewitt_1,prewitt_2)
cv2.imwrite('prewitt1.jpg',img_prewitt1)
