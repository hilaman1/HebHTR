import cv2
import numpy as np
import re
from matplotlib import pyplot as plt

regex = r'^[.,/"=+_?!*%~\'{}\[\]:().,;]+$'
path = r'..\HebHTR\data'

# Resize image to fit model's input size, and place it on model's size empty image.
def preprocessImageForPrediction(img, imgSize):
    # Read the gray-value image
    img = cv2.bitwise_not(img)
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('grayscale_image', grayscale_image)
    # create target image and copy sample image into it
    (wt, ht) = imgSize
    print("in preprocessImageForPrediction imgSize ", imgSize)
    print("in preprocessImageForPrediction img.shape ", img.shape)
    (h, w) = grayscale_image.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)

    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)),
                                                1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(grayscale_image, newSize)
    #cv2.imshow('grayscale_image', img)
    print("in preprocessImageForPrediction after resize", img.shape)
    target = np.ones([ht, wt]) * 255
    cv2.imwrite(fr'{path}\target.png', target)
    c = cv2.imread(fr'{path}\target.png', 1)
    #cv2.imshow('target', c)
    # map image into target image
    target[0:newSize[1], 0:newSize[0]] = img
    print("target.shape", target.shape)
    print("target[:, :].shape", target[:, :].shape)
    cv2.imwrite(fr'{path}\target.png', target)
    c = cv2.imread(fr'{path}\target.png', 1)
    #cv2.imshow('target', c)
    #cv2.waitKey(0)

    # transpose for TF
    img_tf = cv2.transpose(target)
    print("img_tf.shape", img_tf.shape)
    #cv2.imshow('img_tf', img_tf)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # normalize
    (m, s) = cv2.meanStdDev(img_tf)
    m = m[0][0]
    s = s[0][0]
    img_tf = img_tf - m
    img_tf = img_tf / s if s > 0 else img_tf
    #cv2.imshow('img_tf_afterall', img_tf)
    return img_tf
