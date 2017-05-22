# coding:utf-8
import cv2
from PIL import Image
import numpy as np

n8 = np.array([[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]],
              np.uint8)

n4 = np.array([[0, 1, 0],
               [1, 1, 1],
               [0, 1, 0]],
              np.uint8)


def make_dot(src, k=3, scale=2, color=True, blur=0, erode=0):
    img_pl = Image.open(src)
    if img_pl.mode != 'RGB' and img_pl.mode != 'L':
        img_pl = img_pl.convert('RGB')
    img = np.asarray(img_pl)
    if color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = img.shape
        c = 0
    if erode == 1:
        img = cv2.erode(img, n4, iterations=1)
    elif erode == 2:
        img = cv2.erode(img, n8, iterations=1)
    if blur:
        img = cv2.bilateralFilter(img, 15, blur, 20)
    img = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_NEAREST)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    if color:
        img_cp = img.reshape(-1, c)
    else:
        img_cp = img.reshape(-1)
    img_cp = img_cp.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(img_cp, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    center = center.astype(np.uint8)
    result = center[label.flatten()]
    result = result.reshape((img.shape))
    return result
