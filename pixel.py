# coding:utf-8
import sys
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


def make_dot(src, k=3, scale=2, color=True, blur=0, erode=0, alpha=True, to_tw=True):
    img_pl = Image.open(src)
    if (img_pl.mode == 'RGBA' or img_pl.mode == 'P') and alpha:
        if img_pl.mode != 'RGBA':
            img_pl = img_pl.convert('RGBA')
        alpha_mode = True
    elif img_pl.mode != 'RGB' and img_pl.mode != 'L':
        img_pl = img_pl.convert('RGB')
        alpha_mode = False
    else:
        alpha_mode = False
    img = np.asarray(img_pl)
    if color and alpha_mode:
        a = img[:, :, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        h, w, c = img.shape
    elif color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = img.shape
        c = 0
    d_h = int(h / scale)
    d_w = int(w / scale)
    if erode == 1:
        img = cv2.erode(img, n4, iterations=1)
    elif erode == 2:
        img = cv2.erode(img, n8, iterations=1)
    if blur:
        img = cv2.bilateralFilter(img, 15, blur, 20)
    img = cv2.resize(img, (d_w, d_h), interpolation=cv2.INTER_NEAREST)
    if alpha_mode:
        a = cv2.resize(a, (d_w, d_h), interpolation=cv2.INTER_NEAREST)
        a = cv2.resize(a, (d_w * scale, d_h * scale), interpolation=cv2.INTER_NEAREST)
        a[a != 0] = 255
        if not 0 in a:
            a[0, 0] = 0
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
    result = cv2.resize(result, (d_w * scale, d_h * scale), interpolation=cv2.INTER_NEAREST)
    if alpha_mode:
        r, g, b = cv2.split(result)
        result = cv2.merge((r, g, b, a))
    elif to_tw:
        r, g, b = cv2.split(result)
        a = np.ones(r.shape, dtype=np.uint8) * 255
        a[0, 0] = 0
        result = cv2.merge((r, g, b, a))
    colors = []
    for res_c in center:
        color_code = '#{0:02x}{1:02x}{2:02x}'.format(res_c[2], res_c[1], res_c[0])
        colors.append(color_code)
    return result, colors
