# -*- coding: utf-8 -*-

# 计算两张图片的SSIM
from skimage import measure
import cv2
import glob
import os


dir_path = '/path/to/origin/dir/'
origin_images_glob_path = os.path.join(dir_path, '/train/*/*.png')
origin_images_path = glob.glob(origin_images_glob_path)
origin_images_name = [item.split('/')[-1] for item in origin_images_path]

li = []
for index, name in enumerate(origin_images_name):
    image_after_glob_path = os.path.join('/data/liupeidong/cifar10_step0.1_iter100_nonorm/train', '*', name)
    image_after_path = glob.glob(image_after_glob_path)

    im1 = cv2.imread(origin_images_path[index])
    im2 = cv2.imread(image_after_path[0])

    li.append(measure.compare_ssim(im1, im2, data_range=255, multichannel=True))

print(sum(li)/len(li))

