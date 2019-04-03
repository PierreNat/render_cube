#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:23:31 2019

@author: pierrec
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
fig=plt.figure()
cube = np.load('cube.npy')
sil = np.load('silhouette.npy')
img = cube
img2 = sil
print(np.shape(img))
print(np.shape(img2))
img2 = img2.reshape((512,512))
fig.add_subplot(1, 2, 1)
plt.imshow(img)
fig.add_subplot(1, 2, 2)
plt.imshow(img2, cmap='gray')
a = torch.from_numpy(cube)
print(a.size())
image2show = a
im_resized = (image2show).numpy().transpose(0, 1, 2)
plt.imshow(im_resized)

