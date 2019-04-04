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

img = cube[0]
img2 = sil[1]
fig.add_subplot(1, 3, 1)
plt.imshow(img)
fig.add_subplot(1, 3, 2)
plt.imshow(img2, cmap='gray')

#
#img_expand = np.expand_dims(cube, axis=0)
#img_cat  = np.stack((img,img),0)
#img_sel = img_cat[0]
#img2 = sil
#print(np.shape(img))
#print(np.shape(img2))
#img2 = img2.reshape((512,512))
#fig.add_subplot(1, 3, 1)
#plt.imshow(img)
#fig.add_subplot(1, 3, 2)
#plt.imshow(img2, cmap='gray')
#a = torch.from_numpy(cube) #conversion to torch, will be done by the transform function: train_dataset = DigitDataset(train_im,train_ctr,transforms)
#print(a.size())
#b = a.unsqueeze(0)
#print(b.size())
#c = torch.cat((b,b), 0)
#print(c.size())
#image2show = c[1]
#im_resized = (image2show).numpy().transpose(0, 1, 2)
##plt.imshow(im_resized)
#np.save('extende_cube.npy', c)
#ex_cube = np.load('extende_cube.npy')
#ex_cube_tensor =torch.from_numpy(ex_cube)
#print(ex_cube_tensor.size())
##fig.add_subplot(1, 3, 2)
#plt.imshow(ex_cube_tensor[1])

#
#fig=plt.figure()
#cube = np.load('cube.npy')
#sil = np.load('silhouette.npy')
#img = cube
#img2 = sil
#print(np.shape(img))
#print(np.shape(img2))
#img2 = img2.reshape((512,512))
##fig.add_subplot(1, 3, 1)
##plt.imshow(img)
##fig.add_subplot(1, 3, 2)
##plt.imshow(img2, cmap='gray')
#a = torch.from_numpy(cube) #conversion to torch, will be done by the transform function: train_dataset = DigitDataset(train_im,train_ctr,transforms)
#print(a.size())
#b = a.unsqueeze(0)
#print(b.size())
#c = torch.cat((b,b), 0)
#print(c.size())
#image2show = c[1]
#im_resized = (image2show).numpy().transpose(0, 1, 2)
##plt.imshow(im_resized)
#np.save('extende_cube.npy', c)
#ex_cube = np.load('extende_cube.npy')
#ex_cube_tensor =torch.from_numpy(ex_cube)
#print(ex_cube_tensor.size())
##fig.add_subplot(1, 3, 2)
#plt.imshow(ex_cube_tensor[1])
