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
num = 0
cube = np.load('cubes.npy')
sil = np.load('sils.npy')
param = np.load('params.npy')
torch_cube = torch.from_numpy(cube) 
img = cube[num]
img2 = sil[num]
#img2 = np.squeeze(img2)
fig.add_subplot(1, 3, 1)
plt.imshow(img)
fig.add_subplot(1, 3, 2)
plt.imshow(img2, cmap='gray')
plt.show()

x = np.array((1,2,3))
y = np.array((4,5,6))
xy = np.concatenate((x, y), axis=None)

np.savez('cubes.npz','cubes.npy')
#try to direct append

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
