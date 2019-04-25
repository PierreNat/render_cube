#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:23:31 2019

@author: pierrec
"""
import matplotlib.pyplot as plt
import numpy as np
import math as m
import torch
fig=plt.figure()

cubes = np.load('cubes_rgb_test2.npy')
params = np.load('params_rgb_test_param2.npy')

for i in range(0,10):
    img = cubes[i]
    param = params[i]
    print(param)
    plt.imshow(img)
    plt.show()

