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

cube = np.load('cubesrgb_test.npy')
param = np.load('paramsrgb_test_param.npy')

for i in range(0,10):
    img = cube[i]


    plt.imshow(img)
    plt.show()

