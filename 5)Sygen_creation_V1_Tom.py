import torch
import os
import torch.nn as nn
import imageio
import argparse
import numpy as np
import math as m
from math import pi
import PIL
from numpy.random import uniform
import neural_renderer as nr
import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt


def BuildTransformationMatrix(tx=0, ty=0, tz=0, alpha=0, beta=0, gamma=0):

    # alpha = alpha - pi/2

    Rx = np.array([[1, 0, 0],
                   [0, m.cos(alpha), -m.sin(alpha)],
                   [0, m.sin(alpha), m.cos(alpha)]])

    Ry = np.array([[m.cos(beta), 0, m.sin(beta)],
                   [0, 1, 0],
                   [-m.sin(beta), 0, m.cos(beta)]])

    Rz = np.array([[m.cos(gamma), -m.sin(gamma), 0],
                   [m.sin(gamma), m.cos(gamma), 0],
                   [0, 0, 1]])


# create the rotation object matrix

    Rzy = np.matmul(Rz, Ry)
    Rzyx = np.matmul(Rzy, Rx)

    # R = np.matmul(Rx, Ry)
    # R = np.matmul(R, Rz)

    t = np.array([tx, ty, tz])

    return t, Rzyx

class camera_setttings():
    # instrinsic camera parameter are fixed

    resolutionX = 512  # in pixel
    resolutionY = 512
    scale = 1
    f = 35  # focal on lens
    sensor_width = 32  # in mm given in blender , camera sensor type
    pixels_in_u_per_mm = (resolutionX * scale) / sensor_width
    pixels_in_v_per_mm = (resolutionY * scale) / sensor_width
    pix_sizeX = 1 / pixels_in_u_per_mm
    pix_sizeY = 1 / pixels_in_v_per_mm

    Cam_centerX = resolutionX / 2
    Cam_centerY = resolutionY / 2

    K = np.array([[f/pix_sizeX,0,Cam_centerX],
                  [0,f/pix_sizeY,Cam_centerY],
                  [0,0,1]])  # shape of [nb_vertice, 3, 3]


    def __init__(self, R, t, vert): #R 1x3 array, t 1x2 array, number of vertices
        self.R =R
        self.t = t
        self.alpha = R[0]
        self.beta= R[1]
        self.gamma = R[2]
        self.tx = t[0]
        self.ty= t[1]
        self.tz=t[2]
        # angle in radian
        self.t_mat, self.R_mat = BuildTransformationMatrix(self.tx, self.ty, self.tz, self.alpha, self.beta, self.gamma)

        self.K_vertices = np.repeat(camera_setttings.K[np.newaxis, :, :], vert, axis=0)
        self.R_vertices = np.repeat(self.R_mat[np.newaxis, :, :], vert, axis=0)
        self.t_vertices = np.repeat(self.t_mat[np.newaxis, :], 1, axis=0)

def appendElement(all_elem, elem, first):

    if first:
        all_elem = np.expand_dims(elem, 0)  # create first element array
        first = False
    else:
        elem_exp = np.expand_dims(elem, 0)
        all_elem = np.concatenate((all_elem , elem_exp)) #append element to existing array

    return all_elem, first

def main():

    first_im = True
    first_sil = True
    first_param = True
    cubes_database = 0
    sils_database = 0
    params_database = 0

    vertices_1, faces_1, textures_1 = nr.load_obj("data/rubik_color.obj", load_texture=True)#, texture_size=4)
    print(vertices_1.shape)
    print(faces_1.shape)
    vertices_1 = vertices_1[None, :, :]  # add dimension
    faces_1 = faces_1[None, :, :]  #add dimension
    textures_1 = textures_1[None, :, :]  #add dimension
    nb_vertices = vertices_1.shape[0]
    texture_size = 1 # WHAT IS THIS?

    print(vertices_1.shape)
    print(faces_1.shape)

    # textures_1 = torch.ones(1, faces_1.shape[1], texture_size, texture_size, texture_size, 3,
                            # dtype=torch.float32).cuda()

    nb_im = 1000
    loop = tqdm.tqdm(range(0, nb_im))
    for i in loop:
        # R = np.array([np.radians(round(uniform(0, 90), 0)), np.radians(0), np.radians(0)])  # angle in degree
        # t = np.array([0, 0, 5])  # translation in meter
        alpha = uniform(0, 179)
        print(alpha)
        beta = 0
        gamma = 0
        R = np.array([np.radians(alpha), np.radians(beta), np.radians(gamma)])  # angle in degree
        t = np.array([0, 0, 5])  # translation in meter

        Rt = np.concatenate((R, t), axis=None).astype(np.float16)  # create one array of parameter in radian, this arraz will be saved in .npy file

        cam = camera_setttings(R=R, t=t, vert=nb_vertices) # degree angle will be converted  and stored in radian

        renderer = nr.Renderer(image_size=512, camera_mode='projection',dist_coeffs=None,
                               K=cam.K_vertices, R=cam.R_vertices, t=cam.t_vertices, near=1, background_color=[1,1,1], #changed from 0-255 to 0-1
                               far=1000, orig_size=512,
                               light_intensity_ambient=1.0,  light_intensity_directional=0, light_direction=[0,1,0],
                               light_color_ambient=[1,1,1], light_color_directional=[1,1,1]) #[1,1,1]
        #UNKNOWN: near? orig_size?

        images_1 = renderer(vertices_1, faces_1, textures_1)  # [batch_size, RGB, image_size, image_size]
        image = images_1[0].detach().cpu().numpy()[0].transpose((1, 2, 0)) #float32 from 0 to 255
        image = (image*255).astype(np.uint8) #cast from float32 255.0 to 255 uint8, background is filled now with  value 1-0 instead of 0-255
        if((i % 1000) == 0):
            plt.imshow(image)

            # print(image.shape)
            # print(np.max(image[:, :, 0]))
            # print(np.min(image[:, :, 0]))
            plt.show()
            plt.close()
            # save the image in array form
        cubes_database, first_im = appendElement(all_elem=cubes_database, elem=image, first=first_im)

        params_database, first_param = appendElement(all_elem=params_database, elem=Rt, first=first_param)

        # save database
    np.save('data/test/cubes_{}.npy'.format('rgb_test2'), cubes_database)
    np.save('data/test/params_{}.npy'.format('rgb_test_param2'), params_database)
    print('image saved')

        # plt.imshow(image)
        # plt.show()





if __name__ == '__main__':
    main()
