import torch
import os
import torch.nn as nn
import imageio
import argparse
import numpy as np
import math as m
from math import pi
from numpy.random import uniform
import neural_renderer as nr

# ---------------------------------------------------------------------------------
# convert the coordinate system of rendered image to be the same as the blender object exportation
# creation of the 3x3 rotation matrix and the 1x3 translation vector
# ---------------------------------------------------------------------------------


def AxisBlend2Rend(tx=0, ty=0, tz=0, alpha=0, beta=0, gamma=0):

    alpha = alpha - pi/2

    Rx = np.array([[1, 0, 0],
                   [0, m.cos(alpha), -m.sin(alpha)],
                   [0, m.sin(alpha), m.cos(alpha)]])

    Ry = np.array([[m.cos(beta), 0, -m.sin(beta)],
                   [0, 1, 0],
                   [m.sin(beta), 0, m.cos(beta)]])

    Rz = np.array([[m.cos(gamma), m.sin(gamma), 0],
                   [-m.sin(gamma), m.cos(gamma), 0],
                   [0, 0, 1]])


# create the rotation object matrix

    Rzy = np.matmul(Rz, Ry)
    Rzyx = np.matmul(Rzy, Rx)

    # R = np.matmul(Rx, Ry)
    # R = np.matmul(R, Rz)

    t = np.array([tx, -ty, -tz])

    return t, Rzyx

# ---------------------------------------------------------------------------------
# random set translation and rotation parameter
# ---------------------------------------------------------------------------------


def get_paramR_t():  # translation and rotation

    constraint_x = 2.5
    constraint_y   = 2.5

    constraint_angle = 180

    x = round(uniform(-constraint_x, constraint_x), 1)
    y = round(uniform(-constraint_y, constraint_y), 1)
    z =round( uniform(-15, -5), 1)

    alpha = round(uniform(-constraint_angle,constraint_angle), 0)
    beta = round(uniform(-constraint_angle,constraint_angle), 0)
    gamma = round(uniform(-constraint_angle,constraint_angle), 0)
    return alpha, beta, gamma, x, y, z


def get_param_t():  # only translation

    constraint_x = 2.5
    constraint_y   = 2.5

    constraint_angle = 0

    x = round(uniform(-constraint_x, constraint_x), 1)
    y = round(uniform(-constraint_y, constraint_y), 1)
    z =round( uniform(-15, -5), 1)

    alpha = 0
    beta = 0
    gamma = 0
    return alpha, beta, gamma, x, y, z
# ---------------------------------------------------------------------------------
# definition of the class camera with intrinsic and extrinsic parameter
# ---------------------------------------------------------------------------------
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

    K  = np.array([[f/pix_sizeX,0,Cam_centerX],
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
        self.t_mat, self.R_mat = AxisBlend2Rend(self.tx, self.ty, self.tz, m.radians(self.alpha), m.radians(self.beta), m.radians(self.gamma))

        self.K_vertices = np.repeat(camera_setttings.K[np.newaxis, :, :], vert, axis=0)
        self.R_vertices = np.repeat(self.R_mat[np.newaxis, :, :], vert, axis=0)
        self.t_vertices = np.repeat(self.t_mat[np.newaxis, :], 1, axis=0)


# database creation ---------------------------------------
# in: object (.obj file) name
# in: number of images do produce


def creation_database(Obj_Name, nb_im=10000):
    print("creation of 2 x %d images" % nb_im)
    first_im = True
    first_sil = True
    first_param = True

    for i in range(0, nb_im):
        # create path
        current_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(current_dir, 'data')
        train_dir = os.path.join(current_dir, 'data/test')
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, '{}.obj'.format(Obj_Name)))
        parser.add_argument('-c', '--color_input', type=str, default=os.path.join(data_dir, '{}.mtl'.format(Obj_Name)))
        parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(train_dir, 'cube_{}.png'.format(i)))
        parser.add_argument('-f', '--filename_output2', type=str, default=os.path.join(train_dir, 'silhouette_{}.png'.format(i)))
        parser.add_argument('-g', '--gpu', type=int, default=0)
        args = parser.parse_args()

        texture_size = 2

        # extract information from object
        vertices_1, faces_1 = nr.load_obj(args.filename_input)
        vertices_1 = vertices_1[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
        faces_1 = faces_1[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]
        nb_vertices = vertices_1.shape[0] #number of batch
        textures_1 = torch.ones(1, faces_1.shape[1], texture_size, texture_size, texture_size, 3,
                                dtype=torch.float32).cuda()

        # define extrinsic parameter
        alpha, beta, gamma, x, y, z = get_param_t()  # define transformation parameter

        R = np.array([alpha, beta, gamma])  # angle in degree param have to change
        t = np.array([x, y, z])  # translation in meter

        Rt =  np.concatenate((R, t), axis=None) # create one array of parameter

        # create camera with given parameters
        cam = camera_setttings(R=R, t=t, vert=nb_vertices)

        # create the renderer
        renderer = nr.Renderer(image_size=512, camera_mode='projection',dist_coeffs=None,
                               K=cam.K_vertices, R=cam.R_vertices, t=cam.t_vertices, near=0.1, background_color=[255,255,255],
                               far=1000, orig_size=512, light_direction=[0,-1,0])

        # save the image in pgn form
        #writer = imageio.get_writer(args.filename_output, mode='i')

        # render an image of the 3d object
        images_1 = renderer(vertices_1, faces_1, textures_1)  # [batch_size, RGB, image_size, image_size]
        image = images_1[0].detach().cpu().numpy()[0].transpose((1, 2, 0)) #float32 from 0 to 255
        im_norm = (255*image) #float32 from 76 to 65025
        im2save = im_norm.astype(np.uint8) #uint8 from 1 to 206
        #writer.append_data(im2save)

        # save the image in array form
        filename = 'data/test/cube.npy'
        first_im = save_pny(filename, first_im, im2save)
        #writer.close()

        # create the segmentation of the image
        # writer = imageio.get_writer(args.filename_output2, mode='i')

        images_1 = renderer(vertices_1, faces_1, textures_1, mode='silhouettes')  # [batch_size, RGB, image_size, image_size]
        image = images_1.detach().cpu().numpy().transpose((1, 2, 0))

        # writer.append_data((255 * image).astype(np.uint8))

        filename = 'data/test/silhouettes.npy'
        first_sil = save_pny(filename, first_sil, image.astype(np.int8))

        # writer.close()

        filename = 'data/test/param.npy'
        first_param = save_pny(filename, first_param, Rt.astype(np.float16))

# save images in npy file ---------------------------------------
# in: filename to write in
# in: boolean if it-s the first object to be written (file creation)
# in: image to write in file
# out: boolean update if it was the first image
def save_pny(filename, firstornot, image):
    if firstornot:
        image_expand = np.expand_dims(image, 0)  # [512,512,3] -> [1, 512, 512, 3]
        np.save(filename, image_expand)
        firstornot = False
        return firstornot

    else:
        all_im = np.load(filename)
        image_expand = np.expand_dims(image, 0)  # [512,512,3] -> [1, 512, 512, 3]
        img_cat = np.concatenate((all_im, image_expand))  # append to the existing images
        np.save(filename, img_cat)
