import torch
import os
import torch.nn as nn
import imageio
import argparse
import numpy as np
import math as m
from math import pi
import neural_renderer as nr

def AxisBlend2Rend(tx=0, ty=0, tz=0, alpha=0, beta=0, gamma=0):

    alpha = alpha - pi/2
    # save = beta
    # beta = gamma
    # gamma = -save

    Rx = np.array([[1, 0, 0],
                   [0, m.cos(alpha), -m.sin(alpha)],
                   [0, m.sin(alpha), m.cos(alpha)]])

    Ry = np.array([[m.cos(beta), 0, -m.sin(beta)],
                   [0, 1, 0],
                   [m.sin(beta), 0, m.cos(beta)]])

    Rz = np.array([[m.cos(gamma), m.sin(gamma), 0],
                   [-m.sin(gamma), m.cos(gamma), 0],
                   [0, 0, 1]])

#creaete the rotation object matrix

    Rzy = np.matmul(Rz, Ry)
    Rzyx = np.matmul(Rzy, Rx)

    # R = np.matmul(Rx, Ry)
    # R = np.matmul(R, Rz)

    t = np.array([tx, -ty, -tz])

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
# in: number of images
# out: validation bool

def creation_database(Obj_Name, nb_im=10000, R=np.array([0, 0, 0]),  t=np.array([0, 0, 0])):
    print(nb_im)


    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, '{}.obj'.format(Obj_Name)))
    parser.add_argument('-c', '--color_input', type=str, default=os.path.join(data_dir, '{}.mtl'.format(Obj_Name)))

    for i in range(0, nb_im):
        parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'cube_{}.png'.format(i)))
        parser.add_argument('-f', '--filename_output2', type=str, default=os.path.join(data_dir, 'silhouette_{}.png'.format(i)))
        parser.add_argument('-g', '--gpu', type=int, default=0)
        args = parser.parse_args()

        texture_size = 2
        vertices_1, faces_1 = nr.load_obj(args.filename_input)
        vertices_1 = vertices_1[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
        faces_1 = faces_1[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]
        nb_vertices = vertices_1.shape[0]
        textures_1 = torch.ones(1, faces_1.shape[1], texture_size, texture_size, texture_size, 3,
                                dtype=torch.float32).cuda()

        writer = imageio.get_writer(args.filename_output, mode='i')

        R = np.array([0, 0, 0])  # angle in degree param have to change
        t = np.array([0, 0, -5])  # translation in meter

        cam = camera_setttings(R, t, nb_vertices)
        renderer = nr.Renderer(image_size=512, camera_mode='projection',dist_coeffs=None,
                               K=cam.K_vertices, R=cam.R_vertices, t=cam.t_vertices, near=0.1, background_color=[255,255,255],
                               far=1000, orig_size=1, light_direction=[0,-1,0])

        images_1 = renderer(vertices_1, faces_1, textures_1)  # [batch_size, RGB, image_size, image_size]
        image = images_1[0].detach().cpu().numpy()[0].transpose((1, 2, 0))

        writer.append_data((255*image).astype(np.uint8))

        writer.close()

        writer = imageio.get_writer(args.filename_output2, mode='i')
        images_1 = renderer(vertices_1, faces_1, textures_1,
                            mode='silhouettes')  # [batch_size, RGB, image_size, image_size]
        image = images_1.detach().cpu().numpy().transpose((1, 2, 0))
        writer.append_data((255 * image).astype(np.uint8))
        writer.close()



