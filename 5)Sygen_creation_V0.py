"""
working version of rendering 2 objects (cube-cone) given t and R
"""
import os
import argparse

import torch
import numpy as np
import tqdm
import imageio

import neural_renderer as nr
import math as m
import utils
from math import pi
from utils import creation_database



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)

# current_dir = os.path.dirname(os.path.realpath(__file__))
# data_dir = os.path.join(current_dir, 'data')
#
# def AxisBlend2Rend(tx=0, ty=0, tz=0, alpha=0, beta=0, gamma=0):
#
#     alpha = alpha - pi/2
#     # save = beta
#     # beta = gamma
#     # gamma = -save
#
#     Rx = np.array([[1, 0, 0],
#                    [0, m.cos(alpha), -m.sin(alpha)],
#                    [0, m.sin(alpha), m.cos(alpha)]])
#
#     Ry = np.array([[m.cos(beta), 0, -m.sin(beta)],
#                    [0, 1, 0],
#                    [m.sin(beta), 0, m.cos(beta)]])
#
#     Rz = np.array([[m.cos(gamma), m.sin(gamma), 0],
#                    [-m.sin(gamma), m.cos(gamma), 0],
#                    [0, 0, 1]])
#
# #creaete the rotation object matrix
#
#     Rzy = np.matmul(Rz, Ry)
#     Rzyx = np.matmul(Rzy, Rx)
#
#     # R = np.matmul(Rx, Ry)
#     # R = np.matmul(R, Rz)
#
#     t = np.array([tx, -ty, -tz])
#
#     return t, Rzyx

def main():

    obj_name = 'Large_dice'

    nb_im = 1

    creation_database(obj_name, nb_im)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, '{}.obj'.format(Name_1)))
    #
    # parser.add_argument('-c', '--color_input', type=str, default=os.path.join(data_dir, '{}.mtl'.format(Name_1)))
    # parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'cube_pipeline.png'))
    # parser.add_argument('-f', '--filename_output2', type=str, default=os.path.join(data_dir, 'cube_pipelline_silhouette.png'))
    # parser.add_argument('-g', '--gpu', type=int, default=0)
    # args = parser.parse_args()

    # other settings
    camera_distance = 10
    elevation = 30
    texture_size = 2

    # vertices_1, faces_1 = nr.load_obj(args.filename_input)
    # vertices_1 = vertices_1[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    # faces_1 = faces_1[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]
    #
    #
    # textures_1 = torch.ones(1, faces_1.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()

    # to gpu


# ---------------------------------------------------------------------------------
# intrinsic parameter, link camera coordinate to image plane
# ---------------------------------------------------------------------------------
#     resolutionX = 512  # in pixel
#     resolutionY = 512
#     scale = 1
#     f = 35  # focal on lens
#     sensor_width = 32  # in mm given in blender , camera sensor type
#     pixels_in_u_per_mm = (resolutionX * scale) / sensor_width
#     pixels_in_v_per_mm = (resolutionY * scale) / sensor_width
#     pix_sizeX = 1 / pixels_in_u_per_mm
#     pix_sizeY = 1 / pixels_in_v_per_mm
#
#     Cam_centerX = resolutionX / 2
#     Cam_centerY = resolutionY / 2
#
#     batch = vertices_1.shape[0]

    # K  = np.array([[f/pix_sizeX,0,Cam_centerX],
    #               [0,f/pix_sizeY,Cam_centerY],
    #               [0,0,1]])  # shape of [nb_vertice, 3, 3]
    #
    # K = np.repeat(K[np.newaxis, :, :], batch, axis=0) # shape of [batch=1, 3, 3]

# ---------------------------------------------------------------------------------
# extrinsic parameter, link camera coordinate to world/object coordinate
# ---------------------------------------------------------------------------------
#
#     alpha_1 = 20  # x-axis rotation
#     beta_1 = 30  # y-axis rotation
#     gamma_1 = 40  # z-axis rotation
#
#     tx_1 = -1.6  # in meter
#     ty_1 = 0
#     tz_1 = -5
#     scale_1 = 1
#
#     t_1, R_1 = AxisBlend2Rend(tx_1, ty_1, tz_1, m.radians(alpha_1), m.radians(beta_1), m.radians(gamma_1))
#
#
#     R_1 = np.repeat(R_1[np.newaxis, :, :], batch, axis=0)  # shape of [batc
#     t_1 = np.repeat(t_1[np.newaxis, :], 1, axis=0)  # shape of [1, 3]



# ---------------------------------------------------------------------------------
# create the rendering
# ---------------------------------------------------------------------------------

    # renderer = nr.Renderer(image_size=512, camera_mode='projection',dist_coeffs=None, K=K, R=R_1, t=t_1, near=0.1,background_color=[255,255,255],
    #                        far=1000, orig_size=512/scale_1, light_direction=[0,-1,0])


# ---------------------------------------------------------------------------------
# Render object
# ---------------------------------------------------------------------------------

    # nb_obj2render = 1
    #
    # loop = tqdm.tqdm(range(0, 1, 1))
    # writer = imageio.get_writer(args.filename_output, mode='i')
    #
    # for num, obj in enumerate(loop):
    #     loop.set_description('Rendering objects')
    #
    #     if nb_obj2render == 1:  # render 1 3d object
    #         images_1 = renderer(vertices_1, faces_1, textures_1)  # [batch_size, RGB, image_size, image_size]
    #         image = images_1[0].detach().cpu().numpy()[0].transpose((1, 2, 0))
    #
    #     else: # render 2 3d objects
    #         images_1 = renderer(vertices_1, faces_1, textures_1)
    #         images_2 = renderer2(vertices_2, faces_2, textures_2)
    #
    #         image = np.minimum(images_1[0].detach().cpu().numpy()[0].transpose((1, 2, 0)),
    #                            images_2[0].detach().cpu().numpy()[0].transpose((1, 2, 0)))
    #
    #     image[np.int(resolutionX/2), np.int(resolutionY/2)] = [1, 1, 1]  # draw middle point camera center
    #     writer.append_data((255*image).astype(np.uint8))
    #
    # writer.close()

# ---------------------------------------------------------------------------------
# Render object silhouette
# ---------------------------------------------------------------------------------

    # loop = tqdm.tqdm(range(0, 1, 1))
    # writer = imageio.get_writer(args.filename_output2, mode='i')
    # for num, azimuth in enumerate(loop):
    #     loop.set_description('Rendering objects silhouettes')
    #
    #     if nb_obj2render == 1:  # render 1 3d object
    #         images_1 = renderer(vertices_1, faces_1, textures_1, mode='silhouettes')  # [batch_size, RGB, image_size, image_size]
    #         image = images_1.detach().cpu().numpy().transpose((1, 2, 0))
    #
    #     else:  # render 2 3d objects
    #         images_1 = renderer(vertices_1, faces_1, textures_1, mode='silhouettes')
    #         images_2 = renderer2(vertices_2, faces_2, textures_2, mode='silhouettes')
    #         image = np.maximum(images_1.detach().cpu().numpy().transpose((1, 2, 0)),
    #                            images_2.detach().cpu().numpy().transpose((1, 2, 0)))
    #
    #     writer.append_data((255*image).astype(np.uint8))
    # writer.close()

if __name__ == '__main__':
    main()

