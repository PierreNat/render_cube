"""
Render a set of synthetic images only with dice translation
"""
import os
import argparse

import torch
import numpy as np
import tqdm
import imageio

import neural_renderer as nr
import math as m
from math import pi

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, 'dice3.obj'))
    parser.add_argument('-c', '--color_input', type=str, default=os.path.join(data_dir, 'rubik2_colored.mtl'))
    parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'rubik2_SyGen_proj_2meshes.png'))
    parser.add_argument('-f', '--filename_output2', type=str, default=os.path.join(data_dir, 'rubik2_SyGen_proj_2meshes_silhouette.png'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    # other settings
    camera_distance = 10
    elevation = 30
    texture_size = 2

    #load 
    colors, texture_filenames = nr.load_mtl(args.color_input)
    # load .obj
    vertices, faces = nr.load_obj(args.filename_input)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()

    # to gpu
# ---------------------------------------------------------------------------------
# extrinsic parameter, link world/object coordinate to camera coordinate
# ---------------------------------------------------------------------------------
    alpha = 0
    beta = 0
    gamma = 0
 
    resolutionX = 512 # in pixel
    resolutionY = 512
    scale = 1
    f = 35 # focal on lens
    sensor_width = 32 # in mm given in blender , camera sensor type
    pixels_in_u_per_mm = (resolutionX*scale)/sensor_width
    pixels_in_v_per_mm = (resolutionY*scale)/sensor_width
    pix_sizeX = 1/pixels_in_u_per_mm
    pix_sizeY = 1/pixels_in_v_per_mm
    
    Cam_centerX = resolutionX/2
    Cam_centerY = resolutionY/2
    
    
    batch = vertices.shape[0]
    Rx = np.array([[1,0,0],
                  [0,m.cos(alpha),-m.sin(alpha)],
                  [0,m.sin(alpha),m.cos(alpha)]])
    
    Ry  = np.array([[m.cos(beta),0,-m.sin(beta)],
                  [0,1,0],
                  [m.sin(beta),0,m.cos(beta)]])
    
    Rz = np.array([[m.cos(gamma),-m.sin(gamma),0],
                  [m.sin(gamma),m.cos(gamma),0],
                  [0,0,1]]) 
  
#   creaete the rotation object matrix
    
    R = np.matmul(Rx,Ry)
    R = np.matmul(R,Rz)

    t_1 = np.array([0,0,5]) # object position [x,y, z] 0 0 5
    t_2 = np.array([-1, 2, 5])

    R = np.repeat(R[np.newaxis, :, :], batch, axis=0) # shape of [batch=1, 3, 3]
    t_1 = np.repeat(t_1[np.newaxis, :], 1, axis=0)  # shape of [1, 3]
    t_2 = np.repeat(t_2[np.newaxis, :], 1, axis=0)

# ---------------------------------------------------------------------------------
# intrinsic parameter, link camera coordinate to image plane
# ---------------------------------------------------------------------------------
    
    K  = np.array([[f/pix_sizeX,0,Cam_centerX],
                  [0,f/pix_sizeY,Cam_centerY],
                  [0,0,1]])  # shape of [nb_vertice, 3, 3]
    
    K = np.repeat(K[np.newaxis, :, :], batch, axis=0) # shape of [batch=1, 3, 3]
    
    # create renderer
#    renderer = nr.Renderer(image_size=512, camera_mode='projection')
    renderer = nr.Renderer(image_size=512, camera_mode='projection',dist_coeffs=None, K=K, R=R, t=t_1, near=0.1, far=1000, orig_size=1024)
    renderer2 = nr.Renderer(image_size=512, camera_mode='projection', dist_coeffs=None, K=K, R=R, t=t_2, near=0.1,far=1000, orig_size=512)

# render object

    nb_obj2render = 2

    loop = tqdm.tqdm(range(0, 1, 1))
    writer = imageio.get_writer(args.filename_output, mode='i')

    for num, obj in enumerate(loop):
        loop.set_description('Rendering objects')

        if nb_obj2render == 1:  # render 1 3d object
            images_1 = renderer(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
            image = images_1[0].detach().cpu().numpy()[0].transpose((1, 2, 0))

        else: # render 2 3d objects
            images_1 = renderer(vertices, faces, textures)
            images_2 = renderer2(vertices, faces, textures)

            image = np.minimum(images_1[0].detach().cpu().numpy()[0].transpose((1, 2, 0)),
                               images_2[0].detach().cpu().numpy()[0].transpose((1, 2, 0)))

        image[np.int(resolutionX/2), np.int(resolutionY/2)] = [1, 1, 1]  # draw middle point camera center
        writer.append_data((255*image).astype(np.uint8))

    writer.close()

# render object silhouette
    loop = tqdm.tqdm(range(0, 1, 1))
    writer = imageio.get_writer(args.filename_output2, mode='i')
    for num, azimuth in enumerate(loop):
        loop.set_description('Rendering objects silhouettes')

        if nb_obj2render == 1:  # render 1 3d object
            images_1 = renderer(vertices, faces, textures, mode='silhouettes')  # [batch_size, RGB, image_size, image_size]
            image = images_1.detach().cpu().numpy().transpose((1, 2, 0))

        else:  # render 2 3d objects
            images_1 = renderer(vertices, faces, textures, mode='silhouettes')
            images_2 = renderer2(vertices, faces, textures, mode='silhouettes')
            image = np.maximum(images_1.detach().cpu().numpy().transpose((1, 2, 0)),
                               images_2.detach().cpu().numpy().transpose((1, 2, 0)))

        writer.append_data((255*image).astype(np.uint8))
    writer.close()

if __name__ == '__main__':
    main()

