"""
Example 1. Drawing a teapot from multiple viewpoints.
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
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, 'Large_dice.obj'))
    parser.add_argument('-c', '--color_input', type=str, default=os.path.join(data_dir, 'Large_dice.mtl'))
    parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'rubik2_SyGen_proj3.png'))
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
#---------------------------------------------------------------------------------
#extrinsic parameter, link world/object coordinate to camera coordinate
#---------------------------------------------------------------------------------
    alpha = 0
    beta = 0
    gamma = 0
 
    resolutionX = 512 #in pixel
    resolutionY = 512
    scale = 1
    f = 35 #focal on lens 
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

#   creaete the rotation camera matrix 
    
    R = np.matmul(Rx,Ry)
    R = np.matmul(R,Rz)

    
    t = np.array([0,0,5]) #camera position [x,y, z] 0 0 5

    R = np.repeat(R[np.newaxis, :, :], batch, axis=0) # shape of [batch=1, 3, 3]
    t = np.repeat(t[np.newaxis, :], 1, axis=0)# shape of [1, 3]


    
#---------------------------------------------------------------------------------    
#intrinsic parameter, link camera coordinate to image plane
#---------------------------------------------------------------------------------
    
    K  = np.array([[f/pix_sizeX,0,Cam_centerX],
                  [0,f/pix_sizeY,Cam_centerY],
                  [0,0,1]])# shape of [nb_vertice, 3, 3]
    
    K = np.repeat(K[np.newaxis, :, :], batch, axis=0) # shape of [batch=1, 3, 3]
    
    # create renderer
#    renderer = nr.Renderer(image_size=512, camera_mode='projection')
    renderer = nr.Renderer(image_size=512, camera_mode='projection',dist_coeffs=None, K=K, R=R, t=t, near=0.1, far=1000)
#    renderer = nr.Renderer(camera_mode='look_at')
    
    
    # draw object
    angle= 0
    loop = tqdm.tqdm(range(angle, angle+1, 4))
    writer = imageio.get_writer(args.filename_output, mode='i')
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
        images = renderer(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
        image = images[0].detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
        image[np.int(resolutionX/2),np.int(resolutionY/2)] = [1,1,1] #draw middle point camera center
        writer.append_data((255*image).astype(np.uint8))
    


    writer.close()

if __name__ == '__main__':
    main()
    
