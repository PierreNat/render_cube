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

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, 'dice2.obj'))
    parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'rubik2_SyGen_lookat.png'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    # other settings for look or look at renderer
    camera_distance = 5
    elevation = 0
    texture_size = 2

    #load 

    # load .obj
    vertices, faces = nr.load_obj(args.filename_input)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()

    # to gpu

    # create renderer
    renderer = nr.Renderer(camera_mode='look_at')

    # draw object
    angle= 0
    loop = tqdm.tqdm(range(angle, angle+1, 4))
    writer = imageio.get_writer(args.filename_output, mode='i')
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
        images = renderer(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
        image = images[0].detach().cpu().numpy()[0]# [image_size, image_size, RGB]
        writer.append_data((255*image).astype(np.uint8))
    


    writer.close()

if __name__ == '__main__':
    main()