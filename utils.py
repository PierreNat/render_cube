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

    constraint_angle = 180

    #draw random value of R and t in a specific span
    x = round(uniform(-constraint_x, constraint_x), 1)
    y = round(uniform(-constraint_y, constraint_y), 1)
    z = round( uniform(-15, -5), 1)

    # alpha = round(uniform(-constraint_angle, constraint_angle), 1)
    # beta = round(uniform(-constraint_angle, constraint_angle), 1)
    # gamma = round(uniform(-constraint_angle, constraint_angle), 1)

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
    cubes_database = 0
    sils_database = 0
    params_database = 0
    loop = tqdm.tqdm(range(0, nb_im))
    for i in loop:
        # loop.set_description('render png {}'.format(i))
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

        cubes_database, first_im = appendElement(all_elem=cubes_database, elem=im2save, first=first_im)
        # first_im = save_pny(filename, first_im, im2save)
        #writer.close()

        # create the segmentation of the image
        # writer = imageio.get_writer(args.filename_output2, mode='i')
        # loop.set_description('render silhouette {}'.format(i))
        images_1 = renderer(vertices_1, faces_1, textures_1, mode='silhouettes')  # [batch_size, RGB, image_size, image_size]
        image = images_1.detach().cpu().numpy().transpose((1, 2, 0))

        # writer.append_data((255 * image).astype(np.uint8))


        sils_database, first_sil = appendElement(all_elem=sils_database, elem=np.squeeze(image.astype(np.int8)), first=first_sil)
        # first_sil = save_pny(filename, first_sil, image.astype(np.int8))

        # writer.close()


        params_database, first_param = appendElement(all_elem=params_database, elem=Rt.astype(np.float16), first=first_param)
        # first_param = save_pny(filename, first_param, Rt.astype(np.float16))

    #save database
    np.save('data/test/cubes.npy', cubes_database)
    np.save('data/test/sils.npy', sils_database)
    np.save('data/test/params.npy', params_database)


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


def appendElement(all_elem, elem, first):

    if first:
        all_elem = np.expand_dims(elem, 0)  # create first element array
        first = False
    else:
        elem_exp = np.expand_dims(elem, 0)
        all_elem = np.concatenate((all_elem , elem_exp)) #append element to existing array



    return all_elem, first

def packFiles(path, filename):
    imcount = 0
    first_cube = True
    first_sil = True

    loop = tqdm.tqdm(os.listdir(path))
    for file in loop:
        image = np.array(PIL.Image.open(path + "/" + file))
        size = image.shape

        if len(size) == 3:  # cube
            if first_cube:
                all_cube = np.expand_dims(image, 0)  # cube image
                first_cube = False
            else:
                img_exp = np.expand_dims(image, 0)
                all_cube = np.concatenate((all_cube, img_exp))

        else:  # silhouette
            if first_sil:
                all_sil = np.expand_dims(image, 0)
                first_sil = False
            else:
                img_exp = np.expand_dims(image, 0)
                all_sil = np.concatenate((all_sil, img_exp))

        imcount = imcount + 1

    np.save('cubes.npy', all_cube)
    np.save('Silhouettes.npy', all_sil)
    return imcount


def render_1_image(Obj_Name, params):

    print("creation of a single image")

    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, 'data')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, '{}.obj'.format(Obj_Name)))
    parser.add_argument('-c', '--color_input', type=str, default=os.path.join(data_dir, '{}.mtl'.format(Obj_Name)))
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
    alpha = params[0]
    beta = params[1]
    gamma = params[2]
    x = params[3]
    y = params[4]
    z = params[5]

    R = np.array([alpha, beta, gamma])  # angle in degree param have to change
    t = np.array([x, y, z])  # translation in meter

    Rt = np.concatenate((R, t), axis=None) # create one array of parameter

    # create camera with given parameters
    cam = camera_setttings(R=R, t=t, vert=nb_vertices)

    # create the renderer
    renderer = nr.Renderer(image_size=512, camera_mode='projection',dist_coeffs=None,
                           K=cam.K_vertices, R=cam.R_vertices, t=cam.t_vertices, near=0.1, background_color=[255,255,255],
                           far=1000, orig_size=512, light_direction=[0,-1,0])


    # render an image of the 3d object
    images_1 = renderer(vertices_1, faces_1, textures_1)  # [batch_size, RGB, image_size, image_size]
    image = images_1[0].detach().cpu().numpy()[0].transpose((1, 2, 0)) #float32 from 0 to 255
    im_norm = (255*image) #float32 from 76 to 65025
    final_im = im_norm.astype(np.uint8) #uint8 from 1 to 206


    # create the segmentation of the image

    images_1 = renderer(vertices_1, faces_1, textures_1, mode='silhouettes')  # [batch_size, RGB, image_size, image_size]
    final_sil = images_1.detach().cpu().numpy().transpose((1, 2, 0))

    return final_im, final_sil




