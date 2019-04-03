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

def main():

    obj_name = 'Large_dice'

    nb_im = 2

    creation_database(obj_name, nb_im) #call



if __name__ == '__main__':
    main()

