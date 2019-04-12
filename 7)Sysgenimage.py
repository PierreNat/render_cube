
"""
test Resnet 50

"""

import numpy as np
import matplotlib.pyplot as plt

from utils import render_1_image

obj_name = 'Large_dice'
predicted_params = np.array([0, 0, 0, 1, 1, -5])
im, sil = render_1_image(obj_name, predicted_params)  # create the dataset

nb_im = 4
for i in range(nb_im):
    plt.subplot(2, nb_im, i + 1)
    plt.imshow(im)

    plt.subplot(2, nb_im, i + 1 + 4)
    plt.imshow(sil.squeeze())
plt.show()
#  ------------------------------------------------------------------

# nb_im = 1
# for i in range(nb_im):
#     plt.subplot(1, nb_im, i+1)
#     print('computed parameter_{}: '.format(i+1))
#     print(predicted_params[i])
#     print('ground truth parameter_{}: '.format(i+1))
#     print(params[i])
#     # plt.imshow(test_im[i])
#     im, sil = render_1_image(obj_name, predicted_params[i])  # create the dataset
#     plt.imshow(im)
