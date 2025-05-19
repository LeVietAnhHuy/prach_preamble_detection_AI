import os
import mat73
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
import numpy as np
import torch
from torchvision import transforms


corr_data_path = 'D:/5G_python/prach_ai/prach_preamble_detection_AI/corr_data_dot_mat'
image_dir = 'D:/5G_python/prach_ai/prach_preamble_detection_AI/image'
# os.makedirs(corr_data_path, exist_ok=True)

corr_data_list = os.listdir(corr_data_path)

for corr_data in corr_data_list:
    gaf_data_name =  corr_data.split(".")[0] + '_gafd.npy'
    gaf_label_name = corr_data.split(".")[0] + '_gafd_label.npy'
    gaf_data = []
    data_dict = mat73.loadmat(os.path.join(corr_data_path, corr_data))
    X = data_dict['B']
    gaf = GramianAngularField(method='difference', image_size=32)

    for x in X:
        x2d = np.array([x])
        gaf_img = gaf.fit_transform(x2d)
        gaf_img = gaf_img[0]
        img_3ch = np.stack([gaf_img, gaf_img, gaf_img], axis=0)

        gaf_data.append(img_3ch)

        del img_3ch

    gaf_label = X[:, -1]

    gaf_data = np.array(gaf_data)

    print(f'Saving {gaf_data_name}...')
    np.save(gaf_data_name, gaf_data)
    print('done!')

    print(f'Saving {gaf_label_name}...')
    np.save(gaf_label_name, gaf_label)
    print('done!')

    del gaf_data
    del gaf_label
    del X

