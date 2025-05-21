import os
import mat73
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
import numpy as np
import torch
from torchvision import transforms

corr_data_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/corr_data_dot_mat'
combined_corr_data_name = ['combined_train_corr_data.npy', 'combined_test_corr_data.npy']

gaf_data_name = ['combined_gafd_train_corr_data.npy', 'combined_gafd_test_corr_data.npy']
gaf_label_name = ['combined_gafd_train_corr_data_label.npy', 'combined_gafd_test_corr_data_label.npy']
gaf = GramianAngularField(method='difference', image_size=32)

for data_idx in range(len(combined_corr_data_name)):
    combined_corr_data = np.load(os.path.join(corr_data_path, combined_corr_data_name[data_idx]))
    gaf_data = []
    for sample in combined_corr_data:
        sample_2d = np.array([sample])
        gaf_img = gaf.fit_transform(sample_2d)
        gaf_img = gaf_img[0]
        img_3ch = np.stack([gaf_img, gaf_img, gaf_img], axis=0)

        gaf_data.append(img_3ch)

        del img_3ch

    gaf_label = combined_corr_data[:, -1]

    gaf_data = np.array(gaf_data)

    print(f'Saving {gaf_data_name[data_idx]}...')
    np.save(gaf_data_name[data_idx], gaf_data)
    print('done!')

    print(f'Saving {gaf_label_name[data_idx]}...')
    np.save(gaf_label_name[data_idx], gaf_label)
    print('done!')

    del gaf_data
    del gaf_label
    del combined_corr_data
