import os
import mat73
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
import numpy as np
import torch
from torchvision import transforms

corr_data_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/corr_data_dot_mat'

combined_corr_data = 'combined_corr_data.mat'

gaf_data_name = 'combined_gafd_corr_data.npy'
gaf_label_name = 'combined_gafd_corr_data_label.npy'
gaf_data = []
data_dict = mat73.loadmat(os.path.join(corr_data_path, combined_corr_data))

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
