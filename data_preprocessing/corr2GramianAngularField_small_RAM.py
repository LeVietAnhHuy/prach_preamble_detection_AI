import os
import mat73
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

split_corr_data_dot_npy_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/split_corr_data_dot_npy'
split_corr_data_dot_npy_list = os.listdir(split_corr_data_dot_npy_path)

gaf_corr_data_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/gaf_corr_data'
os.makedirs(gaf_corr_data_path, exist_ok=True)

gaf = GramianAngularField(method='difference', image_size=32)

snr_range = np.arange(-50, -34, 5)
types = ['train', 'test']
# split according to snr
for snr in tqdm(snr_range):
    for type in types:
        split_data_name = 'corr_data_' + str(snr) + 'dB_' + type + '.npy'
        split_data_data = np.load(os.path.join(split_corr_data_dot_npy_path, split_data_name))
        gaf_data = []
        for sample in split_data_data:
            sample_2d = np.array([sample])
            gaf_img = gaf.fit_transform(sample_2d)
            gaf_img = gaf_img[0]
            img_3ch = np.stack([gaf_img, gaf_img, gaf_img], axis=0)

            gaf_data.append(img_3ch)

            del img_3ch

        gaf_label = split_data_data[:, -1]
        gaf_data = np.array(gaf_data)

        gaf_data_name = 'gaf' + 'corr_data_' + str(snr) + '_dB_' + type + '_info.npy'
        gaf_label_name = 'gaf' + 'corr_data_' + str(snr) + '_dB_' + type + '_label.npy'

        print(f'Saving {gaf_data_name}...')
        np.save(os.path.join(gaf_corr_data_path, gaf_data_name), gaf_data)
        print('done!')

        print(f'Saving {gaf_label_name}...')
        np.save(os.path.join(gaf_corr_data_path, gaf_label_name), gaf_label)
        print('done!')

        del gaf_data
        del gaf_label
        del split_data_data

# split the whole folder

# for data_idx in tqdm(range(len(split_corr_data_dot_npy_list))):
#     split_data_data = np.load(os.path.join(split_corr_data_dot_npy_path, split_corr_data_dot_npy_list[data_idx]))
#     gaf_data = []
#     for sample in split_data_data:
#         sample_2d = np.array([sample])
#         gaf_img = gaf.fit_transform(sample_2d)
#         gaf_img = gaf_img[0]
#         img_3ch = np.stack([gaf_img, gaf_img, gaf_img], axis=0)
#
#         gaf_data.append(img_3ch)
#
#         del img_3ch
#
#     gaf_label = split_data_data[:, -1]
#     gaf_data = np.array(gaf_data)
#
#     gaf_data_name = 'gaf' + split_corr_data_dot_npy_list[data_idx].split('.')[0] + '_info.npy'
#     gaf_label_name = 'gaf' + split_corr_data_dot_npy_list[data_idx].split('.')[0] + '_label.npy'
#
#     print(f'Saving {gaf_data_name}...')
#     np.save(os.path.join(gaf_corr_data_path, gaf_data_name), gaf_data)
#     print('done!')
#
#     print(f'Saving {gaf_label_name}...')
#     np.save(os.path.join(gaf_corr_data_path, gaf_label_name), gaf_label)
#     print('done!')
#
#     del gaf_data
#     del gaf_label
#     del split_data_data
