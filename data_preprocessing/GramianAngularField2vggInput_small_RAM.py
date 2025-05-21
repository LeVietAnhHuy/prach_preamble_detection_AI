import os
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch
import glob


gaf_corr_data_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/gaf_corr_data'
gaf_corr_data_list = glob.glob(os.path.join(gaf_corr_data_path, '*info*.npy'))

input_vgg_corr_data_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/input_vgg_gaf_corr_data'
os.makedirs(input_vgg_corr_data_path, exist_ok=True)

for data_idx in tqdm(range(len(gaf_corr_data_list))):
    input_vgg_gaf_corr_data_name = 'input_vgg_' + gaf_corr_data_list[data_idx].split('.')[0] + '.pt'
    gaf_corr_data = np.load(os.path.join(gaf_corr_data_path, gaf_corr_data_list[data_idx]))
    input_vgg_gaf_cor_data = []

    to_tensor_vgg_input = transforms.Compose([
        transforms.Resize(224),          # VGG default
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print('Converting Gramian Angular Field correlation data into VGG input format:')
    for samp_idx in tqdm(range(gaf_corr_data.shape[0])):
        input_vgg_gaf_cor_data.append(to_tensor_vgg_input(torch.squeeze(torch.from_numpy(gaf_corr_data[samp_idx, :, :, :]), 0)))

    input_vgg_gaf_cor_data = torch.tensor(input_vgg_gaf_cor_data)
    print('Done!')

    print(f'Saving to {input_vgg_gaf_corr_data_name}...')
    torch.save(os.path.join(input_vgg_corr_data_path, input_vgg_gaf_corr_data_name), input_vgg_gaf_cor_data)