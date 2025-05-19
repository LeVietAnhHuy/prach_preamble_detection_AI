import os
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch

gaf_corr_data_path = 'D:/5G_python/prach_ai/prach_preamble_detection_AI/corr_data_dot_mat/gaf_corr_data.npy'
gaf_corr_vgg_input_data_path = 'D:/5G_python/prach_ai/prach_preamble_detection_AI/corr_data_dot_mat/gaf_corr_vgg_input_data.pt'
image_dir = 'D:/5G_python/prach_ai/prach_preamble_detection_AI/image'

gaf_corr_data = np.load(gaf_corr_data_path)

path_InputData = '/content/gdrive/MyDrive/MachineLearningAndBigData_Semester2Year4/Final_FaceRecognition&Classification/data_img_CollectedFace/WelcomeCamera'
path_OutputData = '/content/gdrive/MyDrive/MachineLearningAndBigData_Semester2Year4/Final_FaceRecognition&Classification/data_img_CollectedFace'

gaf_corr_vgg_input_data = []

to_tensor_vgg_input = transforms.Compose([
    transforms.ToTensor(),           # (3,32,32), values 0–1
    transforms.Resize(224),          # VGG default
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

gaf_corr_vgg_input_data = torch.tensor(gaf_corr_vgg_input_data)

print('Converting Gramian Angular Field correlation data into VGG input format:')
for samp_idx in tqdm(range(gaf_corr_data.shape[0])):
    gaf_corr_vgg_input_data.append(to_tensor_vgg_input(gaf_corr_data[samp_idx, :, :, :]))
print('Done!')

print(f'Saving to {gaf_corr_vgg_input_data_path}...')
torch.save(gaf_corr_vgg_input_data, gaf_corr_vgg_input_data_path)