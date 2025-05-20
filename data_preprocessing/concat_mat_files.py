from scipy.io import savemat
import mat73
import numpy as np
import os

corr_data_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/corr_data_dot_mat'
corr_data_list = os.listdir(corr_data_path)

combined_corr_data = []

for corr_data in corr_data_list:
    data_dict = mat73.loadmat(os.path.join(corr_data_path, corr_data))
    X = data_dict['B']

    combined_corr_data.append(data_dict['B'])

combined_corr_data = np.concatenate(combined_corr_data, axis=0)

savemat("combined_corr_data.mat", {'B': combined_corr_data})
