from scipy.io import savemat
import mat73
import numpy as np
import os

corr_data_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/corr_data_dot_mat'
corr_data_list = os.listdir(corr_data_path)

print(f'Appending {corr_data_list[0]}...')
data_dict = mat73.loadmat(os.path.join(corr_data_path, corr_data_list[0]))
combined_corr_data = data_dict['B']
print('done!')

del data_dict

for corr_data_idx in range(1, 2):
    print(f'Appending {corr_data_list[corr_data_idx]}...')
    data_dict = mat73.loadmat(os.path.join(corr_data_path, corr_data_list[corr_data_idx]))
    combined_corr_data = np.concatenate((combined_corr_data, data_dict['B']), axis=0)

    del data_dict

    print('done!')

print('Saving final file...')
savemat("combined_corr_data.mat", {'B': combined_corr_data})
print('DONE!')
