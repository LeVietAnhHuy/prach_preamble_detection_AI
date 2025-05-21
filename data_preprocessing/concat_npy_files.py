import numpy as np
import glob
import os

# corr_data_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/corr_data_dot_mat'
# combined_test_corr_data_name = 'combined_test_corr_data.npy'
#
# print('Starting to concatenate corr test data...')
# corr_test_data_list = glob.glob(os.path.join(corr_data_path, '*test*.npy'))
#
# combined_corr_test_data = np.load(corr_test_data_list[0])
#
# for corr_test_data_idx in range(1, len(corr_test_data_list)):
#     print(f'Concatenate {corr_test_data_list[corr_test_data_idx]}...')
#     corr_test_data = np.load(corr_test_data_list[corr_test_data_idx])
#     combined_corr_test_data = np.concatenate((combined_corr_test_data, corr_test_data), axis=0)
#     del corr_test_data
#     print('done!')
#     print('')
#
# print(f'Saving {combined_test_corr_data_name}...')
# np.save(os.path.join(corr_data_path, combined_test_corr_data_name), combined_corr_test_data)
# print('done!')
# print(f'Done concatenating {combined_test_corr_data_name}\n\n')
#
# del combined_corr_test_data
#####################################################################################################

corr_data_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/corr_data_dot_mat'
combined_train_corr_data_name = 'combined_train_corr_data.npy'

print('Starting to concatenate corr train data...')
corr_train_data_list = glob.glob(os.path.join(corr_data_path, '*train*.npy'))

combined_corr_train_data = np.load(corr_train_data_list[0])

for corr_train_data_idx in range(1, len(corr_train_data_list)):
    print(f'Concatenate {corr_train_data_list[corr_train_data_idx]}...')
    corr_train_data = np.load(corr_train_data_list[corr_train_data_idx])
    combined_corr_train_data = np.concatenate((combined_corr_train_data, corr_train_data), axis=0)
    del corr_train_data
    print('done!')
    print('')

print(f'Saving {combined_train_corr_data_name}...')
np.save(os.path.join(corr_data_path, combined_train_corr_data_name), combined_corr_train_data)
print('done!')
print(f'Done concatenating {combined_train_corr_data_name}\n\n')

del combined_corr_train_data



