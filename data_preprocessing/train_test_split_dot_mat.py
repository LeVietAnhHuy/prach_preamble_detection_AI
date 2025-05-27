import mat73
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

corr_data_dot_mat_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/corr_data_dot_mat'
corr_data_list = os.listdir(corr_data_dot_mat_path)

split_data_dot_npy_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/split_corr_data_dot_npy'
os.makedirs(split_data_dot_npy_path, exist_ok=True)

test_size = 0.25
snr_range = np.arange(-50, -34, 5)
# split according to snr
for snr in tqdm(snr_range):
    corr_data_name = 'corr_data_' + str(snr) + 'dB.mat'
    print(f'Splitting {corr_data_name} into train and test data...')
    data_dict = mat73.loadmat(os.path.join(corr_data_dot_mat_path, corr_data_name))
    train_corr_data, test_corr_data = train_test_split(data_dict['B'],
                                                       test_size=test_size,
                                                       train_size=(1 - test_size),
                                                       random_state=42,
                                                       shuffle=True)

    del data_dict
    train_corr_name = 'corr_data_' + str(snr) + 'dB_train.npy'
    test_corr_name = 'corr_data_' + str(snr) + 'dB_test.npy'

    print(f'Saving {train_corr_name}...')
    np.save(os.path.join(split_data_dot_npy_path, train_corr_name), train_corr_data)
    print('done!')

    print(f'Saving {test_corr_name}...')
    np.save(os.path.join(split_data_dot_npy_path, test_corr_name), test_corr_data)
    print('done!')

    del train_corr_data, test_corr_data
    print('done splitting!')


# split the whole folder
# for corr_data_idx in tqdm(range(len(corr_data_list))):
#     print(f'Splitting {corr_data_list[corr_data_idx]} into train and test data...')
#     data_dict = mat73.loadmat(os.path.join(corr_data_dot_mat_path, corr_data_list[corr_data_idx]))
#     train_corr_data, test_corr_data = train_test_split(data_dict['B'],
#                                                        test_size=test_size,
#                                                        train_size=(1 - test_size),
#                                                        random_state=42,
#                                                        shuffle=True)
#
#     del data_dict
#     train_corr_name = corr_data_list[corr_data_idx].split('.')[0] + '_train.npy'
#     test_corr_name = corr_data_list[corr_data_idx].split('.')[0] + '_test.npy'
#
#     print(f'Saving {train_corr_name}...')
#     np.save(os.path.join(split_data_dot_npy_path, train_corr_name), train_corr_data)
#     print('done!')
#
#     print(f'Saving {test_corr_name}...')
#     np.save(os.path.join(split_data_dot_npy_path, test_corr_name), test_corr_data)
#     print('done!')
#
#     del train_corr_data, test_corr_data
#     print('done splitting!')

