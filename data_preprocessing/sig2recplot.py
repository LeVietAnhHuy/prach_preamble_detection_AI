import os
import mat73
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot


corr_data_path = 'D:/5G_python/prach_ai/prach_preamble_detection_AI/corr_data_dot_mat'
image_dir = 'D:/5G_python/prach_ai/prach_preamble_detection_AI/image'
# os.makedirs(corr_data_path, exist_ok=True)

corr_data_list = os.listdir(corr_data_path)

for corr_data in corr_data_list:

    data_dict = mat73.loadmat(os.path.join(corr_data_path, corr_data))


    X = data_dict['B']

    plt.plot(X[300, :-1])
    full_image_X = os.path.join(image_dir, 'X.png')
    plt.savefig(full_image_X, dpi=300)
    print(f"Plot saved as '{full_image_X}'")
    plt.close()

    print('')
