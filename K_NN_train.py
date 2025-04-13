from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from pyimagesearch.preprocessing import SimplePreprocessor
# from pyimagesearch.datasets import SimpleDatasetLoader
# from imutils import paths
import numpy as np
import os


data_path = 'generated_dataset/pi_63_pci_158_rsi_39_prscs_30_puscs_30_zczc_8_fr_FR1_s_UnrestrictedSet_st_Unpaired_fs_0_snrRange_-40_21_'
test_data_path = 'generated_dataset/pi_63_pci_158_rsi_39_prscs_30_puscs_30_zczc_8_fr_FR1_s_UnrestrictedSet_st_Unpaired_fs_0_snrRange_-30_-29_'

data_name = ['rx_1_freqComb_1_numFrame_1freq_comb.npy',
             'rx_1_freqComb_12_numFrame_1.npy',
             'rx_2_freqComb_3_numFrame_1freq_comb.npy',
             'rx_2_freqComb_12_numFrame_1.npy',
             'rx_4_freqComb_1_numFrame_1freq_comb.npy',
             'rx_4_freqComb_12_numFrame_1.npy',
             'rx_8_freqComb_3_numFrame_1freq_comb.npy',
             'rx_8_freqComb_12_numFrame_1.npy']

test_data_name = ['testing_rx_1_freqComb_12_numFrame_1.npy']

data_file_path = os.path.join(data_path, data_name[7])
test_data_file_path = os.path.join(test_data_path, test_data_name[0])

data = np.load(data_file_path)
test_data = np.load(test_data_file_path)

labels = data[:, -1].astype(int)
data = np.delete(data, -1, 1)
data = np.abs(data)

test_data_labels = test_data[:, -1].astype(int)
test_data = np.delete(test_data, -1, 1)
test_data = np.abs(test_data)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

n_neighbors = 5
le = LabelEncoder()
# # Define a custom distance function for complex numbers
# def complex_distance(z1, z2):
#     return np.sqrt((np.real(z1) - np.real(z2))**2 + (np.imag(z1) - np.imag(z2))**2)

print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(trainX, trainY)

# Create KNN classifier (using Euclidean distance between real and imaginary parts)
# Use 'precomputed' for custom distance
# model = KNeighborsClassifier(n_neighbors=n_neighbors, metric="precomputed")
# x = np.array([[np.real(x), np.imag(x)] for x in trainX])
# # Flatten each complex number into a real valued pair
# model.fit(np.array([[np.real(x), np.imag(x)] for x in trainX]), trainY)
#
# # Calculate distances and predict
# distances = np.array([[complex_distance(test_point, train_point) for train_point in train_data] for test_point in test_data])
# predictions = []
#
# for i in range(len(testX)):
#     model.set_params(n_neighbors=n_neighbors)
#     predictions.append(model.predict(np.array([[np.real(testX[i]),np.imag(testX[i])]]))[0])

print('Accuracy Test:')
print(classification_report(testY, model.predict(testX)))

print('Reliability Test:')
print(classification_report(test_data_labels, model.predict(test_data)))



