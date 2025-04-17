import numpy as np
import os

from configuration import PrachConfig, CarrierConfig
from get_random_access_configuration import RandomAccessConfig
from get_ncs_root_cv import get_NCS, get_u, get_C_v
from prach_modulation_demodulation import prach_modulation

generated_data_dir = 'generated_dataset'
config_data_dir = 'corr_antenna_gain_combining_dataset'
config_data_path = os.path.join(generated_data_dir, config_data_dir)

dataset_names = ['rx_2_corr_selectionComb_freqComb.npy']

dataset_dir = os.path.join(config_data_path, dataset_names[-1])

data = np.load(dataset_dir)
num_samples = data.shape[0]

Y = data[:, -1].astype(int)
X = np.delete(data, -1, 1)

prach_config = PrachConfig()
##########
prach_config.preambleIndex = 10
##########

prach_config.prachConfigurationIndex = 158
prach_config.rootSequenceIndex = 39
prach_config.subcarrierSpacing = 30
prach_config.zeroCorrelationZoneConfig = 8
prach_config.frequencyRange = 'FR1'
prach_config.set = 'UnrestrictedSet'
prach_config.spectrumType = 'Unpaired'
prach_config.frequencyStart = 0

carrier_config = CarrierConfig()

carrier_config.n_UL_RB = 273
carrier_config.subcarrierSpacing = 30
carrier_config.numFrame = 1

random_access_config = RandomAccessConfig()
random_access_config.get_full_random_access_config(prach_config, carrier_config)

N_CS = get_NCS(prach_config, random_access_config)

_, u_arr_unique = get_u(prach_config, random_access_config, N_CS)

_, C_v_arr = get_C_v(prach_config, random_access_config, N_CS)

num_preamble_per_seq = np.zeros(len(u_arr_unique))

for u_index in range(len(u_arr_unique)):
    if u_index < len(u_arr_unique) - 1:
        num_preamble_per_seq[u_index] = C_v_arr.size
    else:
        num_preamble_per_seq[u_index] = 64 - u_index * C_v_arr.size

for sample_idx in range(num_samples):







