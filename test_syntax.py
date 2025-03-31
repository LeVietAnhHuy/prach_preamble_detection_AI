from configuration import PrachConfig, CarrierConfig
import os
from get_random_access_configuration import RandomAccessConfig

prach_config = PrachConfig()

prach_config.preambleIndex = 60
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

start_snr_dB = -40
end_snr_dB = 21
step_snr_dB = 5

num_tx_antennas = 1
num_rx_antennas = 1

generated_data_dir = 'generated_dataset'
config_data_dir = 'pi_' + str(prach_config.preambleIndex) + \
                       '_pci_' + str(prach_config.prachConfigurationIndex) + \
                       '_rsi_' + str(prach_config.rootSequenceIndex) + \
                       '_prscs_' + str(prach_config.subcarrierSpacing) + \
                       '_puscs_' + str(carrier_config.subcarrierSpacing) + \
                       '_zczc_' + str(prach_config.zeroCorrelationZoneConfig) + \
                       '_fr_' + prach_config.frequencyRange + \
                       '_s_' + prach_config.set + \
                       '_st_' + prach_config.spectrumType + \
                       '_fs_' + str(prach_config.frequencyStart) + \
                       '_snrRange_' + str(start_snr_dB) + '_' + str(end_snr_dB - 1) + '_' + str(step_snr_dB)


config_data_path = os.path.join(generated_data_dir, config_data_dir)
if not os.path.exists(config_data_path):
    os.makedirs(config_data_path)

non_preprocessing_data_dir = 'non_preprocessing_dataset'
non_preprocessing_data_path = os.path.join(config_data_path, non_preprocessing_data_dir)
if not os.path.exists(non_preprocessing_data_path):
    os.makedirs(non_preprocessing_data_path)

dataset_name = 'non_preprocessing_rx_' + str(num_rx_antennas) + '_numFrame_' + str(carrier_config.numFrame) + '.npy'
dataset_dir = os.path.join(non_preprocessing_data_path, dataset_name)

print(dataset_dir)