import numpy as np
from numpy.fft import fft, fftshift, ifft
from pyphysim.reference_signals.zadoffchu import calcBaseZC
import time

from configuration import PrachConfig, CarrierConfig
from get_random_access_configuration import RandomAccessConfig
from get_ncs_root_cv import get_NCS, get_u, get_C_v
from prach_ofdm_info import PachOFDMInfo
from prach_modulation_demodulation import prach_modulation
from fading import TdlChannel, TdlChannelProfile
from fading_generators import JakesSampleGenerator

from pyphysim.channels.fading import TdlChannel as ppTdlChannel, TdlChannelProfile as ppTdlChannelProfile
from pyphysim.channels.fading_generators import JakesSampleGenerator as ppJakesSampleGenerator
# from get_snr import snr_db_from_received

tap_powers_dB = np.array([-6.9, 0, -7.7, -2.5, -2.4, -9.9, -8.0, -6.6, -7.1, -13.0, -14.2, -16.0])

tap_delays = np.array([0, 65, 70, 190, 195, 200, 240, 325, 520, 1045, 1510, 2595])

num_tx_antennas = 1
num_rx_antennas = 16

print('')
print('-----------------MIMO Configuration-----------------')
print(f"num_tx_antennas = {num_tx_antennas}")
print(f"num_rx_antennas = {num_rx_antennas}")

bandwidth = 100e6  # in Hetz
Fd = 100  # Doppler frequency (in Hz)

start_snr_dB = -30
end_snr_dB = 16
step_snr_dB = 5

print('')
print('-----------------Noise-----------------')
print("TDLC300-100")
print(f"AWGN: {start_snr_dB}dB -> {end_snr_dB}dB, step = {step_snr_dB}")
print('')

all_x_u_arr = []
all_preamble_arr = []
all_preamble_start_mapping_symbol_arr = []
all_preamble_end_mapping_symbol_arr = []

preamble_index_range = range(0, 64)

for preindex in preamble_index_range:
    prach_config = PrachConfig()

    ##########
    prach_config.preambleIndex = preindex
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

    u, u_arr_unique = get_u(prach_config, random_access_config, N_CS)

    C_v, C_v_arr = get_C_v(prach_config, random_access_config, N_CS)

    x_u = calcBaseZC(random_access_config.L_RA, u)
    all_x_u_arr.append(x_u)

    x_uv = np.roll(x_u, -C_v)

    y_uv = fft(x_uv)

    prach_ofdm_information = PachOFDMInfo()
    prach_ofdm_information.getPrachOFDMInfo(prach_config, random_access_config)

    (time_domain_signal,
     start_mapping_symbol_arr,
     end_mapping_symbol_arr,
     PrachStartingResourceElementIndex_freqDomain) = prach_modulation(prach_config,
                                                                      carrier_config,
                                                                      random_access_config)
    all_preamble_arr.append(time_domain_signal)
    all_preamble_start_mapping_symbol_arr.append(start_mapping_symbol_arr)
    all_preamble_end_mapping_symbol_arr.append(end_mapping_symbol_arr)

Ts = all_preamble_arr[0].size  # Sampling interval

jakesObj = JakesSampleGenerator(Fd, Ts, L=np.size(tap_powers_dB))
tdlChanlelProfile = TdlChannelProfile(tap_powers_dB=tap_powers_dB, tap_delays=tap_delays)
tdlChannel = TdlChannel(jakesObj, channel_profile=tdlChanlelProfile)

ppjakesObj = ppJakesSampleGenerator(Fd, Ts, L=np.size(tap_powers_dB))
pptdlChanlelProfile = ppTdlChannelProfile(tap_powers_dB=tap_powers_dB, tap_delays=tap_delays)
pptdlChannel = ppTdlChannel(ppjakesObj, channel_profile=pptdlChanlelProfile)

tdlChannel.set_num_antennas(num_rx_antennas=num_rx_antennas, num_tx_antennas=num_tx_antennas)
pptdlChannel.set_num_antennas(num_rx_antennas=num_rx_antennas, num_tx_antennas=num_tx_antennas)

preamble_index = 0

ppstart = time.time()
ppreceived_test_signal = pptdlChannel.corrupt_data(all_preamble_arr[preamble_index])
ppend = time.time()

start = time.time()
received_test_signal = tdlChannel.corrupt_data(all_preamble_arr[preamble_index])
end = time.time()

print(f"ppExecution time: {ppend - ppstart:.20f} seconds")
print(f"Execution time: {end - start:.20f} seconds")

# print(ppreceived_test_signal[0, 972686:972689])
# print('---------------------------------------')
# print(received_test_signal[:, 0, 972686:972689])

print(np.array_equal(ppreceived_test_signal, received_test_signal))
print('')