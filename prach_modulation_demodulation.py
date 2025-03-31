# prach_modulation_demodulation.py
import math
import numpy as np
import numpy.matlib
from prach_ofdm_info import PachOFDMInfo
from get_ncs_root_cv import get_NCS, get_u, get_C_v
from pyphysim.reference_signals.zadoffchu import calcBaseZC
from numpy.fft import fft, fftshift, ifft

def only_prach_modulation(PrachConfig, CarrierConfig, RandomAccessConfig):
    default_numSample_perSlot = 30720
    BW = 98.28e3
    mui = math.log2(CarrierConfig.subcarrierSpacing / 15)

    num_frame = CarrierConfig.numFrame
    frame_indices = range(num_frame)

    num_subframe = num_frame * 10

    if RandomAccessConfig.preambleFormat == '1':
        num_subframe = round(num_subframe*10*(2 / 3))

    numResourceElement_freqDomain = BW / PrachConfig.subcarrierSpacing

    match RandomAccessConfig.preambleFormat:
        case '0' | '1' | '2':
            numResourceElement_timeDomain = num_subframe
        case '3':
            numResourceElement_timeDomain = num_subframe*4
        case default:
            numResourceElement_per_Slot = 14
            numSlot = num_subframe * pow(2, mui)
            numResourceElement_timeDomain = numResourceElement_per_Slot * numSlot

    numResourceElement_freqDomain = BW / PrachConfig.subcarrierSpacing

    N_RA_RB = RandomAccessConfig.N_RA_RB_forPusch

    N_start_mui_grid = 0
    N_size_mui_grid = CarrierConfig.n_UL_RB
    N_RB_sc = 12


    mui_0 = mui
    N_start_mui_0_grid = N_start_mui_grid
    N_size_mui_0_grid = N_size_mui_grid

    k_0_mui = ((N_start_mui_grid + N_size_mui_grid / 2) * N_RB_sc -
               (N_start_mui_0_grid + N_size_mui_0_grid / 2) * N_RB_sc * pow(2, mui_0 - mui))

    if RandomAccessConfig.L_RA in [571, 1151] and PrachConfig.frequencyRange == "FR1":
        RBSetOffset = 0
        rbsetOffset = RBSetOffset
    else:
        FrequencyIndex = 0
        rbsetOffset = FrequencyIndex * N_RB_sc

    RBOffset = 0

    FrequencyStart = 0

    k_1 = k_0_mui + RBOffset * N_RB_sc - N_size_mui_grid * N_RB_sc / 2 + FrequencyStart * N_RB_sc + rbsetOffset

    K = CarrierConfig.subcarrierSpacing / PrachConfig.subcarrierSpacing

    first = K * k_1 + RandomAccessConfig.k_bar - 1

    zeroFreq = numResourceElement_freqDomain / 2
    PrachStartingResourceElementIndex_freqDomain = math.ceil(first + zeroFreq)

    first = PrachStartingResourceElementIndex_freqDomain

    prach_ofdm_information = PachOFDMInfo()
    prach_ofdm_information.getPrachOFDMInfo(PrachConfig, RandomAccessConfig)
    prach_ofdm_information.display_prach_ofdm_info()

    match RandomAccessConfig.preambleFormat:
        case '0':
            nfft = prach_ofdm_information.sequenceLen
        case '1' | '2' | '3':
            nfft = prach_ofdm_information.sequenceLen / RandomAccessConfig.prachDuration
        case default:
            min_nfft = 128
            while min_nfft < numResourceElement_freqDomain:
                min_nfft *= 2

            nfft = min_nfft

            if PrachConfig.subcarrierSpacing in [30, 120] and RandomAccessConfig.numPrachSlotsWithinASubframe == 1:
                n_RA_slot = 1
            else:
                n_RA_slot = [0, 1]

    match RandomAccessConfig.preambleFormat:
        case '0' | '2':
            subframe_factor = 10
        case '1':
            subframe_factor = 6
        case '3':
            subframe_factor = 40

    N_CS = get_NCS(PrachConfig, RandomAccessConfig)

    u, u_arr_unique = get_u(PrachConfig, RandomAccessConfig, N_CS)

    C_v, C_v_arr = get_C_v(PrachConfig, RandomAccessConfig, N_CS)

    x_u = calcBaseZC(RandomAccessConfig.L_RA, u)

    x_uv = np.roll(x_u, -C_v)

    y_uv = fft(x_uv)

    y_uv_norm = y_uv / math.sqrt(y_uv.size)

    ifftin = np.zeros((1, nfft), dtype=complex)
    ifftin = ifftin.squeeze()

    mapping_index_freq = np.arange(RandomAccessConfig.L_RA) + first

    ifftin[mapping_index_freq] = y_uv_norm

    ifftout = ifft(fftshift(ifftin), nfft)

    seq = np.matlib.repmat(ifftout, 1, RandomAccessConfig.prachDuration)
    seq = seq.squeeze()

    mapping_index_cp =  int(seq.size - prach_ofdm_information.cyclicPrefixLen)

    cp = seq[mapping_index_cp:]

    prach_sequence = np.concatenate((cp, seq), axis=0)

    return prach_sequence

def prach_modulation(PrachConfig, CarrierConfig, RandomAccessConfig):
    default_numSample_perSlot = 30720
    BW = 98.28e3
    mui = math.log2(CarrierConfig.subcarrierSpacing / 15)

    num_frame = CarrierConfig.numFrame
    frame_indices = range(num_frame)

    num_subframe = num_frame * 10

    if RandomAccessConfig.preambleFormat == '1':
        num_subframe = round(num_subframe*10*(2 / 3))

    numResourceElement_freqDomain = BW / PrachConfig.subcarrierSpacing

    match RandomAccessConfig.preambleFormat:
        case '0' | '1' | '2':
            numResourceElement_timeDomain = num_subframe
        case '3':
            numResourceElement_timeDomain = num_subframe*4
        case default:
            numResourceElement_per_Slot = 14
            numSlot = num_subframe * pow(2, mui)
            numResourceElement_timeDomain = numResourceElement_per_Slot * numSlot

    numResourceElement_freqDomain = BW / PrachConfig.subcarrierSpacing

    N_RA_RB = RandomAccessConfig.N_RA_RB_forPusch

    N_start_mui_grid = 0
    N_size_mui_grid = CarrierConfig.n_UL_RB
    N_RB_sc = 12


    mui_0 = mui
    N_start_mui_0_grid = N_start_mui_grid
    N_size_mui_0_grid = N_size_mui_grid

    k_0_mui = ((N_start_mui_grid + N_size_mui_grid / 2) * N_RB_sc -
               (N_start_mui_0_grid + N_size_mui_0_grid / 2) * N_RB_sc * pow(2, mui_0 - mui))

    if RandomAccessConfig.L_RA in [571, 1151] and PrachConfig.frequencyRange == "FR1":
        RBSetOffset = 0
        rbsetOffset = RBSetOffset
    else:
        FrequencyIndex = 0
        rbsetOffset = FrequencyIndex * N_RB_sc

    RBOffset = 0

    FrequencyStart = 0

    k_1 = k_0_mui + RBOffset * N_RB_sc - N_size_mui_grid * N_RB_sc / 2 + FrequencyStart * N_RB_sc + rbsetOffset

    K = CarrierConfig.subcarrierSpacing / PrachConfig.subcarrierSpacing

    first = K * k_1 + RandomAccessConfig.k_bar - 1

    zeroFreq = numResourceElement_freqDomain / 2
    PrachStartingResourceElementIndex_freqDomain = math.ceil(first + zeroFreq)

    prach_ofdm_information = PachOFDMInfo()
    prach_ofdm_information.getPrachOFDMInfo(PrachConfig, RandomAccessConfig)
    # prach_ofdm_information.display_prach_ofdm_info()

    match RandomAccessConfig.preambleFormat:
        case '0':
            nfft = prach_ofdm_information.sequenceLen
        case '1' | '2' | '3':
            nfft = prach_ofdm_information.sequenceLen / RandomAccessConfig.prachDuration
        case default:
            min_nfft = 128
            while min_nfft < numResourceElement_freqDomain:
                min_nfft *= 2

            nfft = min_nfft

            if PrachConfig.subcarrierSpacing in [30, 120] and RandomAccessConfig.numPrachSlotsWithinASubframe == 1:
                n_RA_slot = [1]
            else:
                n_RA_slot = [0, 1]

    match RandomAccessConfig.preambleFormat:
        case '0' | '2':
            subframe_factor = 10
        case '1':
            subframe_factor = 6
        case '3':
            subframe_factor = 40

    match RandomAccessConfig.preambleFormat:
        case '0' | '1' | '2' | '3':
            numSubframe = numResourceElement_timeDomain
            numSample_persubframe = default_numSample_perSlot*(CarrierConfig.subcarrierSpacing / 15)*2
            frame_mod_x =  np.mod(frame_indices, RandomAccessConfig.x)
            frame_contain_prach = np.where(frame_mod_x == RandomAccessConfig.y)[0] - 1

            allframe = range(numSubframe / subframe_factor - 1)
        case default:
            numSample_perSlot = int(default_numSample_perSlot*(PrachConfig.subcarrierSpacing / 15))
            numSlot_perFrame = 10 * (PrachConfig.subcarrierSpacing / 15)
            totalSlot = int(CarrierConfig.numFrame * numSlot_perFrame)

            frame_mod_x =  np.mod(frame_indices, RandomAccessConfig.x)
            frame_contain_prach = np.where(frame_mod_x == RandomAccessConfig.y)[0]

            slot_contain_prach = []
            for frame_index in range(frame_contain_prach.size):
                numSubframe = RandomAccessConfig.subframeNumber
                if type(RandomAccessConfig.subframeNumber) == int:
                    numSubframe = [RandomAccessConfig.subframeNumber]
                for subframe_index in range(len(numSubframe)):
                    for n_RA_slot_index in range(len(n_RA_slot)):
                        slot_contain_prach.append(((frame_contain_prach[frame_index]*10 + numSubframe[subframe_index]) * 2 + n_RA_slot[n_RA_slot_index]))

            start_symbol_contain_prach = RandomAccessConfig.startingSymbol + np.arange(RandomAccessConfig.numTimeDomainPrachOccasionsWithinAPrachSlot)*RandomAccessConfig.prachDuration

    N_CS = get_NCS(PrachConfig, RandomAccessConfig)

    u, u_arr_unique = get_u(PrachConfig, RandomAccessConfig, N_CS)

    C_v, C_v_arr = get_C_v(PrachConfig, RandomAccessConfig, N_CS)

    x_u = calcBaseZC(RandomAccessConfig.L_RA, u)

    x_uv = np.roll(x_u, -C_v)

    y_uv = fft(x_uv)

    y_uv_norm = y_uv / math.sqrt(y_uv.size)

    ifftin = np.zeros((1, nfft), dtype=complex)
    ifftin = ifftin.squeeze()

    mapping_index_freq = np.arange(RandomAccessConfig.L_RA) + PrachStartingResourceElementIndex_freqDomain

    ifftin[mapping_index_freq] = y_uv_norm

    ifftout = ifft(fftshift(ifftin), nfft)

    seq = np.matlib.repmat(ifftout, 1, RandomAccessConfig.prachDuration)
    seq = seq.squeeze()

    mapping_index_cp =  int(seq.size - prach_ofdm_information.cyclicPrefixLen)

    cp = seq[mapping_index_cp:]

    prach_sequence = np.concatenate((cp, seq), axis=0)

    time_domain_signal = np.array([])
    start_mapping_symbol_arr = []
    end_mapping_symbol_arr = []
    match RandomAccessConfig.preambleFormat:
        case '0' | '1' | '2' | '3':
            print('Do later')
        case default:
            for slot_index in range(totalSlot):
                slot = np.zeros((numSample_perSlot, 1), dtype=complex).squeeze()
                if slot_index in slot_contain_prach:
                    for symbol_index in range(14):
                        if symbol_index in start_symbol_contain_prach:
                            starting_sample_symbol = symbol_index*nfft
                            mapping_slot_index = range(starting_sample_symbol, starting_sample_symbol + prach_sequence.size)

                            slot[mapping_slot_index] = prach_sequence

                            start_mapping_symbol = slot_index*slot.size + starting_sample_symbol
                            start_mapping_symbol_arr.append(start_mapping_symbol)

                            end_mapping_symbol = start_mapping_symbol + prach_sequence.size - 1
                            end_mapping_symbol_arr.append(end_mapping_symbol)

                time_domain_signal = np.concatenate((time_domain_signal, slot), axis=0)

            start_mapping_symbol_arr = np.reshape(start_mapping_symbol_arr, (CarrierConfig.numFrame, -1))
            end_mapping_symbol_arr = np.reshape(end_mapping_symbol_arr, (CarrierConfig.numFrame, -1))

    return time_domain_signal, start_mapping_symbol_arr, end_mapping_symbol_arr, PrachStartingResourceElementIndex_freqDomain


def get_preamble_corrlation_windows(PrachConfig, CarrierConfig, RandomAccessConfig, preamble_arr):

    print('')








