# get_ncs_root_cv.py
import json
import numpy as np
import numpy.matlib
import math

def get_NCS(PrachConfig, RandomAccessConfig):
    with open('./table6_prach.txt', 'r') as f:
        prachConfigTables = json.load(f)

    if (PrachConfig.zeroCorrelationZoneConfig > 15) or (PrachConfig.zeroCorrelationZoneConfig < 0):
        raise(f"Expect zeroCorrelationZoneConfig must be in range [0, 15], but got {PrachConfig.zeroCorrelationZoneConfig}")

    match RandomAccessConfig.L_RA:
        case 839:
            match PrachConfig.subcarrierSpacing:
                case 1.25:
                    return prachConfigTables['Table6.3.3.1-5'][PrachConfig.set][PrachConfig.zeroCorrelationZoneConfig]
                case 5:
                    return prachConfigTables['Table6.3.3.1-6'][PrachConfig.set][PrachConfig.zeroCorrelationZoneConfig]
                case default:
                    raise(f"Expect PrachConfig.subcarrierSpacing ∈ {{1.25, 5}} for L_RA = 839 but got {PrachConfig.subcarrierSpacing}")
        case 139 | 571 | 1151:
            return prachConfigTables['Table6.3.3.1-7']['L_RA=' + str(RandomAccessConfig.L_RA)][PrachConfig.zeroCorrelationZoneConfig]
        case default:
            raise(f"Expect RandomAccessConfig.L_RA ∈ {{139, 839, 571, 1151}} but got {RandomAccessConfig.L_RA}")

def get_u(PrachConfig, RandomAccessConfig, N_CS):
    with open('./table6_prach.txt', 'r') as f:
        prachConfigTables = json.load(f)

    match RandomAccessConfig.L_RA:
        case 839:
            u_table_name = 'Table6.3.3.1-3'
        case 139:
            u_table_name = 'Table6.3.3.1-4'
        case 1151:
            u_table_name = 'Table6.3.3.1-4A'
        case 571:
            u_table_name = 'Table6.3.3.1-4B'
        case default:
            raise (f"Expect RandomAccessConfig.L_RA ∈ {{139, 839, 571, 1151}} but got {RandomAccessConfig.L_RA}")

    if N_CS == 0:
        v = np.arange(64)
    else:
        v = np.arange(math.floor(RandomAccessConfig.L_RA / N_CS))

    u_arr_unique = np.zeros((v.size,), dtype=int)
    if pow(v.size, 2) < 64:
        u_arr_unique = np.zeros((math.ceil(64 / v.size),), dtype=int)

    for u_arr_index in range(u_arr_unique.size):
        root_sequence_index = (u_arr_index + PrachConfig.rootSequenceIndex) % len(prachConfigTables[u_table_name])

        if root_sequence_index == 0:
            root_sequence_index = len(prachConfigTables[u_table_name])

        u_arr_unique[u_arr_index] = prachConfigTables[u_table_name][root_sequence_index]

    if N_CS == 0:
        u_preamble_arr = u_arr_unique
    else:
        u_preamble_arr = np.matlib.repmat(u_arr_unique, v.size, 1)
        u_preamble_arr = np.reshape(u_preamble_arr, (1, u_arr_unique.size * v.size), order='F')

    u_preamble_arr = np.squeeze(u_preamble_arr)
    u = u_preamble_arr[PrachConfig.preambleIndex]
    u_preamble_arr = u_preamble_arr[:64]
    # u_arr_unique = np.unique(u_preamble_arr)
    u_arr_unique_indexes = np.unique(u_preamble_arr, return_index=True)[1]
    u_arr_unique = [u_preamble_arr[index] for index in sorted(u_arr_unique_indexes)]

    return u, u_arr_unique


def get_C_v(PrachConfig, RandomAccessConfig, N_CS):
    match PrachConfig.set:
        case 'UnrestrictedSet':
            if N_CS == 0:
                C_v = 0
                C_v_arr = np.array([])
                return C_v, C_v_arr
            else:
                v = np.arange(math.floor(RandomAccessConfig.L_RA / N_CS))
                C_v_arr = np.multiply(v, N_CS)
        case 'RestrictedSetTypeA':
            print('Do it later')
        case default:
            raise(f'Expect PrachConfig.set ∈ {{UnrestrictedSet, RestrictedSetTypeA, RestrictedSetTypeA}} but got {PrachConfig.set}')

    C_v_preamble_arr = np.zeros((PrachConfig.preambleIndex + 1,), dtype=int)

    C_v_arr_index = 0

    for idx in range(C_v_preamble_arr.size):
        if C_v_arr_index > C_v_arr.size - 1:
            C_v_arr_index = 0

        C_v_preamble_arr[idx] = C_v_arr[C_v_arr_index]
        C_v_arr_index += 1

    C_v = C_v_preamble_arr[PrachConfig.preambleIndex]

    return C_v, C_v_arr






