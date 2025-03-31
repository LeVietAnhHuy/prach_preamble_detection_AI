import math
import json
import numpy as np

class PachOFDMInfo:
    sequenceLen: int
    cyclicPrefixLen: int
    guardPeriodLen: int
    pathProfileLen: int

    ljust_num_chars: int = len('cyclicPrefixLen') + 3

    def getPrachOFDMInfo(self, PrachConfig, RandomAccessConfig):
        with open('./table6_prach.txt', 'r') as f:
             prachConfigTables = json.load(f)

        nfft = 4096
        prachNumerology = math.log2(PrachConfig.subcarrierSpacing / 15)
        kappa = 64
        T_C = 0.509e-6
        default_samplingRate = 15000 * 2048
        samplingRate = (PrachConfig.subcarrierSpacing * 1000) * nfft
        samplingRate_ratio = samplingRate / default_samplingRate

        preamble_format = RandomAccessConfig.preambleFormat.split('/')[0]
        if RandomAccessConfig.L_RA == 839:
            L_RA_column_name = '839'
            multiplier = 1
        elif RandomAccessConfig.L_RA == 139 or RandomAccessConfig.L_RA == 1151 or RandomAccessConfig.L_RA == 571:
            L_RA_column_name = '139,1151,571'
            multiplier = pow(2, prachNumerology*(-1))

        format_index = np.where(np.array(prachConfigTables['Table2'][L_RA_column_name]['Format']) == preamble_format)[0].item()
        N_u = prachConfigTables['Table2'][L_RA_column_name]['N_u'][format_index]*multiplier*samplingRate_ratio
        N_RA_CP = prachConfigTables['Table2'][L_RA_column_name]['N_RA_CP'][format_index]*multiplier*samplingRate_ratio
        N_RA_GP = prachConfigTables['Table2'][L_RA_column_name]['N_RA_GP'][format_index]*multiplier*samplingRate_ratio
        pathProfile = prachConfigTables['Table2'][L_RA_column_name]['PathProfile'][format_index]*multiplier*samplingRate_ratio

        match PrachConfig.subcarrierSpacing:
            case 15 | 30 | 60 | 120 | 480 | 960:
                time_interval = (N_u + N_RA_CP) * kappa * T_C

                if time_interval > 0.5:
                    n = 1
                else:
                    n = 0
            case 1.25 | 5:
                n = 0
            case default:
                raise(f"Expect PrachConfig.subcarrierSpacing ∈ {{1.25, 5, 15, 30, 60, 120, 480, 960}} but got {PrachConfig.subcarrierSpacing}")

        N_RA_CP_l = N_RA_CP + n*16*samplingRate_ratio

        self.sequenceLen = int(N_u)
        self.cyclicPrefixLen = int(N_RA_CP_l)
        self.guardPeriodLen = int(N_RA_GP)
        self.pathProfileLen = int(pathProfile)

    def display_prach_ofdm_info(self):
        print('-------------PRACH OFDM Information-------------')
        print("sequenceLen".ljust(self.ljust_num_chars) + "= " + str(self.sequenceLen))
        print("cyclicPrefixLen".ljust(self.ljust_num_chars) + "= " + str(self.cyclicPrefixLen))
        print("guardPeriodLen".ljust(self.ljust_num_chars) + "= " + str(self.guardPeriodLen))
        print("pathProfileLen".ljust(self.ljust_num_chars) + "= " + str(self.pathProfileLen))




