import json
import numpy as np
from functools import reduce

class RandomAccessConfig:
    preambleFormat: str
    L_RA: int
    prachDuration: int
    prachConfigurationIndex: int
    x: int
    y: int
    subframeNumber: int
    startingSymbol: int
    numPrachSlotsWithinASubframe: int
    numTimeDomainPrachOccasionsWithinAPrachSlot: int
    N_RA_RB_forPusch: int
    k_bar: int

    ljust_num_chars: int = len('numTimeDomainPrachOccasionsWithinAPrachSlot') + 3

    def get_full_random_access_config(self, PrachConfig, CarrierConfig):
        with open('./table6_prach.txt', 'r') as f:
            prachConfigTables = json.load(f)

        match PrachConfig.frequencyRange:
            case 'FR1':
                self.preambleFormat = prachConfigTables['Table6.3.3.2-3']['PreambleFormat'][PrachConfig.prachConfigurationIndex]
                self.prachConfigurationIndex = PrachConfig.prachConfigurationIndex
                self.x = prachConfigTables['Table6.3.3.2-3']['x'][PrachConfig.prachConfigurationIndex]

                self.y = prachConfigTables['Table6.3.3.2-3']['y'][PrachConfig.prachConfigurationIndex]

                self.subframeNumber = prachConfigTables['Table6.3.3.2-3']['SubframeNumber'][PrachConfig.prachConfigurationIndex]
                self.startingSymbol = prachConfigTables['Table6.3.3.2-3']['StartingSymbol'][PrachConfig.prachConfigurationIndex]
                self.numPrachSlotsWithinASubframe = prachConfigTables['Table6.3.3.2-3']['NumberOfPrachSlotsWithinASubframe'][PrachConfig.prachConfigurationIndex]
                self.numTimeDomainPrachOccasionsWithinAPrachSlot = prachConfigTables['Table6.3.3.2-3']['NumberOfTimeDomainPrachOccassionsWithinAPrachSlot'][PrachConfig.prachConfigurationIndex]
            case default:
                raise(f"Expect PrachConfig.frequencyRange ∈ {{FR1, FR2}} but got {PrachConfig.frequencyRange}")

        match self.preambleFormat:
            case '0':
                self.L_RA = 839
                self.prachDuration = 1
            case '1':
                self.L_RA = 839
                self.prachDuration = 2
            case '2':
                self.L_RA = 839
                self.prachDuration = 4
            case default:
                self.L_RA = 139
                self.prachDuration = prachConfigTables['Table6.3.3.2-3']['PrachDuration'][PrachConfig.prachConfigurationIndex]

        L_RA_index_arr = np.where(np.array(prachConfigTables['Table6.3.3.2-1']['L_RA']) == self.L_RA)[0]
        delta_f_RA_forPrach_index_arr = np.where(np.array(prachConfigTables['Table6.3.3.2-1']['delta_f_RA_forPrach']) == PrachConfig.subcarrierSpacing)[0]
        delta_f_forPusch_index_arr = np.where(np.array(prachConfigTables['Table6.3.3.2-1']['delta_f_forPusch']) == CarrierConfig.subcarrierSpacing)[0]
        k_bar_index_arr = reduce(np.intersect1d, (L_RA_index_arr, delta_f_RA_forPrach_index_arr, delta_f_forPusch_index_arr))

        if np.size(k_bar_index_arr) == 0:
            raise(f"No k_bar and N_RA_RB for delta_f_RA_forPrach = {PrachConfig.subcarrierSpacing} and delta_f_forPusch = {CarrierConfig.subcarrierSpacing}")

        k_bar_index = k_bar_index_arr[0]

        self.N_RA_RB_forPusch = prachConfigTables['Table6.3.3.2-1']['N_RA_RB_forPusch'][k_bar_index]
        self.k_bar = prachConfigTables['Table6.3.3.2-1']['k_bar'][k_bar_index]

    def display_random_access_config(self):
        print('-------------Random Access Configuration-------------')
        print("preambleFormat".ljust(self.ljust_num_chars) + "= " + str(self.preambleFormat))
        print("L_RA".ljust(self.ljust_num_chars) + "= " + str(self.L_RA))
        print("prachDuration".ljust(self.ljust_num_chars) + "= " + str(self.prachDuration))
        print("prachConfigurationIndex".ljust(self.ljust_num_chars) + "= " + str(self.prachConfigurationIndex))
        print("x".ljust(self.ljust_num_chars) + "= " + str(self.x))
        print("y".ljust(self.ljust_num_chars) + "= " + str(self.y))

        if type(self.subframeNumber) == int:
            print("subframeNumber".ljust(self.ljust_num_chars) + "= " + str(self.subframeNumber))
        else:
            subframeNumber_str = ', '.join(list(map(str, self.subframeNumber)))
            print("subframeNumber".ljust(self.ljust_num_chars) + "= " + subframeNumber_str)

        print("startingSymbol".ljust(self.ljust_num_chars) + "= " + str(self.startingSymbol))
        print("numPrachSlotsWithinASubframe".ljust(self.ljust_num_chars) + "= " + str(self.numPrachSlotsWithinASubframe))
        print("numTimeDomainPrachOccasionsWithinAPrachSlot".ljust(self.ljust_num_chars) + "= " + str(self.numTimeDomainPrachOccasionsWithinAPrachSlot))
        print("N_RA_RB_forPusch".ljust(self.ljust_num_chars) + "= " + str(self.N_RA_RB_forPusch))
        print("k_bar".ljust(self.ljust_num_chars) + "= " + str(self.k_bar))







