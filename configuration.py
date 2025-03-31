class PrachConfig:
    preambleIndex: int
    prachConfigurationIndex: int
    rootSequenceIndex: int
    subcarrierSpacing: float
    zeroCorrelationZoneConfig: int
    frequencyRange: str
    set: str
    spectrumType: str
    frequencyStart: float
    ljust_num_chars: int = len('prachConfigurationIndex') + 3

    def display_config(self):
        print('-------------PRACH Configuration-------------')
        print("preambleIndex".ljust(self.ljust_num_chars) + "= " + str(self.preambleIndex))
        print("prachConfigurationIndex".ljust(self.ljust_num_chars) + "= " + str(self.prachConfigurationIndex))
        print("rootSequenceIndex".ljust(self.ljust_num_chars) + "= " + str(self.rootSequenceIndex))
        print("subcarrierSpacing".ljust(self.ljust_num_chars) + "= " + str(self.subcarrierSpacing))
        print("zeroCorrelationZoneConfig".ljust(self.ljust_num_chars) + "= " + str(self.zeroCorrelationZoneConfig))
        print("frequencyRange".ljust(self.ljust_num_chars) + "= " + self.frequencyRange)
        print("frequencyRange".ljust(self.ljust_num_chars) + "= " + self.set)
        print("spectrumType".ljust(self.ljust_num_chars) + "= " + self.spectrumType)
        print("frequencyStart".ljust(self.ljust_num_chars) + "= " + str(self.frequencyStart))


class CarrierConfig:
    n_UL_RB: int
    subcarrierSpacing: float
    numFrame: int

    ljust_num_chars: int = len('subcarrierSpacing') + 3

    def display_config(self):
        print('-------------Carrier Configuration-------------')
        print("n_UL_RB".ljust(self.ljust_num_chars) + "= " + str(self.n_UL_RB))
        print("subcarrierSpacing".ljust(self.ljust_num_chars) + "= " + str(self.subcarrierSpacing))
        print("numFrame".ljust(self.ljust_num_chars) + "= " + str(self.numFrame))






