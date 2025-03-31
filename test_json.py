import json
import numpy as np
from functools import reduce

with open('./table6_prach.txt', 'r') as f:
  data = json.load(f)

# Output: {'name': 'Bob', 'languages': ['English', 'French']}
print(data['Table6.3.3.2-3']['PreambleFormat'][0])

L_RA_index = np.where(np.array(data['Table6.3.3.2-1']['L_RA']) == 139)[0]
delta_f_RA_forPrach_index = np.where(np.array(data['Table6.3.3.2-1']['delta_f_RA_forPrach']) == 30)[0]
delta_f_forPusch_index = np.where(np.array(data['Table6.3.3.2-1']['delta_f_forPusch']) == 30)[0]

print(L_RA_index)
print(delta_f_RA_forPrach_index)
print(delta_f_forPusch_index)

k_bar_index = reduce(np.intersect1d, (L_RA_index, delta_f_RA_forPrach_index, delta_f_forPusch_index))

print(k_bar_index)

y = data['Table6.3.3.2-3']['y']

print(f"y len = {len(y)}\n")

numberTimeDomainPrachOccasionsWithinPrachSlot_column = [
    -1] * 67 + [6, 6, 6, 6, 3, 3] + [6] * 5 +  \
    [3, 6, 6, 6, 3, 6, 6, 6] + [3] * 6 + [1, 1] + \
    [3] * 6 + [1, 3, 3, 3, 3, 1, 3, 3, 3, 1] + \
    [2, 2, 2, 1, 1] + [2] * 8 + [1, 2, 2, 2, 2, 1, 2, 2, 2, 1] + \
    [6, 6, 6, 3, 6, 6, 3, 6, 6, 6, 3, 6] + [1] * 24 + [6] * 6 + \
    [3, 3, 6, 6, 6, 3, 6, 6, 6, 3, 6, 6, 6, 3] + [2] * 6 + \
    [1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1] + \
    [6, 3, 3, 6, 6, 6, 6, 3, 6, 6, 6, 3, 6, 6, 3, 3,
     2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3] + \
    [2] * 16 + [-1] * 7

print(numberTimeDomainPrachOccasionsWithinPrachSlot_column[168])
print(numberTimeDomainPrachOccasionsWithinPrachSlot_column)

numberPrachSlotWithinSubframe_column = [
    -1] * 67 + [2, 2] + [1] * 5 + [2, 2] + \
    [1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2] + \
    [1] * 4 + [2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2] + \
    [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2] + [1] * 6 + \
    [2, 2, 1, 1, 2, 1, 1, 1, 2, 2] + [1] * 5 + \
    [2, 2] + [1] * 7 + \
    [2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2] + \
    [1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2] + \
    [1] * 6 + [2, 2, 2, 1, 1, 2, 1, 1, 2] + [1] * 5 + \
    [2, 2, 2, 1, 1, 2, 1, 1, 1, 2] + [1] * 6 + \
    [2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1] + [-1] * 7

print(numberPrachSlotWithinSubframe_column[166])
print(numberPrachSlotWithinSubframe_column)





