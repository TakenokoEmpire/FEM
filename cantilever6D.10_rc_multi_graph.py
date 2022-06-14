from calendar import c
import enum
import numpy as np
from matplotlib import pyplot as plt
import math
import time
import csv
import csv_write
from tqdm import tqdm
import copy
import cmath
import csv_read

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_yscale('log')
ax2 = fig.add_subplot(212)

# fig = plt.figure()
def csv_to_graph(file_name):
    """
    target:2次元のlist
    log_checker:【未対応】 縦軸をログスケールにするなら"log"
    """
    csv_output = csv_read.csv_read_float(file_name)
    
    ax1.plot(csv_output[2], csv_output[0], label = file_name)
    ax2.plot(csv_output[2], csv_output[1], label = file_name)


# csv_number_list = list(range(68, 76))
csv_number_list = [68,75,76]

for csv_number in csv_number_list:
    csv_to_graph('{}_rc'.format(csv_number))
plt.grid()
plt.legend()
plt.show()

# csv_to_graph([[1,2,3,5,7],[2,4,6,8,10]],[0,4,6,8,9])

# fig = plt.figure()
# # ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# # ax1.set_yscale('log')
# for i in [g11,r11,r12]:
#     rs_amp = i[0]
#     rs_phase = i[1]
#     # plt.plot(freq_axis, rs_amp)
#     plt.plot(freq_axis, rs_phase)
# plt.ylim([-2e-7,2e-7])

# ax1 = fig.add_subplot(211)
# # ax2 = fig.add_subplot(212)
# # ax1.set_yscale('log')
# for i in [g11,r11,r12]:
#     rs_amp = i[0]
#     rs_phase = i[1]
#     plt.plot(freq_axis, rs_amp)
#     # plt.plot(freq_axis, rs_phase)
# plt.ylim([-2e-7,2e-7])