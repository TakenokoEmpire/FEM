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


# 使い方
# まず、cantilever6Bで要素のRとGをcsvに出力する
# 例のRC法の式の、それぞれの値とファイル名を対応させる



# .csvは不要
g11_filename = "76_g11"
r11_filename = "53_r11"
r12_filename = "53_r12"
r21_filename = "53_r12"
r22_filename = "53_r11"

# 結果をcsvに保存するならon
csv_save = "on"
save_file_name = g11_filename[:-4]

def csv_read_float(file_name):
    with open('data_box/{}.csv'.format(file_name)) as f:
        reader = csv.reader(f)
        for order, row in enumerate(reader):
            csv_content = [row for row in reader]
    csv_values = csv_content[29:]
    csv_output = []
    for order, row in enumerate(csv_values):
        csv_output.append(np.array([float(r) for r in row]))
    csv_output = np.array(csv_output)
    return csv_output

def frf_mag_phase_to_complex(receptance):
    """振幅・位相 → 実部（0行目）・虚部（1行目）
    引数・戻り値ともに2次元配列"""
    re= list(map(lambda mag, phase: cmath.rect(mag, phase).real, receptance[0], np.radians(receptance[1])))
    im= list(map(lambda mag, phase: cmath.rect(mag, phase).imag, receptance[0], np.radians(receptance[1])))
    return np.array([re, im])

def frf_complex_to_mag_phase(receptance):
    amp = np.sqrt((receptance[0] ** 2) + (receptance[1] ** 2))
    phase = np.arctan2(receptance[1], receptance[0])
    phase = np.degrees(phase)
    return amp, phase

# 各レセプタンスの、振幅と位相を持ってきて、それを実部・虚部に変換する
g11 = frf_mag_phase_to_complex(csv_read_float(g11_filename)[0:2])
r11 = frf_mag_phase_to_complex(csv_read_float(r11_filename)[0:2])
r12 = frf_mag_phase_to_complex(csv_read_float(r12_filename)[0:2])
r21 = frf_mag_phase_to_complex(csv_read_float(r21_filename)[0:2])
r22 = frf_mag_phase_to_complex(csv_read_float(r22_filename)[0:2])

# 周波数軸を取ってくる（全部のレセプタンスで共通のはず）
freq_axis = csv_read_float(g11_filename)[2]

rs = r21 / (r11 - g11) * r12 - r22

rs_amp, rs_phase = frf_complex_to_mag_phase(rs)
# rs_amp = rs[0]
# rs_phase = rs[1]

if csv_save == "on":
    csv_write.csv_write_2d_list([rs_amp, rs_phase, freq_axis], "{}_rc".format(save_file_name), [["g11", g11_filename], ["r11", r11_filename], ["r12", r12_filename], ["r21", r21_filename], ["r22", r22_filename]], [], "null")
    exit()

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_yscale('log')
plt.plot(freq_axis, rs_amp)
ax2 = fig.add_subplot(212)
ax2.set_yticks(np.linspace(-180, 180, 5))
plt.plot(freq_axis, rs_phase)
ax1.grid()
ax2.grid()
plt.show()

