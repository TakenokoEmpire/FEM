import numpy as np
from matplotlib import pyplot as plt
import math
import time
import csv_write
import fft


# 振動の教科書のP70の例題4.2を再現
# 減衰比は0.2にしてある


# パラメータを設定
# 解析対象のパラメータ
total_length = 1 # 全体の長さ[m]
cross_section = 1e-3 # [m^2] 対象の断面積
young = 2.05e+11 # [Pa] 対象のヤング率。GpaではなくPaなので注意
density = 7850 # [kg/m^3]
# 解析条件設定
qty_x = 2 # x方向の要素の数
element_length = total_length / qty_x # [m] 要素の長さ。全体の長さではなく要素の長さ
# 要素の質量
# m_element = density * element_length * cross_section
m_element = 1
# 要素と壁の間、要素同士、治具、の剛性
k_internal = 8e+4
k_wall = 8e+5 # 壁（固定端）と片持はりの接点の剛性
k_jig = 0
# 減衰関係。とりあえず適当な値。
damping_ratio = 0.2
c_per_m_diag = int(2 * (m_element * k_internal) ** 0.5 * damping_ratio) 
# 
dt = 5e-7 # [s] 10^-3だとちゃんとできない（無限増幅する）、サイン加振なら10^-5でおk（ダメかも？）、インパルス加振なら10^-7.5以上必要　→減衰を入れないとこうなるっぽい
fs = 5000 # サンプリング周波数[Hz]
simu_time = 1 # [s]
# 測定点の位置 list型で複数位置指定可能 ※番号は0スタートなので注意
measurement_positions = [0, 1]
# 何番目の位置にジグを入れるか ※番号は0スタートなので注意
jig_position = 1
# 外力の位置 ※番号は0スタートなので注意
f_position = 1
# 外力のタイプ
f_wave_type = "impulse"
# 外力の大きさ
f_mag = 0.1 # impulseなら[N・s]
f_freq = 2000

# ここまで入力

# 計算を楽にするための色々なもの
simu_length = int(simu_time / dt)
sampling_interval = int(1 / (dt * fs))
sampling_length = int(simu_length / sampling_interval * 1.05)
f_impulse = (np.zeros((qty_x, 1)))
f_impulse[f_position, 0] += f_mag / dt
# zero_mat = np.zeros((qty_x, qty_x))
# i_mat = np.identity(qty_x)

# 外力関連
def wave_shape(wave_type, mag, dt, wave_length, freq = None):
    """
    Args:
        wave_type (str): 波形。impulse,sine,sine_sweep(未対応)
        mag (float): 振幅（inpulseの場合は力積）
        dt(float): dt
    """
    wave = np.zeros((1, wave_length))
    if wave_type == "impulse":
        wave[0, 0] += mag / dt
    elif wave_type == "sine":
        for i in range(wave_length):
            wave[0, i] += math.sin(dt * i * freq * 2 * math.pi) * mag
    elif wave_type == "sine_sweep":
        for i in range(wave_length):
            wave[0, i] += math.sin((dt * i) ** 2 *  freq / simu_time * math.pi) * mag
    else:
        print("wave_type name error")
        exit()
    return wave

f_wave_shape = wave_shape(f_wave_type, f_mag, dt, simu_length, f_freq)

# 時間変化を記録する箱
# x_box = np.linspace(0, 0, simu_length)
x_box = np.zeros((qty_x, sampling_length))
signal_box = np.zeros((1, sampling_length))
yokojiku = np.linspace(0, simu_time * 1.05, sampling_length)

# k1,k2,...,knのベクトル
k_vec = np.ones((qty_x, 1))* k_internal
k_vec[0, 0] = k_wall

# 要素の位置ベクトルと速度ベクトル、及びそれらを縦に連ねた状態ベクトル
x_vec = np.zeros((qty_x, 1))
v_vec = np.zeros((qty_x, 1))
state_vec = np.concatenate((x_vec, v_vec), 0)

# 質量マトリクスを作成
# m_mat = np.zeros((qty_x, qty_x)) + np.identity(qty_x)*m_element
m_mat = np.zeros((qty_x, qty_x))
m_mat[0,0] += 160
m_mat[1,1] += 2000

# 減衰マトリクスを作成
c_mat = np.zeros((qty_x,qty_x))
c_mat += m_mat ** 0.5 * c_per_m_diag

# 剛性マトリクスを作成
k_mat = np.zeros((qty_x, qty_x))
k_mat[0, 0] += k_vec[0, 0]
for i in range(qty_x-1):
    k_mat[i, i] += k_vec[i+1, 0]
    k_mat[i, i+1] -= k_vec[i+1, 0]
    k_mat[i+1, i] -= k_vec[i+1, 0]
    k_mat[i+1, i+1] += k_vec[i+1, 0]
# 剛性マトリクスに治具の剛性を追加
k_mat[jig_position, jig_position] += k_jig


a_mat_1 = np.zeros((qty_x, qty_x))
a_mat_2 = np.identity(qty_x)
a_mat_3 = -np.dot(np.linalg.inv(m_mat), k_mat)
# a_mat_3[3,3] += -100
a_mat_4 = -np.dot(np.linalg.inv(m_mat), c_mat)
a_mat = np.concatenate((
    np.concatenate((a_mat_1, a_mat_2), 1),
    np.concatenate((a_mat_3, a_mat_4), 1)),
0)

b_mat_1 = np.zeros((qty_x, qty_x))
b_mat_2 = np.linalg.inv(m_mat)
b_mat = np.concatenate((b_mat_1, b_mat_2), 0)
    
# 位置と速度を計算
time_box = time.time()
for i in range(simu_length):
    f_vec = np.zeros((qty_x, 1))
    f_vec[f_position, 0] += f_wave_shape[0, i]
    state_vec += (np.dot(a_mat, state_vec) + np.dot(b_mat, f_vec)) * dt # インパルス加振なので、最初以外の外力は0
    if i % sampling_interval == 0:
        x_box[:, int(i / sampling_interval)] = state_vec[0:qty_x, 0]
        signal_box[0, int(i / sampling_interval)] = f_wave_shape[0, i]
    if i % (sampling_interval * 100) == 0:
        progress = (i + 1) / simu_length
        print("{:.1f}% finished".format(progress * 100))
        time_passed = time.time() - time_box
        time_estimation = (time_passed / progress / 60) * (1 - progress)
        print("{:.1f} min to finish".format(time_estimation))


csv_write.csv_write_2d_list(x_box, "elements{}_measurement_pos{}_jig_pos{}_force_pos{}_dt{}_simu_time{}".format(qty_x, measurement_positions, jig_position, f_position, dt, simu_time))

for i in range(len(measurement_positions)):
    plt.plot(yokojiku, signal_box[0])
    plt.plot(yokojiku, x_box[measurement_positions[i]])
    plt.legend(measurement_positions)
plt.show()

a = x_box[measurement_positions[0]]
fft.main(a, 1/fs)

b = x_box[measurement_positions[1]]
fft.main(b, 1/fs)




# print(a_mat_1)
# print(a_mat_2)
# print(a_mat_3)
# print(a_mat_4)
# print(a_mat)
# print(b_mat)
    



# # 位置と速度を計算
# state_vec += (np.dot(a_mat, state_vec) + np.dot(b_mat, f_impulse)) * dt # インパルス加振
# x_box[0] = state_vec[measurement_position, 0]

# for i in range(simu_length-1):
#     state_vec += (np.dot(a_mat, state_vec) + np.dot(b_mat, np.zeros((qty_x, 1)))) * dt # インパルス加振なので、最初以外の外力は0
#     x_box[i+1] = state_vec[measurement_position, 0]




# # グラフ化のために自由度軸を作成
# dof = np.linspace(0, len(sort_index), len(sort_index)+1)

# # ここからグラフ描画
# # フォントの種類とサイズを設定する。
# plt.rcParams['font.size'] = 14
# plt.rcParams['font.family'] = 'Times New Roman'

# # 目盛を内側にする。
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'

# # グラフの上下左右に目盛線を付ける。
# fig = plt.figure(figsize=(6,3))
# ax1 = fig.add_subplot(111)
# ax1.yaxis.set_ticks_position('both')
# ax1.xaxis.set_ticks_position('both')

# # 軸のラベルを設定する。
# ax1.set_xlabel('Degree of freedom')
# ax1.set_ylabel('Eigenvector')

# # # データの範囲と刻み目盛を明示する。
# # ax1.set_xticks(np.arange(0, element_qty_x+1, 1))
# # ax1.set_yticks(np.arange(-2, 2, 0.5))
# # ax1.set_xlim(0, element_qty_x)
# # ax1.set_ylim(-1, 1)

# ax1.plot(x_box)

# fig.tight_layout()
# plt.legend()

# # グラフを表示する。
# plt.show()
# plt.close()




# # 質量マトリクスの逆行列を計算
# M_inv = np.linalg.inv(M)

# print("M Matrix")
# print(M)
# print("K Matrix")
# print(K)

# # 固有値と固有ベクトルを計算
# omega, v = np.linalg.eig(np.dot(M_inv, K))

# # 固有値の順番を昇順にソート
# omega_sort = np.sort(omega)

# # 固有値のソート時のインデックスを取得
# # ⇒固有ベクトルと対応させるため
# sort_index = np.argsort(omega)

# # 固有値に対応する固有ベクトルをソート
# v_sort = []
# for i in range(len(sort_index)):
#     v_sort.append(v.T[sort_index[i]])
# v_sort = np.array(v_sort)

# # 結果をコンソールに表示
# print("Natural freq")
# print(np.sqrt(omega_sort))
# print("Mode shape")
# print(v_sort)

# # グラフ化のために自由度軸を作成
# dof = np.linspace(0, len(sort_index), len(sort_index)+1)

# # ここからグラフ描画
# # フォントの種類とサイズを設定する。
# plt.rcParams['font.size'] = 14
# plt.rcParams['font.family'] = 'Times New Roman'

# # 目盛を内側にする。
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'

# # グラフの上下左右に目盛線を付ける。
# fig = plt.figure(figsize=(6,3))
# ax1 = fig.add_subplot(111)
# ax1.yaxis.set_ticks_position('both')
# ax1.xaxis.set_ticks_position('both')

# # 軸のラベルを設定する。
# ax1.set_xlabel('Degree of freedom')
# ax1.set_ylabel('Eigenvector')

# # データの範囲と刻み目盛を明示する。
# ax1.set_xticks(np.arange(0, element_qty_x+1, 1))
# ax1.set_yticks(np.arange(-2, 2, 0.5))
# ax1.set_xlim(0, element_qty_x)
# ax1.set_ylim(-1, 1)

# # データプロット 固有ベクトルの形を見る
# for i in range(len(sort_index)):
#     eigen_vector = np.concatenate([[0], v_sort[i]])
#     ax1.plot(dof, eigen_vector,lw=1, marker='o')

# fig.tight_layout()
# plt.legend()

# # グラフを表示する。
# plt.show()
# plt.close()