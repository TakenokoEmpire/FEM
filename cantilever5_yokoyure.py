import numpy as np
from matplotlib import pyplot as plt
import math
import time
import csv_write
import fft
from tqdm import tqdm


# 注意点
# 出力されるフーリエ変換されたグラフは、変位をフーリエ変換しただけのもの（加振力で割っていないので、FRFではない）
# 左側（点0）の一端固定で解析（片持ち梁）

# パラメータを設定
# 解析対象のパラメータ
total_length = 250e-3 # 全体の長さ[m]
height = 5e-3 # [m] 曲げ方向の長さに相当
depth = 20e-3 # [m]
cross_section = height * depth # [m^2] 対象の断面積
young = 2.05e+11 # [Pa] 対象のヤング率。GpaではなくPaなので注意 そのままだとkが大きくなりすぎて解析できないので1e-6倍している
density = 7850 # [kg/m^3]
second_moment = depth * height ** 3 / 3 # 断面二次モーメント [m^4]

# 解析条件設定
fix_type = "left" # "left"で、左端（点0）固定の片持ち梁になる。
qty_x = 10 # x方向の要素の数
ele_len = total_length / qty_x # [m] 要素の長さ。全体の長さではなく要素の長さ
# 要素の質量
m_element = density * ele_len * cross_section
# 要素と壁の間、要素同士、治具、の剛性
k_internal = cross_section / ele_len * young
# k_wall = k_internal # 壁（固定端）と片持はりの接点の剛性
k_jig_mijissou = 0
# 公式からの固有振動数の推定。「片持ち梁」の場合に限る。
m_total = m_element *  qty_x
k_total = 3 * young * second_moment / total_length ** 3
estimated_natural_freq = 1 / (2 * math.pi * (m_total / k_total) ** 0.5)
print("estimated natural freq: {}".format(estimated_natural_freq))
# 減衰関係。
damping_ratio = 0.3
c_diag = int(2 * (m_total * k_total) ** 0.5 * damping_ratio) 
# 
dt = 1e-10 # [s] 10^-3だとちゃんとできない（発散する）、サイン加振なら10^-5でおk（ダメかも？）、インパルス加振なら10^-7.5以上必要　→減衰を入れないとこうなるっぽい
fs = 1e6 # サンプリング周波数[Hz] ヤング率を補正するなら値を変更する（2e11で500k、2e5で5k？）
simu_time = 0.1 # [s] ヤング率を補正するなら値を変更する（2e11で0.05、2e5なら1以上）
# 測定点の位置 list型で複数位置指定可能 ※番号は0スタートなので注意
measurement_positions = [0, 6, 12, -2] # 点0の変位、点0の角度、点1の変位、…のList。番号に注意。　※一端固定の場合、点0は固定端ではなくその右隣の点である点に注意
# FFTする点の位置
fft_positions = [0, -2] # 点0の変位、点0の角度、点1の変位、…のList。番号に注意。　※一端固定の場合、点0は固定端ではなくその右隣の点である点に注意
# 何番目の位置にジグを入れるか ※番号は0スタートなので注意
jig_position_mijissou = 0 # 点0の変位、点0の角度、点1の変位、…のList。番号に注意
# 外力の位置 ※番号は0スタートなので注意
f_position = -2 # 点0の変位、点0の角度、点1の変位、…のList。番号に注意
# 外力のタイプ
f_wave_type = "impulse"
# 外力の大きさ
f_mag = 100 # impulseなら[N・s]
f_freq = 1000

# ここまで入力


# 計算を楽にするための色々なもの
vec_length_kari = (qty_x + 1) * 2
if fix_type == "left":
    vec_length = vec_length_kari - 2
else:
    vec_length = vec_length_kari
simu_length = int(simu_time / dt)
print_interval_length = int(simu_length / 100)
sampling_interval = int(1 / (dt * fs))
sampling_length = int(simu_length / sampling_interval * 1.05)
f_impulse = (np.zeros((vec_length, 1)))
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
        print("calculating wave shape...")
        for i in tqdm(range(wave_length)):
            wave[0, i] += math.sin(dt * i * freq * 2 * math.pi) * mag
    elif wave_type == "sine_sweep":
        print("calculating wave shape...")
        for i in tqdm(range(wave_length)):
            wave[0, i] += math.sin((dt * i) ** 2 *  freq / simu_time * math.pi) * mag
    else:
        print("wave_type name error")
        exit()
    return wave

f_wave_shape = wave_shape(f_wave_type, f_mag, dt, simu_length, f_freq)

# 時間変化を記録する箱
# x_box = np.linspace(0, 0, simu_length)
x_box = np.zeros((vec_length, sampling_length))
signal_box = np.zeros((1, sampling_length))
yokojiku = np.linspace(0, simu_time * 1.05, sampling_length)

# k,mマトリクス
# density =1
# cross_section = 1
# element_length = 10
# young = 1000
# second_moment = 1
# まず、振動の教科書P132に出てくる要素の質量・剛性マトリクスを設定する
m_mat_element_func = lambda x: x*density*cross_section*ele_len/420
m_mat_element = np.array([[156, 22*ele_len, 54, -13*ele_len], [22*ele_len, 4*ele_len**2, 13*ele_len, -3*ele_len**2], [54, 13*ele_len, 156, -22*ele_len], [-13*ele_len, -3*ele_len**2, -22*ele_len, 4*ele_len**2]])
m_mat_element = m_mat_element_func(m_mat_element)
k_mat_element_func = lambda x: x*young*second_moment/ele_len**3
k_mat_element = np.array([[12, 6*ele_len, -12, 6*ele_len], [6*ele_len, 4*ele_len**2, -6*ele_len, 2*ele_len**2], [-12,-6*ele_len, 12, -6*ele_len], [6*ele_len, 2*ele_len**2, -6*ele_len, 4*ele_len**2]])
k_mat_element = k_mat_element_func(k_mat_element)


# 質量・剛性マトリクスを生成

# 要素の位置ベクトルと速度ベクトル、及びそれらを縦に連ねた状態ベクトル
x_vec = np.zeros((vec_length, 1))
v_vec = np.zeros((vec_length, 1))
state_vec = np.concatenate((x_vec, v_vec), 0)

# 質量マトリクスを作成
m_mat_kari = np.zeros((vec_length_kari, vec_length_kari))
for i in range(qty_x):
    for j1 in range(4):
        for j2 in range(4):
            m_mat_kari[i*2 + j1, i*2 + j2] += m_mat_element[j1, j2]

# 減衰マトリクスを作成
c_mat_kari = np.zeros((vec_length_kari, vec_length_kari))
# 角度（奇数番目）には減衰を設定せず、位置（偶数番目）に対してのみ減衰を設定する
for i in range(vec_length_kari):
    if i%2 == 0:
        c_mat_kari[i, i] += c_diag

# 剛性マトリクスを作成
k_mat_kari = np.zeros((vec_length_kari, vec_length_kari))
for i in range(qty_x):
    for j1 in range(4):
        for j2 in range(4):
            k_mat_kari[i*2 + j1, i*2 + j2] += k_mat_element[j1, j2]
# 剛性マトリクスに治具の剛性を追加
# k_mat[jig_position, jig_position] += k_jig_mijissou

# 片持ち梁の場合、固定端の位置や角度が動かないようにする
if fix_type == "left":
    m_mat = np.zeros((vec_length,vec_length))
    c_mat = np.zeros((vec_length,vec_length))
    k_mat = np.zeros((vec_length,vec_length))
    for i in range(vec_length):
        for j in range(vec_length):
            m_mat[i, j] = m_mat_kari[i+2, j+2]
            c_mat[i, j] = c_mat_kari[i+2, j+2]
            k_mat[i, j] = k_mat_kari[i+2, j+2]
else:
    m_mat = m_mat_kari
    c_mat = c_mat_kari
    k_mat = k_mat_kari
#     for i in range(vec_length):
#         for j in range(vec_length):
#             m_mat[i, 0] = 0
#             m_mat[i, 1] = 0
#             m_mat[0, j] = 0
#             m_mat[1, j] = 0
#             c_mat[i, 0] = 0
#             c_mat[i, 1] = 0
#             c_mat[0, j] = 0
#             c_mat[1, j] = 0
#             k_mat[i, 0] = 0
#             k_mat[i, 1] = 0
#             k_mat[0, j] = 0
#             k_mat[1, j] = 0

a_mat_1 = np.zeros((vec_length, vec_length))
a_mat_2 = np.identity(vec_length)
a_mat_3 = -np.dot(np.linalg.inv(m_mat), k_mat)
# a_mat_3[3,3] += -100
a_mat_4 = -np.dot(np.linalg.inv(m_mat), c_mat)
a_mat = np.concatenate((
    np.concatenate((a_mat_1, a_mat_2), 1),
    np.concatenate((a_mat_3, a_mat_4), 1)),
0)

b_mat_1 = np.zeros((vec_length, vec_length))
b_mat_2 = np.linalg.inv(m_mat)
b_mat = np.concatenate((b_mat_1, b_mat_2), 0)
    
# 位置と速度を計算
# time_box = time.time()
print("simulating...")
for i in tqdm(range(simu_length)):
    f_vec = np.zeros((vec_length, 1))
    f_vec[f_position, 0] += f_wave_shape[0, i]
    state_vec += (np.dot(a_mat, state_vec) + np.dot(b_mat, f_vec)) * dt # インパルス加振なので、最初以外の外力は0
    if i % sampling_interval == 0:
        x_box[:, int(i / sampling_interval)] = state_vec[0:vec_length, 0]
        signal_box[0, int(i / sampling_interval)] = f_wave_shape[0, i]
    if i % (print_interval_length) == 0:
        # 発散してないか確認するために、進捗1%毎に自由端の変位をプロット
        tqdm.write("x_free_end: {}".format(state_vec[vec_length-2]))
        # progress = (i + 1) / simu_length
        # print("{:.1f}% finished".format(progress * 100))
        # time_passed = time.time() - time_box
        # time_estimation = (time_passed / progress / 60) * (1 - progress)
        # print("{:.1f} min to finish".format(time_estimation))


csv_write.csv_write_2d_list(x_box, "elements{}_measurement_pos{}_jig_pos{}_force_pos{}_dt{}_simu_time{}".format(qty_x, measurement_positions, jig_position_mijissou, f_position, dt, simu_time))

for i in range(len(measurement_positions)):
    # plt.plot(yokojiku, signal_box[0])
    plt.plot(yokojiku, x_box[measurement_positions[i]])
    plt.legend(measurement_positions)
plt.show()

for position_num in fft_positions:
    print("showing point {}".format(position_num))
    fft.main(x_box[position_num], 1/fs)


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