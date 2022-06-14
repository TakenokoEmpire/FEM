from calendar import c
import numpy as np
from matplotlib import pyplot as plt
import math
import time
import csv_write
import fft
from tqdm import tqdm


# 改善点
# ★治具をKマトリクスに組み込む
# スイープ加振に対応させる
# →runge_kuttaに、加振力を対応させる

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
second_moment = depth * height ** 3 / 3 # 断面二次モーメント [m^4] 直方体の場合の値

# 解析条件設定
fix_type = "left" # "left"で、左端（点0）固定の片持ち梁になる。
qty_x = 25 # x方向の要素の数 32を超えると異常に遅くなる。32にするならdtを1e-7以下にする
ele_len = total_length / qty_x # [m] 要素の長さ。全体の長さではなく要素の長さ
# 要素の質量
m_element = density * ele_len * cross_section
# 要素と壁の間、要素同士、治具、の剛性
k_internal = cross_section / ele_len * young
# k_wall = k_internal # 壁（固定端）と片持はりの接点の剛性

m_total = m_element *  qty_x
k_total = 3 * young * second_moment / total_length ** 3
#  固有振動数を推定　https://www.jsme.or.jp/sed/guide/dynamics5.pdf
lambda_const_list = [1.875, 4.694, 7.855] # 片持ち梁でない場合は値を変える
for lambda_const in lambda_const_list:
    estimated_natural_freq = (lambda_const / total_length) ** 2 / (2 * math.pi) * (young * second_moment / density / cross_section) ** 0.5
    print("estimated natural freq: {}".format(estimated_natural_freq))

# 減衰関係。
damping_ratio = 0.007 # 本当の減衰比ではない 
c_diag = int(2 * (m_total * k_total) ** 0.5 * damping_ratio) 
# 
dt = 2e-7 # [s] ルンゲクッタ法なら、qty_x=25で2e-7、16で5e-7
fs = 0.01/dt # サンプリング周波数[Hz] ヤング率を補正するなら値を変更する（2e11で500k、2e5で5k？）
simu_time = 0.1 # [s] ヤング率を補正するなら値を変更する（2e11で0.05、2e5なら1以上）
# 測定点の位置 list型で複数位置指定可能 ※番号は0スタートなので注意
measurement_positions = [0, 2*int(qty_x/3), -2*int(qty_x/3), -2] # 点0の変位、点0の角度、点1の変位、…のList。番号に注意。　※一端固定の場合、点0は固定端ではなくその右隣の点である点に注意
# FFTする点の位置
fft_positions = [-2] # 点0の変位、点0の角度、点1の変位、…のList。番号に注意。　※一端固定の場合、点0は固定端ではなくその右隣の点である点に注意

# 外力の位置 ※番号は0スタートなので注意
f_position = -2 # 点0の変位、点0の角度、点1の変位、…のList。番号に注意
# 外力のタイプ
f_wave_type = "impulse"
# 外力の大きさ
f_mag = 100 # impulseなら[N・s]
f_freq = 3000

# 治具関連
k_jig = k_total * 10  # 治具の剛性
# 何番目の位置にジグを入れるか ※番号は0スタートなので注意
jig_position = 2 * int(qty_x * 0.5) # 点0の変位、点0の角度、点1の変位、…のList。番号に注意

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
sampling_length = int(simu_length / sampling_interval)
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
signal_box = np.zeros((1, sampling_length))
yokojiku = np.linspace(0, simu_time , sampling_length)

# k,mマトリクス
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

# 治具の剛性を追加
k_mat[jig_position, jig_position]  += k_jig

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



def runge_kutta_v2(M, C, K, dt, fs, t_max, x_0, v_0):
    # 初期値、変数の定義
    n = len(M)                    # 自由度
    t = np.arange(0, t_max, dt)       # 時間データ
    t_sampled = np.arange(0, t_max, 1/fs)
    sampling_interval = int(1 / (dt * fs))
    M_inv = np.linalg.inv(M)       # 質量マトリクスの逆行列

    # 運動方程式の関数化
    def func(v, x, M_inv, C, K):
        '''運動方程式
        M_invにはMの逆行列を入れる'''
        # 各種変数定義
        return - M_inv @ C @ v.T - M_inv @ K @ x.T

    # ルンゲクッタ法による数値解析
    def runge_kutta_method(n, t, sampling_interval, dt, x_0, v_0, func):
        '''ルンゲクッタ法'''
        x_sampled = np.zeros((int(len(t) / sampling_interval), n))
        x = np.zeros((len(t), n))
        v = np.zeros((len(t), n))
        for i in tqdm(range(len(t) - 1)):
            if i == 0:
                x[i, :] = x_0
                v[i, :] = v_0
                
            k1_x = v[i, :] * dt
            k1_v = func(v[i, :], x[i, :], M_inv, C, K) * dt
            
            k2_x = (v[i, :] + k1_v / 2) * dt
            k2_v = func(v[i, :] + k1_v / 2, x[i, :] + k1_x / 2, M_inv, C, K) * dt
            
            k3_x = (v[i, :] + k2_v / 2) * dt
            k3_v = func(v[i, :] + k2_v / 2, x[i, :] + k2_x / 2, M_inv, C, K) * dt
            
            k4_x = (v[i, :] + k3_v) * dt
            k4_v = func(v[i, :] + k3_v, x[i, :] + k3_x, M_inv, C, K) * dt
            
            x[i + 1, :] = x[i, :] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
            v[i + 1, :] = v[i, :] + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
            
            if i%sampling_interval == 0:
                x_sampled[int(i / sampling_interval), :] = x[i + 1, :]
        return x_sampled

    # プロット(各自由度の初期間隔を2としてプロットする)
    x_rk = runge_kutta_method(n, t, sampling_interval, dt, x_0, v_0, func)
    # x_rk_trimmed = 
    x_rk_even = np.zeros((int(len(t) / sampling_interval), int(n/2)))
    for i in range(int(n/2)):
        x_rk_even[:, i] = x_rk[:, i*2]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(int(n/2)):
        ax.plot(t_sampled, x_rk_even[:, i], label=str(i + 1))
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.legend(loc='best')
    # plt.show()
    
    return x_rk


# シミュレーション開始
print("simulating...")
# 現状のrunge_kutta_v2は加振力を入れられない（初期変位・初期速度のみ）ので、まず加振力を変位・速度に変換するために、dt秒だけオイラー法でシミュレーションする。
for i in range(1000):
    f_vec = np.zeros((vec_length, 1))
    f_vec[f_position, 0] += f_wave_shape[0, i]
    state_vec_dxdt = lambda z: (np.dot(a_mat, z) + np.dot(b_mat, f_vec)) * (dt / 1000) 
    # tqdm.write("x_free_end: {}".format(state_vec[vec_length-2]))
    # tqdm.write("dxdt{}".format(state_vec_dxdt(state_vec)[vec_length-2]))
    # dxdt = state_vec_dxdt(state_vec)[vec_length-2]
    # state_vec = runge_kutta(state_vec, dt, state_vec_dxdt)
    state_vec += state_vec_dxdt(state_vec)
    # if i % sampling_interval == 0:
    #     x_box[:, int(i / sampling_interval)] = state_vec[0:vec_length, 0]
    #     signal_box[0, int(i / sampling_interval)] = f_wave_shape[0, i]
    # if i % (print_interval_length) == 0:
    #     # 発散してないか確認するために、進捗1%毎に自由端の変位をプロット
    #     tqdm.write("x_free_end: {}".format(state_vec[vec_length-2]))
    #     # progress = (i + 1) / simu_length
    #     # print("{:.1f}% finished".format(progress * 100))
    #     # time_passed = time.time() - time_box
    #     # time_estimation = (time_passed / progress / 60) * (1 - progress)
    #     # print("{:.1f} min to finish".format(time_estimation))


x_box = runge_kutta_v2(m_mat, c_mat, k_mat, dt, fs, simu_time, state_vec[:vec_length].ravel(), state_vec[-vec_length:].ravel())


csv_write.csv_write_2d_list(x_box, "elements{}_measurement_pos{}_jig_pos{}_force_pos{}_dt{}_simu_time{}".format(qty_x, measurement_positions, jig_position, f_position, dt, simu_time))

for i in range(len(measurement_positions)):
    # plt.plot(yokojiku, signal_box[0])
    plt.plot(yokojiku, x_box[:,measurement_positions[i]])
    plt.legend(measurement_positions)
# plt.show()

for position_num in fft_positions:
    print("showing point {}".format(position_num))
    fft.main(x_box[:,position_num], 1/fs, "k_jig = {}, jig_position = {} / {}".format(int(k_jig), jig_position, qty_x * 2))
