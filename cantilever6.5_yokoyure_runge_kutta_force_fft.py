from calendar import c
import numpy as np
from matplotlib import pyplot as plt
import math
import time
import csv_write
import fft
import frf
from tqdm import tqdm


# 改善点
# 治具を複数個おけるようにする
# 複数の条件（治具の位置とか）で同時に計算し、結果を重ねて表示できるようにする
# スイープ加振に対応させる
# 　→runge_kuttaに、加振力を対応させる

# 注意点
# 出力されるフーリエ変換されたグラフは、変位をフーリエ変換しただけのもの（加振力で割っていないので、FRFではない）
# 左側（点0）の一端固定で解析（片持ち梁）

# パラメータを設定

# 基本設定
dt = 10e-7 # [s] ルンゲクッタ法なら、qty_x=10でdt=20e-7、14で10、20で5、44で1、100で0.2。自由端でも、同じ条件でいけそう。
fs = 0.1/dt # サンプリング周波数[Hz] ヤング率を補正するなら値を変更する（2e11で500k、2e5で5k？）
simu_time = 100e-3 # [s] 最小の固有振動数の波1つ分が7.5ms
qty_x = 14 # x方向の要素の数 33以上だと一気に遅くなる。
# 測定点の位置 list型で複数位置指定可能 ※番号は0スタートなので注意
measurement_positions = [0, 2*int(qty_x/3), -2*int(qty_x/3), -2] # 点0の変位、点0の角度、点1の変位、…のList。番号に注意。　※一端固定の場合、点0は固定端ではなくその右隣の点である点に注意
# FFTする点の位置
fft_positions = [-2] # 点0の変位、点0の角度、点1の変位、…のList。番号に注意。　※一端固定の場合、点0は固定端ではなくその右隣の点である点に注意

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
damping_ratio = 0.005 # 本当の減衰比ではない
c_diag = 2 * (m_total * k_total) ** 0.5 * damping_ratio


# 外力の位置 ※番号は0スタートなので注意
f_position = -2 # 点0の変位、点0の角度、点1の変位、…のList。番号に注意
# 外力のタイプ
f_wave_type = "impulse"
# 外力の大きさ
f_mag = 100 # impulseなら[N・s]
f_freq = 3000

# 治具関連
k_jig = k_total * 0  # 治具の剛性
# 何番目の位置にジグを入れるか ※番号は0スタートなので注意
jig_position = 2 * int(qty_x * 0.9) # 点0の変位、点0の角度、点1の変位、…のList。番号に注意

# アニメーションの再生速度
animation_speed = int(fs / 2000) # int(fs / [アニメーションのHz])→指定Hzで再生可能
# ここまで入力


# 計算を楽にするための色々なもの
# vec_length_kari: 教科書P133の、上のマトリクスの大きさ
# vec_length: 教科書P133の、下のマトリクスの大きさ。両端自由ならvec_length_kariと同等、片持ち梁ならvec_length_kari-2
# ※m_mat_kariとm_matの関係も、同様の関係（cとkも同様）
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


# エラー警告や情報表示
# 発散判定
print("dt限界: {:.1f} %".format(qty_x ** 2 * dt / 0.0002 * 100))
if qty_x ** 2 * dt > 0.00020005:
    print("値が発散する可能性があります（dtが過大）")
# fs過小警告
if fs < 10000:
    print("Fsが{} Hzです（過小では？）".format(fs))
else:
    print("Fs: {:,.0f} Hz".format(fs))
# アニメーション間隔
print("アニメーション間隔: {:,.0f} Hz".format(fs / animation_speed))
print("アニメーション再生時間: {:.0f} ~ {:.0f} s".format(sampling_length / animation_speed / 5, sampling_length / animation_speed))


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

# 片持ち梁の場合、固定端の位置や角度が動かないようにする（教科書P133の処理、左二列と上二行を削除）
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



def runge_kutta_v2(M, C, K, dt, simu_time, simu_length, sampling_length, sampling_interval, x_0, v_0):
    # 初期値、変数の定義
    n = len(M)                    # 自由度
    t = np.linspace(0, simu_time, simu_length)       # 時間データ
    t_sampled = np.linspace(0, simu_time, sampling_length)
    # sampling_interval = int(1 / (dt * fs))
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
        x_sampled = np.zeros((sampling_length, n))
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
    x_rk_even = np.zeros((sampling_length, int(n/2)))
    for i in range(int(n/2)):
        x_rk_even[:, i] = x_rk[:, i*2]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(int(n/2)):
        try:
            ax.plot(t_sampled, x_rk_even[:, i], label=str(i + 1))
        except ValueError: # dt=0.2e-7, 0.1e-7のときに起きるエラーの対処。0.25e-7とか0.125e-7だと発生しない。
            if len(t_sampled) -1 == len(x_rk_even):
                print("warning 1")
                np.delete(t_sampled, -1)
                # ax.plot(t_sampled, x_rk_even[:, i], label=str(i + 1))
            else:
                print("error 1")
                exit
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


x_box = runge_kutta_v2(m_mat, c_mat, k_mat, dt, simu_time, simu_length, sampling_length, sampling_interval, state_vec[:vec_length].ravel(), state_vec[-vec_length:].ravel())


csv_write.csv_write_2d_list(x_box, "elements{}_measurement_pos{}_jig_pos{}_force_pos{}_dt{}_simu_time{}".format(qty_x, measurement_positions, jig_position, f_position, dt, simu_time))

# 力のFFT
# dt<サンプリング間隔なので、力もサンプリング間隔でFFTしないと変位/力ができなくなる。
# 一方、インパルス加振はdt秒しか加振しないので、dt間隔でなければ正しく力がFFTできない。
def averaging_sampling(signal, sampling_interval, sampling_length):
    """値の平均値をとりながらサンプリングする（例:5Hz→1Hzにサンプリングするとき、[5,0,0,0,0]→[1]にする）
    インパクト加振のFFTを想定
    入出力のsignalはnp.array"""
    signal_length = len(signal)
    if sampling_interval * sampling_length < signal_length:
        print("averaging_sample: 末尾の信号が切り捨てられています（長さ{}）".format(signal_length - sampling_interval * sampling_length))
    sampled_signal = []
    for i in range(sampling_length):
        start_point = i * sampling_interval
        end_point = (i+1) * sampling_interval
        if end_point <= signal_length:
            sampled_signal.append(np.average(signal[start_point:end_point]))
        elif start_point < signal_length:
            sampled_signal.append(np.average(signal[start_point:]))
        else:
            sampled_signal.append(0)
    return np.array(sampled_signal)

f_wave_sampled = averaging_sampling(f_wave_shape, sampling_interval, sampling_length)

for i in range(len(measurement_positions)):
    # plt.plot(yokojiku, signal_box[0])
    plt.plot(yokojiku, x_box[:,measurement_positions[i]])
    plt.legend(measurement_positions)
# plt.show()

# # FFTグラフの表示
# for position_num in fft_positions:
#     print("showing point {}".format(position_num))
#     fft.main(x_box[:,position_num], 1/fs, "k_jig = {}, jig_position = {} / {}".format(int(k_jig), jig_position, qty_x * 2))

# FRFグラフの表示
t_sampled = np.linspace(0, simu_time, sampling_length)
for position_num in fft_positions:
    frf.frf_general(f_wave_sampled, x_box[:,position_num], 1/fs, t_sampled)


def animation(shape_list, dt = 1, speed = 1, x_min = 0, x_max = 100):
    """speed: int: インターバル。コマを間引くことによって1コマあたりの経過秒数を増やして、描画速度を上げる。"""
    # 1. 必要なモジュールの読み込み
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.animation import ArtistAnimation
    
    # 2.グラフ領域の作成
    fig, ax = plt.subplots()
    
    # 3. グラフ要素のリスト（artists）作成
    artists = []
    for i in range(len(shape_list[1, :])):
        if i%speed == 0:
            x = np.linspace(x_min, x_max, len(shape_list[:, i]))
            y = shape_list[:, i]
            artist = ax.plot(x, y,"blue")
            plt.vlines(total_length * jig_position / vec_length, np.amin(shape_list) * 0.8, np.amax(shape_list) * 0.8) # 治具の位置に縦線
            ax.text(jig_position / vec_length, 0.02, "Jig", ha='center', transform=ax.transAxes) # 治具の注釈
            plt.hlines(0, 0, total_length, linewidth = 0.1) # y=0の基準線
            title = ax.text(0.5, 1.01, 'k_jig = {}, jig_position = {} / {} , t = {:.4f}'.format(int(k_jig), jig_position, qty_x * 2, i * dt), ha='center', va='bottom', transform=ax.transAxes, fontsize='large') # 追加
            artists.append(artist + [title])
        
    # 4. アニメーション化
    anim = ArtistAnimation(fig, artists)
    plt.show()

x_box_even = []
for i in range(int(len(x_box[1,:]) / 2)):
    x_box_even.append(x_box[:, i*2])
x_box_even = np.array(x_box_even)
animation(x_box_even, 1/fs, animation_speed, 0, total_length)