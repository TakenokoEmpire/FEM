import numpy as np
from matplotlib import pyplot as plt

# シミュレーションモデル
def model():
    # 質量・減衰・剛性の集中定数を設定する
    m1 = 0.5
    m2 = 0.5
    k1 = 30000
    k2 = 20000

    # 質量マトリクスと剛性マトリクスを作成
    M = np.array([[m1, 0],
                  [0, m2]])
    K = np.array([[k1 + k2, -k2],
                  [-k2, k2]])
    return M, K

# Rayleigh dampingのパラメータを決定して減衰マトリクスを生成する関数
def generate_rayleigh_damping(f1, f2, zeta1, zeta2, M, K):
    # 指定周波数を角振動数に変換
    w1 = 2 * np.pi * f1
    w2 = 2 * np.pi * f2

    # alphaとbetaを使って減衰行列を計算
    alpha = (2 * w1 * w2 * (zeta2 * w1 - zeta1 * w2)) / (w1 ** 2 - w2 ** 2)
    beta = (2 * (zeta1 * w1 - zeta2 * w2)) / (w1 ** 2 - w2 ** 2)
    C = np.array(alpha * M + beta * K)

    return C

# 周波数応答解析を計算する関数
def frequency_response_analysis(M, K, C, freq_l, freq_u, step, exc_node, res_node):
    steps = 2 * np.pi * np.arange(freq_l, freq_u, step)                 # 角振動数軸
    excitation = np.ones_like(steps)                                    # 加振力[N]の周波数波形
    n_dofs = len(M)                                                     # 自由度の総数
    F = np.zeros((len(steps), n_dofs))                                  # 周波数毎の外力ベクトルを初期化

    # 角振動数分解能毎に計算するループ
    X = []
    for i in range(len(steps)):
        F[i][exc_node] = excitation[i]                                  # 外力ベクトルの更新
        matrix = (-steps[i] ** 2) * M + (1j * steps[i] * C) + K         # 逆行列にする前の行列を計算しておく
        inv_matrix = np.linalg.inv(matrix)                              # 逆行列計算
        xn = np.dot(inv_matrix, F[i])[res_node]                         # 方程式を解き、複素変位ベクトルを求める
        xn = np.sqrt(xn.real ** 2 + xn.imag ** 2)                       # 振幅成分を計算する
        X.append(xn)
    return steps, X

# 固有値解析を計算する関数
def eigen(M, K):
    # 質量マトリクスの逆行列を計算
    M_inv = np.linalg.inv(M)

    # 固有値と固有ベクトルを計算
    omega, v = np.linalg.eig(np.dot(M_inv, K))

    # 固有値の順番を昇順にソート
    omega_sort = np.sqrt(np.sort(omega)) #(1 / (2 * np.pi)) *

    # 固有値のソート時のインデックスを取得
    sort_index = np.argsort(omega)

    # 固有値に対応する固有ベクトルをソート
    v_sort = []
    for i in range(len(sort_index)):
        v_sort.append(v.T[sort_index[i]])
    v_sort = np.array(v_sort)
    return omega_sort, v_sort, sort_index

# モデルの質量マトリクスと剛性マトリクスを求める関数を実行
M, K = model()

# 減衰マトリクスを計算する関数を実行
C = generate_rayleigh_damping(f1=10, f2=100, zeta1=0.2, zeta2=0.1, M=M, K=K)

# 周波数応答解析を実行
freq_axis, X = frequency_response_analysis(M=M, K=K, C=C, freq_l=10, freq_u=100, step=1, exc_node=1, res_node=1)
freq_axis = freq_axis * (1 / (2 * np.pi))

# 固有値解析を実行(確認用)
omega, v, index= eigen(M, K)
print('Natural frequency[Hz]=', omega * (1 / (2 * np.pi)))

# ここからグラフ描画
# フォントの種類とサイズを設定する。
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Times New Roman'

# 目盛を内側にする。
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# グラフの上下左右に目盛線を付ける。
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')

# スケール設定
ax1.set_xscale('log')
ax1.set_yscale('log')

# 軸のラベルを設定する。
ax1.set_xlabel('Excitation frequency [Hz]')
ax1.set_ylabel('Displacement[m]')
ax1.plot(freq_axis, X, lw=1, color='red', label='Analysis Result')

# レイアウトと凡例の設定
fig.tight_layout()
plt.legend()

# グラフを表示する。
plt.show()
plt.close()