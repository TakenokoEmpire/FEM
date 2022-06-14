import numpy as np
from matplotlib import pyplot as plt

# パラメータを設定
m1 = 50000
m2 = 50000
m3 = 50000
m4 = 50000
k1 = 30000000
k2 = 20000000
k3 = 10000000
k4 = 5000000

# 質量マトリクスと剛性マトリクスを作成
M = np.array([[m1, 0, 0, 0],
             [0, m2, 0, 0],
             [0, 0, m3, 0],
             [0, 0, 0, m4],])
K = np.array([[k1 + k2, -k2, 0, 0],
             [-k2, k2 + k3, -k3, 0],
             [0, -k3, k3 + k4, k4],
             [0, 0, -k4, k4]])

# 質量マトリクスの逆行列を計算
M_inv = np.linalg.inv(M)

print(M)
print(K)

# 固有値と固有ベクトルを計算
omega, v = np.linalg.eig(np.dot(M_inv, K))

# 固有値の順番を昇順にソート
omega_sort = np.sort(omega)

# 固有値のソート時のインデックスを取得
# ⇒固有ベクトルと対応させるため
sort_index = np.argsort(omega)

# 固有値に対応する固有ベクトルをソート
v_sort = []
for i in range(len(sort_index)):
    v_sort.append(v.T[sort_index[i]])
v_sort = np.array(v_sort)

# 結果をコンソールに表示
print(np.sqrt(omega_sort))
print(v_sort)

# グラフ化のために自由度軸を作成
dof = np.linspace(0, len(sort_index), len(sort_index)+1)

# ここからグラフ描画
# フォントの種類とサイズを設定する。
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Times New Roman'

# 目盛を内側にする。
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# グラフの上下左右に目盛線を付ける。
fig = plt.figure(figsize=(6,3))
ax1 = fig.add_subplot(111)
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')

# 軸のラベルを設定する。
ax1.set_xlabel('Degree of freedom')
ax1.set_ylabel('Eigenvector')

# データの範囲と刻み目盛を明示する。
ax1.set_xticks(np.arange(0, 4, 1))
ax1.set_yticks(np.arange(-2, 2, 0.5))
ax1.set_xlim(0, 4)
ax1.set_ylim(-1, 1)

# データプロット 固有ベクトルの形を見る
for i in range(len(sort_index)):
    eigen_vector = np.concatenate([[0], v_sort[i]])
    ax1.plot(dof, eigen_vector,lw=1, marker='o')

fig.tight_layout()
plt.legend()

# グラフを表示する。
plt.show()
plt.close()