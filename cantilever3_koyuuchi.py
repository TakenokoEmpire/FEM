import numpy as np
from matplotlib import pyplot as plt

# パラメータを設定
element_qty_x = 10

m_element = 0.03

k_wall = 200
k_internal = 120
k_jig = 350

# 何番目の位置にジグを入れるか 番号は0スタートなので注意
jig_position = 2

# k1,k2,...,knのベクトル
k_vector = np.ones((1,element_qty_x))* k_internal
k_vector[0,0] = k_wall

# 質量マトリクスと剛性マトリクスを作成
M = np.zeros((element_qty_x,element_qty_x), dtype=int)
M = M + np.identity(element_qty_x, dtype=int)*m_element

K = np.zeros((element_qty_x,element_qty_x), dtype=int)
K[0,0] += k_vector[0,0]
for i in range(element_qty_x-1):
    K[i,i] += k_vector[0,i+1]
    K[i,i+1] -= k_vector[0,i+1]
    K[i+1,i] -= k_vector[0,i+1]
    K[i+1,i+1] += k_vector[0,i+1]
# 剛性マトリクスに治具の剛性を追加
K[jig_position,jig_position] += k_jig


# 質量マトリクスの逆行列を計算
M_inv = np.linalg.inv(M)

print("M Matrix")
print(M)
print("K Matrix")
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
print("Natural freq")
print(np.sqrt(omega_sort))
print("Mode shape")
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
ax1.set_xticks(np.arange(0, element_qty_x+1, 1))
ax1.set_yticks(np.arange(-2, 2, 0.5))
ax1.set_xlim(0, element_qty_x)
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