import numpy as np
import matplotlib.pyplot as plt


# 初期値、変数の定義
m = np.array((10, 5, 10))     # 各自由度の質量 (m1, m2, m3, ... , mn)
c = np.array((1, 2, 2)) # 各自由度の減衰係数 (c1, c2, c3, ... , cn)
k = np.array((5000, 1000, 3000))    # 各自由度の弾性係数(剛性) (k1, k2, k3, ... , kn)
n = len(m)                    # 自由度
x_0 = np.array((-1.0, 0, 0))  # 変位の初期値
v_0 = np.array((0, 0, 0))     # 速度の初期値
t_max = 20                        # 終了時間
dt = 0.001                    # サンプリング間隔
t = np.arange(0, t_max, dt)       # 時間データ



# 運動方程式の関数化
def func(v, x, m, c, k):
    '''運動方程式'''
    # 各種変数定義
    n = len(m)                     # 自由度
    M = np.zeros((n, n))           # 質量マトリクス
    C = np.zeros((n, n))           # 減衰マトリクス
    K = np.zeros((n, n))           # 剛性マトリクス
    
    # マトリクスの生成
    for i in range(n):
        M[i, i] = m[i]
        if i == 0:
            C[i, [i, i + 1]] = [c[i] + c[i + 1], -c[i + 1]]
            K[i, [i, i + 1]] = [k[i] + k[i + 1], -k[i + 1]]
        elif i == n - 1:
            C[i, [-2, -1]] = [-c[i], c[i]]
            K[i, [-2, -1]] = [-k[i], k[i]]
        else:
            C[i, [i - 1, i, i + 1]] = [-c[i], c[i] + c[i + 1], -c[i + 1]]
            K[i, [i - 1, i, i + 1]] = [-k[i], k[i] + k[i + 1], -k[i + 1]]
    M_inv = np.linalg.inv(M)       # 質量マトリクスの逆行列
    return - M_inv @ C @ v.T - M_inv @ K @ x.T



# ルンゲクッタ法による数値解析
def runge_kutta_method(n, t, dt, x_0, v_0, func):
    '''ルンゲクッタ法'''
    x = np.zeros((len(t), n))
    v = np.zeros((len(t), n))
    for i in range(len(t) - 1):
        if i == 0:
            x[i, :] = x_0
            v[i, :] = v_0
            
        k1_x = v[i, :] * dt
        k1_v = func(v[i, :], x[i, :], m, c, k) * dt
        
        k2_x = (v[i, :] + k1_v / 2) * dt
        k2_v = func(v[i, :] + k1_v / 2, x[i, :] + k1_x / 2, m, c, k) * dt
        
        k3_x = (v[i, :] + k2_v / 2) * dt
        k3_v = func(v[i, :] + k2_v / 2, x[i, :] + k2_x / 2, m, c, k) * dt
        
        k4_x = (v[i, :] + k3_v) * dt
        k4_v = func(v[i, :] + k3_v, x[i, :] + k3_x, m, c, k) * dt
        
        x[i + 1, :] = x[i, :] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        v[i + 1, :] = v[i, :] + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
    return x

# プロット(各自由度の初期間隔を2としてプロットする)
x_rk = runge_kutta_method(n, t, dt, x_0, v_0, func)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i in range(n):
    ax.plot(t, x_rk[:, i] + 2 * i, label=str(i + 1))
ax.set_xlabel('t')
ax.set_ylabel('x')
plt.legend(loc='best')
plt.show()
