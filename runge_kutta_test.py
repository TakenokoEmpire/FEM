import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def runge_kutta_v2(M, C, K, dt, t_max, x_0, v_0):
    # 初期値、変数の定義
    n = len(M)                    # 自由度
    t = np.arange(0, t_max, dt)       # 時間データ

    # n = len(m)                     # 自由度
    # M = np.zeros((n, n))           # 質量マトリクス
    # C = np.zeros((n, n))           # 減衰マトリクス
    # K = np.zeros((n, n))           # 剛性マトリクス

    # # マトリクスの生成
    # for i in range(n):
    #     M[i, i] = m[i]
    #     if i == 0:
    #         C[i, [i, i + 1]] = [c[i] + c[i + 1], -c[i + 1]]
    #         K[i, [i, i + 1]] = [k[i] + k[i + 1], -k[i + 1]]
    #     elif i == n - 1:
    #         C[i, [-2, -1]] = [-c[i], c[i]]
    #         K[i, [-2, -1]] = [-k[i], k[i]]
    #     else:
    #         C[i, [i - 1, i, i + 1]] = [-c[i], c[i] + c[i + 1], -c[i + 1]]
    #         K[i, [i - 1, i, i + 1]] = [-k[i], k[i] + k[i + 1], -k[i + 1]]


    # 運動方程式の関数化
    def func(v, x, M, C, K):
        '''運動方程式'''
        # 各種変数定義

        M_inv = np.linalg.inv(M)       # 質量マトリクスの逆行列
        return - M_inv @ C @ v.T - M_inv @ K @ x.T



    # ルンゲクッタ法による数値解析
    def runge_kutta_method(n, t, dt, x_0, v_0, func):
        '''ルンゲクッタ法'''
        x = np.zeros((len(t), n))
        v = np.zeros((len(t), n))
        for i in tqdm(range(len(t) - 1)):
            if i == 0:
                x[i, :] = x_0
                v[i, :] = v_0
                
            k1_x = v[i, :] * dt
            k1_v = func(v[i, :], x[i, :], M, C, K) * dt
            
            k2_x = (v[i, :] + k1_v / 2) * dt
            k2_v = func(v[i, :] + k1_v / 2, x[i, :] + k1_x / 2, M, C, K) * dt
            
            k3_x = (v[i, :] + k2_v / 2) * dt
            k3_v = func(v[i, :] + k2_v / 2, x[i, :] + k2_x / 2, M, C, K) * dt
            
            k4_x = (v[i, :] + k3_v) * dt
            k4_v = func(v[i, :] + k3_v, x[i, :] + k3_x, M, C, K) * dt
            
            x[i + 1, :] = x[i, :] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
            v[i + 1, :] = v[i, :] + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
        return x




    # プロット(各自由度の初期間隔を2としてプロットする)
    x_rk = runge_kutta_method(n, t, dt, x_0, v_0, func)
    x_rk_even = np.zeros((len(t), int(n/2)))
    for i in range(int(n/2)):
        x_rk_even[:, i] = x_rk[:, i*2]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(int(n/2)):
        ax.plot(t, x_rk_even[:, i], label=str(i + 1))
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.legend(loc='best')
    plt.show()
    
    return x_rk
    
