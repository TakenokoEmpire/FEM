# coding:utf-8
import sys, os, json, math
import numpy as np
import pandas as pd
from docopt import docopt
from scipy import interpolate

# 質点系モデルクラス
class LumpedMassModel:
    def __init__(self, size, m_mat, k_mat, c_mat):
        self.size = size
        self.m = m_mat
        self.k = k_mat
        self.c = c_mat

# 解析条件クラス
class AnalysisCondition:
    def __init__(self, config):
        self.grav = float(config['condition']['grav'])
        self.dt = float(config['condition']['dt'])
        self.beta = float(config['condition']['beta'])
        self.damp_factor = float(config['condition']['damp_factor'])
        self.damp_target_period = float(config['condition']['damp_target_period'])

# 波形クラス
class Wave:
    def __init__(self, config):
        self.path = config['wave']['path']
        self.dt = float(config['wave']['dt'])
        self.factor = float(config['wave']['factor'])

# マップオン
def mapon(part, full, row, col):
    row_size = len(part)
    col_size = len(part)
    for i_row in range(row_size):
        for i_col in range(col_size):
            target_row = row + i_row
            target_col = col + i_col
            if 0 <= target_row and 0 <= target_col:
                full[target_row][target_col] += part[i_row][i_col]

# 質量マトリクス生成
def build_mass_matrix(m_array, condition):    
    return np.diag(list(map(lambda x: x / condition.grav, m_array)))
    
# 剛性マトリクス生成
def build_stiff_matrix(k_array):
    size = len(k_array)
    full = np.zeros((size, size))
    for i in range(size):
        k = k_array[i]
        part = [[k , -k],
                [-k,   k]]
        mapon(part, full, i-1, i-1)
    return full

# 減衰マトリクス生成
def build_damp_matrix(k_mat, factor, period):
    omega = 2.0 * math.pi / period
    return 2.0 * factor / omega * k_mat

# 初期化
def init(config_file_path):
    with open(config_file_path, mode='r') as f:
        config = json.load(f)
    condition = AnalysisCondition(config)
    m_array = config['model']['m']
    k_array = config['model']['k']
    size = len(m_array)
    k_array.reverse() # 下階を上としてマトリクス生成するため
    m_array.reverse()
    m_mat = build_mass_matrix(m_array, condition)
    k_mat = build_stiff_matrix(k_array)
    c_mat = build_damp_matrix(k_mat, condition.damp_factor, condition.damp_target_period)
    model = LumpedMassModel(size, m_mat, k_mat, c_mat)
    wave = Wave(config)
    return model, condition, wave

# 固有値解析
def eigen_analysis(model, condition):
    lambda_mat = np.linalg.inv(model.m).dot(model.k)
    eig_val, eig_vec = np.linalg.eig(lambda_mat)
    natural_periods = list(map(lambda omega2: 2.0 * math.pi / math.sqrt(omega2), eig_val))
    eigen_vectors = []
    for vec in eig_vec:
        first_node_vector = vec[-1] # 最下質点の固有ベクトルを１として基準化
        eigen_vectors.append(list(map(lambda x: x / first_node_vector, vec)))
    df = pd.DataFrame()
    df['T'] = sorted(natural_periods, reverse=True) # 固有値は順番がランダムのため、降順で並び替え
    df['Vector'] = eigen_vectors
    return df

# 波形データ（加速度時刻歴）取得
def get_wave_data(condition, wave):
    with open(wave.path, 'r') as f:
        df = pd.read_csv(f)
    time = df['t'].values.tolist()
    acc = list(map(lambda x: x * wave.factor, df['acc'].values.tolist()))
    step = int((len(time)-1) * wave.dt / condition.dt)
    func_interpolate = interpolate.interp1d(time, acc) # 線形補間関数
    new_time = np.arange(0.0, step*condition.dt, condition.dt)
    new_acc = func_interpolate(new_time)
    return new_acc

# 時刻歴応答解析
def dynamic_analysis(model, condition, wave):
    # 初期化
    m = model.m
    k = model.k
    c = model.c
    acc0 = get_wave_data(condition, wave)
    dt = condition.dt
    beta = condition.beta
    unit_vector = np.ones(model.size)
    pre_acc0 = 0.0
    time = 0.0
    dis = np.zeros(model.size)
    vel = np.zeros(model.size)
    acc = np.zeros(model.size)
    ddis = np.zeros(model.size)
    dvel = np.zeros(model.size)
    dacc = np.zeros(model.size)
    dis_history = {}
    vel_history = {}
    acc_history = {}
    for i in range(0, model.size):
        dis_history[i] = []
        vel_history[i] = []
        acc_history[i] = []
    time_history = []
    # Newmarkβ法による数値解析（増分変位による表現）
    for i in range(0, len(acc0)):
        kbar = k + (1.0/(2.0*beta*dt)) * c + (1.0/(beta*dt**2.0)) * m
        dp1 = -1.0 * m.dot(unit_vector) * (acc0[i] - pre_acc0)
        dp2 = m.dot((1.0/(beta*dt))*vel + (1.0/(2.0*beta))*acc)
        dp3 = c.dot((1.0/(2.0*beta))*vel + (1.0/(4.0*beta)-1.0)*acc*dt)
        dp = dp1 + dp2 + dp3
        ddis = np.linalg.inv(kbar).dot(dp)
        dvel = (1.0/(2.0*beta*dt))*ddis - (1.0/(2.0*beta))*vel - ((1.0/(4.0*beta)-1.0))*acc*dt
        dacc = (1.0/(beta*dt**2.0))*ddis - (1.0/(beta*dt))*vel - (1.0/(2.0*beta))*acc
        dis += ddis
        vel += dvel
        acc += dacc
        acc_abs = acc + [acc0[i] for n in range(1,model.size)]
        [dis_history[i].append(x) for i, x in enumerate(dis)]
        [vel_history[i].append(x) for i, x in enumerate(vel)]
        [acc_history[i].append(x) for i, x in enumerate(acc_abs)]
        time_history.append(time)
        time += dt
        pre_acc0 = acc0[i]
    # 出力オブジェクト
    df = pd.DataFrame({'time':time_history,
                       'acc0':acc0})
    for k, v in dis_history.items():
        df['dis' + str(k+1)] = v
    for k, v in vel_history.items():
        df['vel' + str(k+1)] = v
    for k, v in acc_history.items():
        df['acc' + str(k+1)] = v
    return df

# 結果出力
def output(out_file_path, df):
    with open(out_file_path, 'w', newline='') as f:
        df.to_csv(f)

if __name__ == "__main__":
    __doc__ = """
    Usage:
        main.py <config_file_path> <out_file_dir>
        main.py -h | --help
    Options:
        -h --help  show this help message and exit
    """
    args = docopt(__doc__)
    config_file_path = args['<config_file_path>']
    out_file_dir = args['<out_file_dir>']
    model, condition, wave = init(config_file_path)
    df_eigen = eigen_analysis(model, condition)
    df_dyna = dynamic_analysis(model, condition, wave)
    os.makedirs(out_file_dir, exist_ok=True)
    output(out_file_dir + '/out_eigen.csv', df_eigen)
    output(out_file_dir + '/out_dyna.csv', df_dyna)