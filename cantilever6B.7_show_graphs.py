from calendar import c
import enum
import numpy as np
from matplotlib import pyplot as plt
import math
import time
import csv
import csv_write
import fft
import frf
from tqdm import tqdm
import copy

# 結果を表示したい点を選択
measurement_positions = [2,4]
fft_positions = [-2]

# csvファイルをまとめて読み込み
with open('data_box/6.csv') as f:
    reader = csv.reader(f)
    for order, row in enumerate(reader):
        csv_content = [row for row in reader]

# 変数の蘇生
# 変数の数を読み込み
var_qty = int(csv_content[1] [1])
print(int(var_qty))
# 変数名のリストを作る
var_name_list = []
for i in range(var_qty - 2):
    var_name_list.append(csv_content[i+2] [0])
# 変数名と変数の値を対応付ける
for order, var_name in enumerate(var_name_list):
    target_var_value = csv_content[order+2] [1]
    # 読み込んだ値はstr型なので、適切な型に変換する
    # "["を含むならListとして扱う（ndarrayとして出力）。中身はすべてFloatとして扱うので注意。
    if "[" in target_var_value:
        remove_characters = "[]"
        for i in range(len(remove_characters)):
            target_var_value = target_var_value.replace(remove_characters[i], "")
        target_var_value = target_var_value.split(",")
        target_var_value = np.array(list(map(float, target_var_value)))
    else:
        # まず、int化できるかを試して…
        try:
            target_var_value = int(target_var_value)
        except ValueError:
            # ダメならfloat化できるかを試す（1e-7等の指数もここで引っかかるはず）
            try:
                target_var_value = float(target_var_value)
            except ValueError:
                # それでもダメならstrのまま通す
                pass
    
    globals()[var_name] = target_var_value


# 横軸の設定（長過ぎるので、工夫しないとcsvに書けない）
yokojiku = np.linspace(0, simu_time , sampling_length)

# 測定結果部分の抽出
csv_values = csv_content[29:]
# 力の抽出
f_wave_sampled = np.array([float(r[0]) for r in csv_values])
# 変位の抽出
x_box = np.array([list(map(float, r[1:])) for r in csv_values])



var_list = []

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
            for order,jig_pos in enumerate(jig_position):
                if jig_pos % 2 == 0:
                    # 通常バネ
                    plt.vlines(total_length * jig_pos / vec_length, np.amin(shape_list) * 0.8, np.amax(shape_list) * 0.4, colors = "red") # 治具の位置に縦線
                    ax.text(jig_pos / vec_length, 0.02, "Normal Jig{}".format(order), ha='center', transform=ax.transAxes, color = "red") # 治具の注釈
                else:
                    # 回転バネ
                    plt.vlines(total_length * (jig_pos-1) / vec_length, np.amin(shape_list) * 0.4, np.amax(shape_list) * 0.8, colors = "blue") # 治具の位置に縦線
                    ax.text(jig_pos / vec_length, 0.97, "Rotation Jig{}".format(order), ha='center', transform=ax.transAxes, color = "blue") # 治具の注釈
                
            plt.hlines(0, 0, total_length, linewidth = 0.1) # y=0の基準線
            title = ax.text(0.5, 1.01, 'k_jig = {}, jig_position = {} / {} , t = {:.4f}'.format(k_jig, jig_position, qty_x * 2, i * dt), ha='center', va='bottom', transform=ax.transAxes, fontsize='large') # 追加
            artists.append(artist + [title])
        
    # 4. アニメーション化
    anim = ArtistAnimation(fig, artists)
    plt.show()

x_box_even = []
for i in range(int(len(x_box[1,:]) / 2)):
    x_box_even.append(x_box[:, i*2])
x_box_even = np.array(x_box_even)
animation(x_box_even, 1/fs, animation_speed, 0, total_length)