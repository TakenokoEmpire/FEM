from calendar import c
import enum
import numpy as np
from matplotlib import pyplot as plt
import math
import time
import csv
import csv_write
import fft
# import frf_multi
from tqdm import tqdm
import copy

# やること
# FRFを抽出して、例のRCの式に代入できるようにする


# 結果を表示したい点を選択
measurement_positions = [2,4]
fft_positions = [0]

# 既定値よりも速くアニメーションを再生したい場合　int
animation_speed_boost = 100

# csvファイルをまとめて読み込み
# 「_box」をつけたものが、複数ファイルに対応している。
# とりあえず、変数の内容は最後のcsvファイルに合わせるようにしている。これは後々修正しよう。
# 全てのファイルの変数名を考慮　：　×
# 生波形を重ねて表示　：　×
# FRFを重ねて表示　：　×（優先度高）
# アニメーションを重ねて表示　：　概ね完成（治具の位置表示等が未完成）

csv_file_mumber_list = [44]

f_wave_sampled_box = []
x_box_box = []

for csv_file_order, csv_file_number in enumerate(csv_file_mumber_list):
    with open('data_box/{}.csv'.format(str(csv_file_number))) as f:
        reader = csv.reader(f)
        for order, row in enumerate(reader):
            csv_content = [row for row in reader]

    # 変数の蘇生
    # 変数の数を読み込み
    var_qty = int(csv_content[1] [1])
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
    f_wave_sampled_box.append(([float(r[0]) for r in csv_values]))
    # 変位の抽出
    x_box_box.append(np.array([list(map(float, r[1:])) for r in csv_values]))

f_wave_sampled_box = np.array(f_wave_sampled_box)
x_box_box = np.array(x_box_box)

animation_speed  = animation_speed  * animation_speed_boost

for csv_file_order, csv_file_number in enumerate(csv_file_mumber_list):

    # 各々の時間軸波形を個別に見たいなら、ここのコメントアウトを解除
    # for i in range(len(measurement_positions)):
    #     # plt.plot(yokojiku, signal_box[0])
    #     plt.plot(yokojiku, x_box_box[csv_file_order][:,measurement_positions[i]])
    #     plt.legend(measurement_positions)
    # plt.show()

    # FRFグラフの表示
    t_sampled = np.linspace(0, simu_time, sampling_length)

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
# import simulation

def frf_general(input_multi, output_multi, dt, t_axis):
    """複数用。input,outputは、データ毎にListでまとめて入力。dt,t_axisは共通。"""
    
    # ここからグラフ描画-------------------------------------
    # フォントの種類とサイズを設定する。
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'

    # 目盛を内側にする。
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # グラフの上下左右に目盛線を付ける。
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax2 = fig.add_subplot(223)
    ax2.yaxis.set_ticks_position('both')
    ax2.xaxis.set_ticks_position('both')
    ax3 = fig.add_subplot(222)
    ax3.yaxis.set_ticks_position('both')
    ax3.xaxis.set_ticks_position('both')
    ax4 = fig.add_subplot(224)
    ax4.yaxis.set_ticks_position('both')
    ax4.xaxis.set_ticks_position('both')

    # 軸のラベルを設定する。
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Force[N]')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Displacement[m]')
    ax3.set_xlabel('Excitation frequency [Hz]')
    ax3.set_ylabel('Phase[deg.]')
    ax4.set_xlabel('Excitation frequency [Hz]')
    ax4.set_ylabel('Amplitude[m/N]')

    # スケールの設定をする。
    # ax3.set_xticks(np.arange(0, 20, 2))
    # ax3.set_xlim(0, 10)
    ax3.set_yticks(np.arange(-270, 270, 90))
    ax3.set_ylim(-180, 180)
    # ax4.set_xticks(np.arange(0, 20, 2))
    # ax4.set_xlim(0, 10)
    ax4.set_yscale('log')
    

    # 周波数応答関数(FRF)を計算する関数
    def frf_core(input, output, samplerate):
        fft_i = fftpack.fft(input)                          # 入力信号のフーリエ変換
        fft_o = fftpack.fft(output)                         # 出力信号のフーリエ変換

        # FRFを計算
        h_io = (fft_o * fft_i.conjugate()) / (fft_i * fft_i.conjugate())

        amp = np.sqrt((h_io.real ** 2) + (h_io.imag ** 2))  # FRFの振幅成分
        amp = amp / (len(input) / 2)                        # 振幅成分の正規化（辻褄合わせ）
        phase = np.arctan2(h_io.imag, h_io.real)            # 位相を計算
        phase = np.degrees(phase)                           # 位相をラジアンから度に変換
        freq = np.linspace(0, samplerate, len(input))       # 周波数軸を作成
        return h_io, amp, phase, freq

    csv_file_qty = len(input_multi)
    # color_map_base = plt.get_cmap("Blues")
    if csv_file_qty == 1:
        color_maps = [[0,0,1]]
    else:
        color_maps = [[(csv_file_order) / (csv_file_qty-1), 0, 1 - (csv_file_order) / (csv_file_qty-1),] for csv_file_order in range(csv_file_qty)]
    # color_maps = [color_map_base(0.1), color_map_base(0.5), color_map_base(0.9)]
    

    for csv_file_order in range(csv_file_qty):
        input = input_multi[csv_file_order]
        output = output_multi[csv_file_order]
        # FRFを関数を実行して計算
        h_io, amp, phase, freq = frf_core(input, output, 1 / dt)

        # データプロットの準備とともに、ラベルと線の太さ、凡例の設置を行う。
        ax1.plot(t_axis, input, label='Input', lw=1, color = color_maps[csv_file_order])
        ax2.plot(t_axis, output, label='Output', lw=1, color = color_maps[csv_file_order])
        ax3.plot(freq, phase, lw=1, color = color_maps[csv_file_order])
        ax4.plot(freq, amp, lw=1, color = color_maps[csv_file_order])

    # レイアウト設定
    fig.tight_layout()

    # グラフを表示する。
    plt.show()
    plt.close()
    # ---------------------------------------------------

for position_num in fft_positions:
    frf_general(f_wave_sampled_box, x_box_box[:, :, position_num], 1/fs, t_sampled)


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
    for t in range(len(shape_list[1, :])):
        if t%speed == 0:
            x = np.linspace(x_min, x_max, len(shape_list[:, t]))
            y = shape_list[:, t]
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
            title = ax.text(0.5, 1.01, 'k_jig = {}, jig_position = {} / {} , t = {:.4f}'.format(k_jig, jig_position, qty_x * 2, t * dt), ha='center', va='bottom', transform=ax.transAxes, fontsize='large') # 追加
            artists.append(artist + [title])
        
    # 4. アニメーション化
    anim = ArtistAnimation(fig, artists)
    plt.show()

def animation_multi(shape_list_box, dt = 1, speed = 1, x_min = 0, x_max = 100):
    """
    shape_list_box: 3重List（外側から、csv_file_order, element_order, t）
    speed: int: インターバル。コマを間引くことによって1コマあたりの経過秒数を増やして、描画速度を上げる。"""
    # 1. 必要なモジュールの読み込み
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.animation import ArtistAnimation
    
    # 2.グラフ領域の作成
    fig, ax = plt.subplots()
    
    # 3. グラフ要素のリスト（artists）作成
    artists = []
    shape_list_qty = len(shape_list_box)
    
    for t in range(len(shape_list_box[0, 1, :])):
        if t%speed == 0:
            artist_box = []
            # color_maps = ["red", "orange", "yellow", "green", "blue"] # 同時比較数を5以上にするならここを変更
            # 色の指定
            if len(shape_list_box) == 1:
                color_maps = [[0,0,1]]
            else:
                color_maps = [[(csv_file_order) / (len(shape_list_box)-1), 0, 1 - (csv_file_order) / (len(shape_list_box)-1),] for csv_file_order in range(len(shape_list_box))]
            for csv_file_order in range(len(shape_list_box)):
                x = np.linspace(x_min, x_max, len(shape_list_box[csv_file_order, :, t]))
                y = shape_list_box[csv_file_order, :, t]
                artist_box.append(ax.plot(x, y, color = color_maps[csv_file_order]))
                for order,jig_pos in enumerate(jig_position):
                    if jig_pos % 2 == 0:
                        # 通常バネ
                        plt.vlines(total_length * jig_pos / vec_length, np.amin(shape_list_box[csv_file_order]) * 0.8, np.amax(shape_list_box[csv_file_order]) * 0.4, colors = "red") # 治具の位置に縦線
                        ax.text(jig_pos / vec_length, 0.02, "Normal Jig{}".format(order), ha='center', transform=ax.transAxes, color = "red") # 治具の注釈
                    else:
                        # 回転バネ
                        plt.vlines(total_length * (jig_pos-1) / vec_length, np.amin(shape_list_box[csv_file_order]) * 0.4, np.amax(shape_list_box[csv_file_order]) * 0.8, colors = "blue") # 治具の位置に縦線
                        ax.text(jig_pos / vec_length, 0.97, "Rotation Jig{}".format(order), ha='center', transform=ax.transAxes, color = "blue") # 治具の注釈
                
            plt.hlines(0, 0, total_length, linewidth = 0.1) # y=0の基準線
            title = ax.text(0.5, 1.01, 'k_jig = {}, jig_position = {} / {} , t = {:.4f}'.format(k_jig, jig_position, qty_x * 2, t * dt), ha='center', va='bottom', transform=ax.transAxes, fontsize='large') # 追加
            artists.append([])
            for csv_file_order in range(len(shape_list_box)):
                artists[-1] += artist_box[csv_file_order]
            artists[-1] += [title]
        
    # 4. アニメーション化
    anim = ArtistAnimation(fig, artists)
    plt.show()

x_box_box_even = []
for csv_file_order, csv_file_number in enumerate(csv_file_mumber_list):
    x_box_even = []
    x_box = x_box_box[csv_file_order]
    for i in range(int(len(x_box[1,:]) / 2)):
        x_box_even.append(x_box[:, i*2])
    x_box_box_even.append(x_box_even)
x_box_box_even = np.array(x_box_box_even)

animation_multi(x_box_box_even, 1/fs, animation_speed, 0, total_length)