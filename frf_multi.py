import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
# import simulation

def frf_general(input_multi, output_multi, dt, t_axis):
    """もうこのファイル使ってない（cantilever6Bに統合）"""
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