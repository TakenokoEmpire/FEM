import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
# import simulation

def frf_general(input, output, dt, t_axis):
    """過去のバージョンを動かすために保存してある。更新は「frf_multi」へ"""
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


    # # サンプル信号を準備する(別ファイルで作成したsimulation.pyのsim()関数で計算)
    # input, output, dt, t_axis = simulation.sim()

    # FRFを関数を実行して計算
    h_io, amp, phase, freq = frf_core(input, output, 1 / dt)

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


    # データプロットの準備とともに、ラベルと線の太さ、凡例の設置を行う。
    ax1.plot(t_axis, input, label='Input', lw=1, color='red')
    ax2.plot(t_axis, output, label='Output', lw=1, color='blue')
    ax3.plot(freq, phase, lw=1)
    ax4.plot(freq, amp, lw=1)

    # レイアウト設定
    fig.tight_layout()

    # グラフを表示する。
    plt.show()
    plt.close()
    # ---------------------------------------------------