import numpy as np
def averaging_sample(signal, sampling_interval, sampling_length):
    """値の平均値をとりながらサンプリングする（例:5Hz→1Hzにサンプリングするとき、[5,0,0,0,0]→[1]にする）
    インパクト加振のFFTを想定
    入出力のsignalはnp.array"""
    signal_length = len(signal)
    if sampling_interval * sampling_length < signal_length:
        print("averaging_sample: 末尾の信号が切り捨てられています（長さ{}）".format(signal_length - sampling_interval * sampling_length))
    sampled_signal = []
    for i in range(sampling_length):
        start_point = i * sampling_interval
        end_point = (i+1) * sampling_interval
        if end_point <= signal_length:
            sampled_signal.append(np.average(signal[start_point:end_point]))
        elif start_point < signal_length:
            sampled_signal.append(np.average(signal[start_point:]))
        else:
            sampled_signal.append(0)
    return np.array(sampled_signal)

print(averaging_sample(np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,16]),4,3))

