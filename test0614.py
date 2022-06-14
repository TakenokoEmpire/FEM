import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi)
y1 = np.sin(x)
y2 = np.cos(x)
 
 


plt.title("sine curve", size = 20, color = "red")

plt.plot(x, y1, label="sin")
plt.plot(x, y2, label="cos")

plt.xlabel("X-value")  # x軸ラベルの設定
plt.ylabel("Y-value")  # y軸ラベルの設定

plt.xlim(0, 2*np.pi)  # x軸の最大・最小値設定
plt.ylim(-1.5, 1.5)  # y軸の最大・最小値設定

plt.grid()  # グリッド線の表示
plt.legend()
plt.show()
