import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import numpy as np
fig, ax = plt.subplots()
artists = []
x = np.arange(10)
for i in range(10):
    y = np.random.rand(10)
    im = ax.scatter(x, y)
    artists.append([im])
anim = ArtistAnimation(fig, artists, interval=1000)
fig.show()