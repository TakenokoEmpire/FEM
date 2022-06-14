import numpy as np
def animation(shape_list, x_min = 0, x_max = 100, dt = 1):
    # 1. 必要なモジュールの読み込み
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.animation import ArtistAnimation
    
    # 2.グラフ領域の作成
    fig, ax = plt.subplots()
    
    # 3. グラフ要素のリスト（artists）作成
    artists = []
    for i in range(len(shape_list[:, 1])):
        x = np.linspace(x_min, x_max, len(shape_list[i, :]))
        y = shape_list[i, :]
        artist = ax.plot(x, y,"blue")
        artists.append(artist)
        
        
    # 4. アニメーション化
    anim = ArtistAnimation(fig, artists)
    plt.show()
    
animation(np.array([[1,2,3,4],[5,6,7,8],[1,3,4,5],[2,5,5,2],[4,6,7,9]]))