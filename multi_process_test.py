######## 並列計算を使えるように #########
from multiprocessing import Pool
import time
from turtle import end_fill
 
##### 並列計算させる関数(処理):引数1つ ###
##### この場合は，引数の二乗を返す関数 ###
qty = 100000000

def nijou(x):
    y =  x*x 
    # if x%(qty/1000) == 0:
    #     print(x/(qty/100))
 
 
for i in range(qty):
    nijou(i)
###### 並列計算させてみる #########
# time_box = time.time()
# if __name__ == "__main__":
#     p = Pool(2)
#     p.map( nijou, range(qty) )#nijouに0,1,..のそれぞれを与えて並列演算