import csv
import pprint
import time, datetime
import numpy as np


def csv_write_2d_list(save_data, file_name = "", additional_info = [[]], header = [], identify = "number"):
    """
    save_data : 2次元のListや配列　30行目～
    additional_info : 2次元のListや配列 （測定条件等の情報）28行以下　1~28行目　※2次元以上でもよい（その場合、Listの形で保存される）
    header :  ヘッダー　29行目
    identify: ファイル名が重複しないようにするための方法。"time_tag"を選択すると、自動で時刻をファイル名に含める。"number"を選択すると、該当パス内の"number_memory"から最新のナンバーを参照し、numberを割り振る。"null"を選択すると、file_nameがそのままファイル名になる（ファイル名被ったら警告なしで上書きされるため非推奨）
    """

    # if file_name == None:
    #     time = datetime.datetime.now()
    #     file_name = time.strftime('%y%m%d_%H%M%S')]

    # 0~29行目を埋める（ない場合は空白にする）
    if len(additional_info) > 28:
        print("additional_infoが過大です（29行目以降はカットします）")
    additional_info_and_header_box = []
    # additional_infoの入れ込み
    for i in range(28):
        if i < len(additional_info):
            additional_info_and_header_box.append(additional_info[i])
        else:
            additional_info_and_header_box.append([])
    # headerの入れ込み
    additional_info_and_header_box.append(header)

    if identify == "time_tag":
        # ファイル名にタイムタグをつける場合
        time = datetime.datetime.now()
        identify_name = time.strftime('%y%m%d_%H%M%S')
    elif identify == "null":
        # file_nameをそのままファイル名にする場合
        identify_name = ""
    else:
        # ファイル名に通し番号を付ける場合
        # 現在の最新番号の読み込み
        with open('data_box/number_memory.txt') as f:
            identify_name = f.readlines()
            identify_name = int(identify_name[0][:-1])
        # 番号の更新
        with open('data_box/number_memory.txt', 'w') as f:
            print(str(identify_name+1), file=f)
        
        # with open('data_box/number_memory.csv') as f:
        #     identify_name = int(f.read()) + 1
        #     f.close()
        # with open('data_box/number_memory.csv') as f:
        #     writer = csv.writer(f)
        
    
    if file_name != "" and identify != "null":
        file_name = "_" + file_name

    # with open('data_box/{}{}.csv'.format(time_tag, file_name), 'w', newline="") as f: タイムタグをつけたいならこちらを採用
    if identify == "number":
        print("file_number : {}".format(identify_name))
    with open('data_box/{}{}.csv'.format(identify_name, file_name), 'w', newline="") as f:
        writer = csv.writer(f)
        # save_data = numpy.zeros((6,8))
        writer.writerow(["測定結果は下部にあります"])
        writer.writerows(additional_info_and_header_box)
        writer.writerows(save_data)
        
# csv_write_2d_list([1])