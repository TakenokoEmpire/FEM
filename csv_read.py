from calendar import c
import enum
import numpy as np
from matplotlib import pyplot as plt
import math
import time
import csv
import csv_write
from tqdm import tqdm
import copy
import cmath

def csv_read_float(file_name):
    """解析結果記録部分の29行目以降のみ出力"""
    with open('data_box/{}.csv'.format(file_name)) as f:
        reader = csv.reader(f)
        for order, row in enumerate(reader):
            csv_content = [row for row in reader]
    csv_values = csv_content[29:]
    csv_output = []
    for order, row in enumerate(csv_values):
        csv_output.append(np.array([float(r) for r in row]))
    csv_output = np.array(csv_output)
    return csv_output