in_data = []
with open('train/in.txt', encoding='utf-8')as inp:
    for line in inp.readlines():
        data = line.strip()
        # if len(data) > 14:
        #     continue
        in_data.append(data)

out_data = []
with open('train/out.txt', encoding='utf-8')as target:
    for line in target.readlines():
        data = line.strip()
        if len(data) > 14:
            continue
        out_data.append(data)


with open('test/in.txt', encoding='utf-8')as inp:
    for line in inp.readlines():
        data = line.strip()
        if len(data) > 14:
            continue
        in_data.append(data)

with open('test/out.txt', encoding='utf-8')as target:
    for line in target.readlines():
        data = line.strip()
        if len(data) > 14:
            continue
        out_data.append(data)


import pandas as pd

frame = pd.DataFrame({'in': in_data, 'out': out_data})
frame.to_csv('tarin.csv', encoding='utf_8_sig', index=False)

# import numpy as np
#
# array = np.array([['1', '2'], ['1']])
# print(array.shape)
