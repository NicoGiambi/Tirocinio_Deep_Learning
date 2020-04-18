from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import math
import json
import argparse

a = argparse.ArgumentParser()
a.add_argument("-c", default='all')
args = a.parse_args().c

with open('wh.txt') as json_file:
    history_wh = json.load(json_file)
with open('hw.txt') as json_file:
    history_hw = json.load(json_file)

all_history_wh = []
for cat in history_wh:
    for el in history_wh[cat]:
        all_history_wh.append(el)

all_history_hw = []
for cat in history_hw:
    for el in history_hw[cat]:
        all_history_hw.append(el)

history_wh['all'] = all_history_wh
history_hw['all'] = all_history_hw

cat_history_wh=[]
cat_history_hw=[]
k=0
counter = 0
cat = args
iter = 19
max = 1
range = max/iter

el_number = 0

for el in history_wh[cat]:
    el_number += 1

for el in history_wh[cat]:
    if k < iter:
        if (float(el) >= k*range) & (float(el) < k*range + range):
            counter += 1
        else:
            cat_history_wh.append(counter)
            counter = 0
            k+= 1

k=0
for el in history_hw[cat]:
    if k < iter:
        if (float(el) >= k*range) & (float(el) < k*range + range):
            counter += 1
        else:
            cat_history_hw.append(counter)
            counter = 0
            k += 1

index_history = []
index = 0
for el in cat_history_wh:
    index_history.append(index/iter)
    index += 1

percentage_history_wh = []
for el in cat_history_wh:
    percentage_history_wh.append((100*el)/el_number)

percentage_history_hw = []
for el in cat_history_hw:
    percentage_history_hw.append((100*el)/el_number)


label = " Blue: w/h ratio, Red: h/w ratio"
plt.plot(index_history, percentage_history_wh, 'bo',
         index_history, percentage_history_wh, 'k',
         index_history, percentage_history_hw, 'ro',
         index_history, percentage_history_hw, 'k')

plt.xlabel(label)
plt.ylabel("% of " + cat + " images")
plt.show()
