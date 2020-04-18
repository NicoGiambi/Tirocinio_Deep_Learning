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

with open('area.txt') as json_file:
    history_area = json.load(json_file)


all_history_area = []
for cat in history_area:
    for el in history_area[cat]:
        all_history_area.append(el)

history_area['all'] = all_history_area

cat = args
el_number = 0
mean_area = 0
distance = 0.95

for el in history_area[cat]:
    el_number += 1
    mean_area += el

mean_area /= el_number
cat_history_area=[]
k=0
counter = 0
offset = mean_area - (mean_area * distance)
iter = 50
max = mean_area + (mean_area * distance)
range = max/iter

for el in history_area[cat]:
    if (float(el) >= (k * range + offset)):
        if k < iter:
            if ((float(el) >= (k*range + offset)) & (float(el) < (k*range + range + offset))):
                counter += 1
            else:
                cat_history_area.append(counter)
                counter = 0
                k+= 1

percentage_history_area = []
for el in cat_history_area:
    percentage_history_area.append((100*el)/el_number)

pixel_history_area = []
index = 0
for el in cat_history_area:
    pixel_history_area.append(index * (range + offset))
    index += 1



label = " Area in pixels"
plt.plot(pixel_history_area, percentage_history_area, 'bo',
         pixel_history_area, percentage_history_area, 'k')

plt.xlabel(label)
plt.ylabel("% of " + cat + " images")
plt.grid(True)
plt.show()
