import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np
from random import random
import math
import os
import cv2
imgs = []
YSTART = 262
XSTART = 162
YLEN = 38
XLEN = 18
LEN = 13 - 4
#DIRS = [[-1,-1],[-1,0],[-1,1],[0,1]]
DIRS = [[-1,0],[1,0],[0,-1],[0,1]]

LINENUM = 30

if not os.path.exists('truth'):
      os.makedirs('truth')
if not os.path.exists('64data'):
      os.makedirs('64data')

for i0 in range(0,30000):
    imgnew = np.ones((XLEN,YLEN,3),dtype=np.uint8)*255
    for i in range(0,LINENUM):
        while True:
            x = int(random() * XLEN)
            y = int(random() * YLEN)
            dip = int(random()*4) 
            di = DIRS[dip]
            cross = 0
            value = int(random()*LEN)+4
            for j in range(value):
                ix = x + di[0] * j
                iy = y + di[1] * j
                if ix < 0 or ix >= XLEN or iy < 0 or iy >= YLEN: break
                if imgnew[ix][iy][0] == 0: cross += 1
            if cross >= 3: continue
            if cross != 1 and random() < 0.5: continue

            adj = 0
            erp = 0
            for p in range(4):
                if p == dip: continue
                for j in range(value):
                    ix = x + DIRS[p][0] + di[0] * j
                    iy = y + DIRS[p][1] + di[1] * j
                    if ix < 0 or ix >= XLEN or iy < 0 or iy >= YLEN: break
                    if imgnew[ix][iy][0] == 0: adj += 1
                    else: adj = 0
                    if adj >= 3:
                        erp = 1
                        break
                if erp == 1:break
            if erp == 1:continue

            for j in range(value):
                ix = x + di[0] * j
                iy = y + di[1] * j
                if ix < 0 or ix >= XLEN or iy < 0 or iy >= YLEN: break
                imgnew[ix][iy][0] = 0
                imgnew[ix][iy][1] = 0
                imgnew[ix][iy][2] = 0
            break  


    mpimg.imsave(r"truth/"+str(i0)+r".jpg", imgnew)

    noize = np.random.normal(-50, 50, (XLEN,YLEN))
    noizes = np.stack((noize, noize, noize),axis = 2)
    imgnoize = np.clip(imgnew + noizes, 0, 255).astype(np.uint8)

    imblur = cv2.blur(imgnoize,(3,3))

    for i in range(8):
        for j in range(8):
            imlarge = cv2.resize(imblur, (YLEN*8,XLEN*8))
            img1 = np.roll(imlarge, i, axis=0)
            img2 = np.roll(img1, j, axis=1)
            imgsmall = cv2.resize(img2, (YLEN,XLEN))

            mpimg.imsave(r"64data/"+str((i*8+j)+64*i0)+r".jpg", imgsmall)
    if(i0%1000==0):print(i0)

