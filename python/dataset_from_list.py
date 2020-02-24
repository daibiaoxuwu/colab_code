import cv2

truthpics = [cv2.imread('truth/'+str(i)+'.jpg',cv2.IMREAD_GRAYSCALE)/255.0 for i in range(PIC_NUM)]
datapics = [cv2.imread('64data/'+str(i*64)+'.jpg',cv2.IMREAD_GRAYSCALE)/255.0 for i in range(PIC_NUM)]

from matplotlib import pyplot as plt
import cv2
import numpy as np
def similarity(i1, i2):
    im1 = i1>0.5
    im2 = i2>0.5
    return(np.sum((im1*im2)>0)/np.sum((im1+im2)>0))    


def train_generator():
  while True:
        for i in range(int(PIC_NUM*0.8)):
            i1 = truthpics[i]
            i2 = datapics[i]
            yield(np.stack((i1,i2),axis=2),1.0)
        shuffle = np.random.permutation(np.arange(int(PIC_NUM*0.8)))
        for i in range(int(PIC_NUM*0.8)):
            i1 = truthpics[shuffle[i]]
            i2 = datapics[i]
            yield(np.stack((i1,i2),axis=2),similarity(i1,truthpics[i]))
def test_generator():
  while True:
        for i in range(int(PIC_NUM*0.8),PIC_NUM):
            i1 = truthpics[i]
            i2 = datapics[i]
            yield(np.stack((i1,i2),axis=2),1.0)
        shuffle = np.random.permutation(np.arange(int(PIC_NUM*0.8),PIC_NUM))
        for i in range(int(PIC_NUM*0.8),PIC_NUM):
            i1 = truthpics[shuffle[i]]
            i2 = datapics[i]         
            yield(np.stack((i1,i2),axis=2),similarity(i1,truthpics[i]))
train_dataset = tf.data.Dataset.from_generator(train_generator, (tf.float32, tf.float32), (tf.TensorShape([18,38,2]),tf.TensorShape([])))
train_dataset=train_dataset.batch(100).shuffle(100)
test_dataset = tf.data.Dataset.from_generator(test_generator, (tf.float32, tf.float32), (tf.TensorShape([18,38,2]),tf.TensorShape([])))
test_dataset=test_dataset.batch(100).shuffle(100)
