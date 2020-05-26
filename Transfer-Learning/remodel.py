import keras
import numpy as np
from keras import losses
import keras.backend as K
from keras.layers import Dense, GlobalAveragePooling2D, Conv1D, Conv2D
from keras.models import Model
from keras.applications import inception_v3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras import losses
from keras.models import load_model
import matplotlib.pyplot as plt
import json
import ast
import os
import sys
from pycocotools.coco import COCO
import tensorflow_datasets as tfds
import tensorflow as tf
import cv2
import tensorflow.compat.v1 as tf1


IM_WIDTH, IM_HEIGHT = 459,459
nb_train_samples = 118287           #COCO train samples
nb_val_samples = 5000               #COCO val samples
BATCH = 100
EPOCHS = nb_train_samples//BATCH


#Leggo il txt contenente le ground_truth
with open('all_ground_truth.txt') as json_file:
    ground_truth = json.load(json_file)



def my_loss(y_true, y_pred):
  crossentropy = losses.sparse_categorical_crossentropy(y_true, y_pred, axis=-1)
  bool = tf.ones(shape=(1, 13, 13, 1), dtype=tf.float32)
  mask = tf.math.minimum(bool, y_true)
  mask = tf.squeeze(mask, axis=-1)
  final = mask * crossentropy
  loss = K.sum(final, axis=(1,2))
  return loss


#Funzione che aggiunge due layer convoluzionali per avere un output da 13x13x81
def add_new_last_layer(base_model):
  x = base_model.output
  x = Conv2D(filters=512, kernel_size=(1), data_format="channels_last")(x)
  x = Conv2D(filters=81, kernel_size=(1), data_format="channels_last")(x)
  model = Model(inputs=base_model.input, outputs=x)
  return model


def setup_to_transfer_learn(base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  return base_model


#Funzione per caricare pezzi di dataset piccoli per evitare di riempire la Memoria
def get_x_y (dataset):
    names = []
    for el in dataset:
        a = str(np.array(el['image/filename']))
        a = a.strip('b').strip("'").strip("0").strip(".jpg")
        names.append(a)
    # names = [str(np.array(el['image/filename'])).strip('b').strip("'").strip("0").strip(".jpg") for el in dataset] #---- Versione List comprehension

    # Carica in y la parte di ground truth corrispondente al pezzo di dataset che sto caricando
    y = [np.array(ast.literal_eval(ground_truth[el])) for el in names]
    y = [np.expand_dims(el, axis=2) for el in y]
    #y = [keras.utils.np_utils.to_categorical(el) for el in y]
    # Carico in x la parte di dataset interessata e la preparo per il training
    x = [np.array(el['image']) for el in dataset]
    x = [cv2.resize(el, dsize=(459,459), interpolation=cv2.INTER_CUBIC) for el in x]
    x = [el/255 for el in x]
    #print(np.array(x).shape)
    #print (type(x[0]), type(y[0]), type(np.array(x)), type(np.array(y)), x[0].shape, y[0].shape, np.array(x).shape, np.array(y).shape)
    return (np.array(x),np.array(y))

def train(model):

    for i in range(EPOCHS-1):
        print("Step: " + str(i))
        k = BATCH * i
        string = "train[" + str(k) + ':' + str(k + BATCH) + "]"
        dataset = tfds.load(name="coco/2017", split=string)
        x_train,y_train = get_x_y(dataset)
        model.fit(x=x_train, y=y_train, epochs=1, verbose=1, batch_size=25)
    val = tfds.load(name="coco/2017", split="validation")
    x_val, y_val = get_x_y(val)
    model.fit(x_val, y_val, validation_split=1.0, steps_per_epoch=1, verbose=1, epochs=1,  validation_steps=1)
    model.save("coco_model.txt")


#tf.compat.v1.enable_eager_execution()
inception_model = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
coco_model = add_new_last_layer(inception_model)
coco_model = setup_to_transfer_learn(coco_model)
#coco_model.save("new_model.txt")
#coco_model = load_model("new_model.txt")
coco_model.compile(optimizer='rmsprop',
                    loss=my_loss,
                    metrics=['accuracy']
                    )

'''  ------------------ In questo modo inizia il training, non accetta my_loss
coco_model.compile(optimizer='rmsprop',
                    loss=losses.sparse_categorical_crossentropy,
                    metrics=['accuracy']
                    )
'''
train(coco_model)
predict(coco_model)