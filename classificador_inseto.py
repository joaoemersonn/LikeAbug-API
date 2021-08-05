# initialize the initial learning rate, number of epochs to train for,
# and batch size
import cv2
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
#from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, plot_confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet_v2 import ResNet152V2 as App
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
INIT_LR = 1e-3
EPOCHS = 400
img_shape = (299, 299)
images_per_classe = 100  # use 0 for all images
BS = 64
steps = 20
reload_img = False
H = None
model = None
scores = None
CAMINHO_MODELO = "modelo.h5"

# import the necessary packages

# INIT TENSORFLOW
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# LOAD DATASET OF IMAGES


def train_val_test_split(df, class_labels):
    train = pd.DataFrame()
    test = pd.DataFrame()
    val = pd.DataFrame()
    for e in df1[class_labels].unique():
        temp = df1[df1[class_labels] == e]
        temp_train, temp_val, temp_test = np.split(
            temp.sample(frac=1), [int(.5*len(temp)), int(.7*len(temp))])
        train = pd.concat([train, temp_train])
        test = pd.concat([test, temp_test])
        val = pd.concat([val, temp_val])
    return train, val, test

# LOAD IMAGES FROM FOLDER


def load_images_from_folder(folder, only_path=False, label=""):
    if only_path == False:
        images = []
        file_name = []
        for filename in os.listdir(folder):
            img = plt.imread(os.path.join(folder, filename))

            if img is not None:
                end = filename.find(".")
                file_name.append(file[0:end])
                images.append(img)

        return images, file_name
    else:
        path = []
        for filename in os.listdir(folder)[:images_per_classe]:
            img_path = os.path.join(folder, filename)
            if img_path is not None:
                path.append([label, img_path])
        return path

# load the VGG16 network, ensuring the head FC layer sets are left off


def Compile_Model():
  # initialize the training data augmentation object
    global model, scores, img_shape
    scores = None
    app = App(weights="imagenet", include_top=False,
              input_tensor=Input(shape=(img_shape[0], img_shape[1], 3)))
    for layer in app.layers:
        layer.trainable = False
    initial_model = Sequential(app)
   # initial_model.add(Conv2D(124, kernel_size =(5,5), activation='relu', data_format='channels_last', padding='same', name='MyConv_1'))
    initial_model.add(MaxPooling2D(pool_size=(2, 2), name='MyMaxpool_1'))
    # initial_model.add(Conv2D(124, kernel_size =(5,5), activation='relu', data_format='channels_last', padding='same', name='MyConv_2'))#lite
 #   initial_model.add(Conv2D(248, kernel_size =(5,5), activation='relu', data_format='channels_last', padding='same', name='MyConv_2'))#power
    initial_model.add(Flatten())

#    initial_model.add(Dense(2048, activation='relu')) #power
    initial_model.add(Dense(1024, activation='relu'))  # lite
    initial_model.add(Dense(1024, activation='relu'))  # lite
    initial_model.add(Dense(1024, activation='relu'))
    #initial_model.add(Dense(582, activation='relu'))
    initial_model.add(Dense(291, activation='softmax'))

    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    initial_model.compile(loss='categorical_crossentropy',
                          optimizer=opt, metrics=['accuracy'])
    model = initial_model

 # train the head of the network


def Train_Model(train_data, val_data, epochs):
    global model, scores, steps, save_path
    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=0.001, verbose=0, mode="min", patience=20)
    checkpoint = ModelCheckpoint(save_path+'best_model(ResNetv2).h5',
                                 monitor="val_loss", verbose=0, save_best_only=True, mode="min")
 # scores = model.fit_generator(train_data,validation_data=val_data, steps_per_epoch=steps, validation_steps=steps, epochs=epochs,callbacks=[early_stop,checkpoint])
    scores = model.fit_generator(train_data, validation_data=val_data, steps_per_epoch=steps,
                                 validation_steps=steps, epochs=epochs, callbacks=[checkpoint])
    return scores

# make predictions on the testing set


def Evaluate_Model(dataset):
    global model, scores
    if scores != None:
        fig = plt.figure(figsize=(16, 4))
        ax = fig.add_subplot(121)
        ax.plot(scores.history["val_loss"])
        ax.set_title("loss")
        ax.set_xlabel("epochs")

        ax2 = fig.add_subplot(122)
        ax2.plot(scores.history["val_accuracy"])
        ax2.set_title("validation accuracy")
        ax2.set_xlabel("epochs")
        ax2.set_ylim(0, 1)
        plt.show()
    loss, acc = model.evaluate_generator(dataset)
    print(f'Loss: {loss} ::::::::> Acc: {acc}')


def Execute_Model():
    global train_dataset, val_dataset, EPOCHS, model, test_dataset
    print('Compilar')
    Compile_Model()
    print('Pré-teste')
    # Evaluate_Model(test_dataset)
    print('Treinar')
    Train_Model(train_dataset, val_dataset, EPOCHS)
    print('Pós-teste')
    Evaluate_Model(test_dataset)


def Init_Model():
    Compile_Model()
    model.load_weights(CAMINHO_MODELO)


def predict(image):
    result = model.predict(image)
    return result


def predict_processing(image):
    result = model.predict(image)
    return result
