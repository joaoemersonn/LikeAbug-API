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
gbifcodes = ['1035167', '1035185', '1035194', '1035195', '1035204', '1035208', '1035231', '1035290', '1035366', '1035434', '1035542', '1035551', '1035578', '1035864', '1035929', '1035931', '1036066', '1036128', '1036154', '1036192', '1036203', '1036216', '1036255', '1036286', '1036789', '1036796', '1036893', '1036899', '1036917', '1036958', '1037293', '1037319', '1037633', '4308786', '4308787', '4308789', '4308790', '4308800', '4308801', '4308804', '4308805', '4308806', '4308807', '4308811', '4308812', '4308815', '4470539', '4470555', '4470765', '4470801', '4471071', '4471113', '4471202', '4471235', '4471238', '4471269', '4472828', '4472849', '4472858', '4472884', '4472897', '4472900', '4472907', '4472913', '4472929', '4473127', '4473277', '4473297', '4473319', '4473320', '4473617', '4473795', '4473834', '4473860', '4473874', '4473903', '4473974', '4474146', '4474169', '4474653', '4474780', '4474861', '4474974', '4474998', '4475099', '4475127', '4475128', '4475140', '4475157', '4475179', '4475181', '4475188', '4475213', '4475677', '4475685', '4475707', '4475757', '4475761', '4475794', '4475846', '4475902', '4475908', '4476667', '4476832', '4478057', '4478679', '4478695', '4478699', '4478842', '4479052', '4479870', '4479890', '4479895', '4480200', '4480271', '4480302', '4480315', '4480348', '4480480', '4480485', '4480502', '4988207', '4988354', '4988363', '4988447', '4988484', '4988516', '5023047', '5023058', '5716406', '5716408', '5716409', '5716411', '5753612', '5753653', '5753697', '5753720', '5753725', '5753757', '5754974', '5755011', '5755027', '5755044', '5755051', '5755060',
             '5755066', '5755075', '5755079', '5755080', '5755082', '5755175', '5755203', '5755293', '5755296', '5755302', '5755322', '5755339', '5755451', '5755456', '5755562', '5755950', '5755952', '5755954', '5755969', '5755978', '5755982', '5755986', '5756010', '5756015', '5756018', '5756021', '5756368', '5756374', '5756814', '5756874', '5756945', '5757120', '5757155', '5757192', '5757205', '5757207', '5757211', '5757304', '5757308', '5757310', '5872109', '5872111', '5872119', '5872124', '5872127', '5872129', '5872143', '5872147', '5872148', '5872219', '5872829', '5872954', '5873211', '5873214', '5873215', '5873218', '5873220', '5873271', '5873357', '5873524', '5873525', '6097854', '6097857', '6097861', '6097862', '6097864', '6097865', '6097867', '6097868', '6097869', '6097871', '6097872', '6097878', '6097879', '6097882', '6097883', '6097885', '6097886', '6097890', '6097892', '6097896', '6097897', '6097900', '6097906', '6097912', '6097923', '6097924', '6097926', '6097927', '6097930', '7387665', '7416497', '7419401', '7508714', '7582513', '7613168', '7639821', '7641814', '7642452', '7683246', '7708350', '7727698', '7792354', '7840196', '7856746', '7873810', '7888598', '7917481', '7975487', '8019310', '8047965', '8051076', '8056040', '8068514', '8104778', '8137203', '8139233', '8186956', '8236590', '8264200', '8267077', '8300237', '8307815', '8335452', '8378220', '8417764', '8423772', '8563753', '8600499', '8607776', '9027622', '9206709', '9251010', '9252314', '9268484', '9314786', '9346940', '9364935', '9377664', '9414758', '9415910', '9444427', '9479706', '9491931', '9533851', '9581584']
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
    result = model.predict_classes(image)
    return result


def predict_processing(image):
    result = model.predict(image)
    return result


def get_GBIFCODE(index):
    return gbifcodes[index]
