from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential

import pandas as pd
import os


class RegressionResnetModel:
    def __init__(self):
        self.model = Sequential()

    def build(self,  input_shape, optimizer, num_of_hidden_layers, activation_rule):
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.num_of_hidden_layers = num_of_hidden_layers
        self.activation_rule = activation_rule

        resnet_model = VGGFace(model='resnet50', weights='vggface', include_top=False, input_shape=self.input_shape)
        resnet_model.trainable = False
        self.model.add(resnet_model)
        self.model.add(Flatten())
        self.model.add(Dense(self.num_of_hidden_layers))
        self.model.add(Activation(self.activation_rule))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mse', optimizer=self.optimizer,
                           metrics=['mse'])

    def load_weights(self, path):
        self.model = load_model(path)

    def train(self, train=None, val=None, test=None, path=None, batch_size=32, epochs=5):
        self.batch_size = batch_size

        self.train_df = pd.read_csv(train, sep=" ")
        self.test_df = pd.read_csv(test, sep=" ")
        self.val_df = pd.read_csv(val, sep=" ")

        check_point_callback = "resnet_ep={}_opt={}__act={}_hid_l={}.h5".format(epochs, self.optimizer,
                                                                                self.activation_rule,
                                                                                self.num_of_hidden_layers)
        path = os.path.join(os.getcwd(), "weights")
        path = os.path.join(path, check_point_callback)

        self.callbacks = [ModelCheckpoint(path,
                                          monitor='val_loss',
                                          save_best_only=True)]
        print(self.input_shape[0], self.input_shape[1])

        self.train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                                rotation_range=30,
                                                width_shift_range=0.1,
                                                height_shift_range=0.1,
                                                shear_range=0.1,
                                                zoom_range=0.1,
                                                horizontal_flip=True)
        self.datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.train_generator = self.train_datagen.flow_from_dataframe(
            dataframe=self.train_df,
            directory=path,
            x_col="name",
            y_col="score",
            batch_size=self.batch_size,
            shuffle=False,
            class_mode="other",
            target_size=(self.input_shape[0], self.input_shape[1]))
        self.val_generator = self.datagen.flow_from_dataframe(
            dataframe=self.val_df,
            directory=path,
            x_col="name",
            y_col="score",
            batch_size=self.batch_size,
            shuffle=False,
            class_mode="other",
            target_size=(self.input_shape[0], self.input_shape[1]))
        self.test_generator = self.datagen.flow_from_dataframe(
            dataframe=self.test_df,
            directory=path,
            x_col="name",
            y_col="score",
            batch_size=self.batch_size,
            shuffle=False,
            class_mode="other",
            target_size=(self.input_shape[0], self.input_shape[1]))
        self.model.fit_generator(
            generator=self.train_generator,
            steps_per_epoch=self.train_generator.n // self.batch_size,
            epochs=epochs,
            callbacks=self.callbacks,
            validation_data=self.val_generator,
            validation_steps=self.val_generator.n // self.batch_size)

    def evaluate(self, path, test=None, batch_size=32, input_shape=(224, 224, 3)):
        self.test_df = pd.read_csv(test, sep=" ")
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.test_generator = self.datagen.flow_from_dataframe(
            dataframe=self.test_df,
            directory=path,
            x_col="name",
            y_col="score",
            batch_size=self.batch_size,
            shuffle=False,
            class_mode="other",
            target_size=(self.input_shape[0], self.input_shape[1]))
        scores = self.model.evaluate_generator(self.test_generator,
                                               self.test_generator.n // self.batch_size)
        print("Mean square error = ", scores[1])
