import flwr as fl
import tensorflow as tf
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

#controller code
from controller import client1_weights
from controller import include_top
from controller import input_shape
from controller import client1_number_of_classes

from controller import client1_training_dir
from controller import client1_testing_dir
from controller import client1_batch_size
from controller import client1_epochs
from controller import client1_verbose
from controller import client1_grpc_max_message_length

# Auxiliary methods
def getDist(y):
    ax = sns.countplot(x=y)
    ax.set(title="Count of data classes")
    plt.show()

# Load and compile Keras model
vgg = VGG19(weights=client1_weights, include_top=include_top, input_shape=input_shape)      #called in the controller
# Freeze first 10 layers
for layer in vgg.layers[:10]:
    layer.trainable = False
x = vgg.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(client1_number_of_classes, activation='softmax')(x)  # change number of classes to 2 for covid and normal  #called in controller
model = Model(inputs=vgg.input, outputs=predictions)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load dataset
train_dir = client1_training_dir
test_dir = client1_testing_dir
batch_size = client1_batch_size
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=batch_size, class_mode='sparse')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=batch_size, class_mode='sparse')

# Visualize data distribution
getDist(train_generator.classes)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        print("Get parameters")
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(train_generator, epochs=client1_epochs, validation_data=test_generator, verbose=client1_verbose)
        hist = r.history
        print("Fit history : " ,hist)
        return model.get_weights(), train_generator.n, {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_generator, verbose=0)
        print("Eval accuracy : ", accuracy)
        return loss, test_generator.n, {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]),  #not added in controller
        client=FlowerClient(), 
        grpc_max_message_length = client1_grpc_max_message_length * client1_grpc_max_message_length * client1_grpc_max_message_length
)
