import flwr as fl
import tensorflow as tf
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten
from keras.models import Model

# Auxiliary methods
def getDist(y):
    ax = sns.countplot(x=y)
    ax.set(title="Count of data classes")
    plt.show()

def getData(dist, x, y):
    dx = []
    dy = []
    counts = [0 for i in range(10)]
    for i in range(len(x)):
        if counts[y[i]]<dist[y[i]]:
            dx.append(x[i])
            dy.append(y[i])
            counts[y[i]] += 1
    return np.array(dx), np.array(dy)

# Load and compile Keras model
vgg = VGG19(weights='imagenet', include_top=False, input_shape=(112, 112, 3))
# Freeze first 10 layers
for layer in vgg.layers[:10]:
    layer.trainable = False
x = vgg.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=predictions)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.repeat(np.expand_dims(x_train, axis=-1), 3, axis=-1)
x_test = np.repeat(np.expand_dims(x_test, axis=-1), 3, axis=-1)
x_train = tf.image.resize(x_train, (112, 112))
x_test = tf.image.resize(x_test, (112, 112))
dist = [0, 10, 10, 10, 1000, 500, 1000, 1000, 10, 4500]

x_train, y_train = getData(dist, x_train, y_train)

# Verify y_train contains multiple classes
print(np.unique(y_train))

# Visualize data distribution
getDist(y_train)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        print("Get parameters")
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0)
        hist = r.history
        print("Fit history : " ,hist)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Eval accuracy : ", accuracy)
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)




