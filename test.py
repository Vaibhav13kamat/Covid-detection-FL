# import tensorflow as tf
# # tf.config.list_physical_devices('GPU')
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[1:],'GPU')
# print(tf.test.is_gpu_available())

Python3 import tensorflow as tf tf.config.list_physical_devices ('GPU')

# import flwr as fl
# import tensorflow as tf
# from tensorflow import keras
# import sys
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np


# # AUxillary methods
# def getDist(y):
#     ax = sns.countplot(y)
#     ax.set(title="Count of data classes")
#     plt.show()

# def getData(dist, x, y):
#     dx = []
#     dy = []
#     counts = [0 for i in range(10)]
#     for i in range(len(x)):
#         if counts[y[i]]<dist[y[i]]:
#             dx.append(x[i])
#             dy.append(y[i])
#             counts[y[i]] += 1
        
#     return np.array(dx), np.array(dy)



# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
# dist = [4000, 4000, 4000, 3000, 10, 10, 10, 10, 4000, 10]
# x_train, y_train = getData(dist, x_train, y_train)
# print("here")
# getDist(y_train)