import numpy as np
import pickle
import LoadMnist
import tensorflow as tf
from tensorflow import keras as ke
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from History import HistoryPlotCallback
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
trainX, trainY, testX, testY = LoadMnist.GetData(128, 50000)

print("trainX shape: ", trainX.shape)
print("trainY shape: ", trainY.shape)

# One hot encoder
trainY = to_categorical(trainY)
testY = to_categorical(testY)
print(trainX.shape, testX.shape, trainY.shape, testY.shape)
print(trainY[0])
print(trainY[1])

n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]
print("n_timesteps: ", n_timesteps)
print("n_features: ", n_features)
print("n_outputs: ", n_outputs)

inputs = tf.keras.Input(shape=(128, 2))
x = Conv1D(filters=150, kernel_size=5, input_shape=(n_timesteps, n_features))(inputs)
#x = Conv1D(filters=100, kernel_size=3)(x)
x = Dropout(0.8)(x)
x = Conv1D(filters=100, kernel_size=4)(x)
x = Dropout(0.5)(x)
x = Conv1D(filters=80, kernel_size=3)(x)
x = Dropout(0.3)(x)
x = Conv1D(filters=40, kernel_size=2)(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(100, activation='relu')(x)
embeddings = layers.Dense(2)(x)
model_embeddings = tf.keras.Model(inputs=inputs, outputs=embeddings)
print(model_embeddings.summary())
outputs = Dense(10, activation='softmax')(embeddings)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="test_model")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs=300, batch_size=1500, validation_data=(testX, testY), verbose=1, callbacks=[HistoryPlotCallback()])

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('model accuracy')
# ax1.ylabel('accuracy')
# ax1.xlabel('epoch')
ax1.legend(['train', 'test'], loc='upper left')
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
# ax2.ylabel('loss')
# ax2.xlabel('epoch')
ax2.legend(['train', 'test'], loc='upper left')
f.savefig('acc_loss.png')
