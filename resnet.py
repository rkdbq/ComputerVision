import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Activation, MaxPooling2D, Add, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class Residual_block(tf.keras.Model):
    def __init__(self, filters, kernel_size, activation="relu", strides=1, padding='same', projection=False):
        super(Residual_block, self).__init__()

        self.projection = projection

        self.conv1 = Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_regularizer=l2(0.001))
        self.bn1 = BatchNormalization()
        self.activation1 = Activation(activation)
        self.conv2 = Conv2D(filters, kernel_size, padding=padding, kernel_regularizer=l2(0.001))
        self.bn2 = BatchNormalization()
        self.activation2 = Activation(activation)
        self.conv_projection = Conv2D(filters, (1, 1), strides=strides, padding=padding, kernel_regularizer=l2(0.001))
        self.bn_projection = BatchNormalization()
        self.add = Add()
        self.activation_add = Activation(activation)

    def call(self, x):
        shortcut = x
        if self.projection:
            shortcut = self.conv_projection(shortcut)
            shortcut = self.bn_projection(shortcut)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = self.add([x, shortcut])
        x = self.activation_add(x)
        
        return x

with tf.device("/gpu:0"):
    (x_train,y_train),(x_test,y_test)=ds.cifar10.load_data()
    x_train=x_train.astype(np.float32)/255.0
    x_test=x_test.astype(np.float32)/255.0
    y_train=tf.keras.utils.to_categorical(y_train,10)
    y_test=tf.keras.utils.to_categorical(y_test,10)
    
    resnet=Sequential()
    resnet.add(Conv2D(64, (7, 7), strides=(2, 2), padding='same', input_shape=(32,32,3), kernel_regularizer=l2(0.001)))
    resnet.add(BatchNormalization())
    resnet.add(Activation("relu"))
    resnet.add(MaxPooling2D(pool_size=(2,2)))
    resnet.add(Residual_block(64, (3, 3), strides=2, padding='same', projection=True))
    resnet.add(Residual_block(64, (3, 3)))
    resnet.add(Residual_block(64, (3, 3)))
    resnet.add(Dropout(0.2))

    resnet.add(Residual_block(128, (3, 3), strides=2, padding='same', projection=True))
    resnet.add(Residual_block(128, (3, 3)))
    resnet.add(Residual_block(128, (3, 3)))
    resnet.add(Residual_block(128, (3, 3)))
    resnet.add(Dropout(0.3))

    resnet.add(Residual_block(256, (3, 3), strides=2, padding='same', projection=True))
    resnet.add(Residual_block(256, (3, 3)))
    resnet.add(Residual_block(256, (3, 3)))
    resnet.add(Residual_block(256, (3, 3)))
    resnet.add(Residual_block(256, (3, 3)))
    resnet.add(Residual_block(256, (3, 3)))
    resnet.add(Dropout(0.4))

    resnet.add(Residual_block(512, (3, 3), strides=2, padding='same', projection=True))
    resnet.add(Residual_block(512, (3, 3)))
    resnet.add(Residual_block(512, (3, 3)))
    resnet.add(GlobalAveragePooling2D())
    resnet.add(Dropout(0.5))

    resnet.add(Flatten())
    resnet.add(Dense(units=512,activation='relu'))
    resnet.add(Dropout(0.5))
    resnet.add(Dense(units=10,activation='softmax'))
    
    # 컴파일 및 학습
    resnet.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    hist = resnet.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=1)

    # 평가
    res = resnet.evaluate(x_test, y_test, verbose=0)
    print('정확률=', res[1]*100)

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()