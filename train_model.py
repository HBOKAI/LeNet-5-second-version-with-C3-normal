import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses, optimizers, datasets
import numpy as np
import matplotlib.pyplot as plt
import os

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)


(train_x, train_y),(test_x, test_y) = tf.keras.datasets.mnist.load_data()


train_x = 2*tf.convert_to_tensor(train_x,dtype=tf.float32)
train_x = tf.pad(train_x,[[0,0],[2,2],[2,2]],"CONSTANT",0) # 外圍填充
train_x = train_x / 255 # 圖像歸一化 0~1
train_x = tf.expand_dims(train_x,-1)
train_y = tf.one_hot(train_y, depth=10)

test_x = 2*tf.convert_to_tensor(test_x,dtype=tf.float32)
test_x = tf.pad(test_x,[[0,0],[2,2],[2,2]],"CONSTANT",0) # 外圍填充
test_x = test_x / 255 # 圖像歸一化 0~1
test_x = tf.expand_dims(test_x,-1)
test_y = tf.one_hot(test_y, depth=10)

# print(train_x[1])
print(train_x.shape, train_y.shape)
model = Sequential([
    layers.Conv2D(6, kernel_size=5, strides=1),  # 第一个卷积核，6个5x5的卷积核，
    # layers.ReLU(),  # 激活函数
    layers.Activation('relu'),
    layers.AveragePooling2D(pool_size=2, strides=2), # 高宽各减半的池化层 
    # layers.MaxPooling2D(pool_size=2, strides=2),  
    layers.Conv2D(16, kernel_size=5, strides=1),  # 第二个卷积核，16个3X3的卷积核，
    # layers.ReLU(),  # 激活函数 
    layers.Activation('relu'),
    layers.AveragePooling2D(pool_size=2, strides=2), # 高宽各减半的池化层
    # layers.MaxPooling2D(pool_size=2, strides=2), 
    layers.Flatten(),  # 打平层，方便全连接层处理
    layers.Dense(120, activation='relu'),  # 全连接层，120个结点
    layers.Dense(84, activation='relu'),  # 全连接层，84个结点
    layers.Dense(10, activation='softmax')  # 全连接层，10个结点
])
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    zoom_range=0.3,
    width_shift_range=0.5,
    height_shift_range=0.5)

model.build(input_shape=(None,32,32,1)) # 創建神經網路，定義輸入資料大小
model.summary() # 看建立的架構
tf.keras.utils.plot_model(model, "./images/my_first_model_with_shape_info.png", show_shapes=True)
model.compile(loss="categorical_crossentropy",optimizer=tf.keras.optimizers.Adam(),metrics=["accuracy"]) # 定義所要採用的loss funtion, optimizer, metrics
# history = model.fit(x=train_x,y=train_y,batch_size=32,epochs=16,verbose=1,validation_split=0.1) # 設定 batch(批), epochs(跌代), verbose, validation(驗證，功能還不太確定)
history = model.fit_generator(datagen.flow(train_x,train_y,batch_size=32),steps_per_epoch=len(train_x)/32,epochs=16,validation_data=(test_x,test_y))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
# plt.ylim([0.5, 1])
plt.legend(loc='lower right')

score = model.evaluate(test_x,test_y) #評估誤差
print("Test Loss: " ,score[0])
print("Test Accuracy: ",score[1])

result = model.predict(test_x[0:9])
print('前9筆預測結果: ',np.argmax(result, axis=-1),'\n')
print('前9筆實際值: ',np.argmax(test_y[0:9],axis=-1),'\n')
model.save('./CNN_MODEL.h5') # CNN_MODEL.h5
model.save('./CNN_MODEL')
plt.show()