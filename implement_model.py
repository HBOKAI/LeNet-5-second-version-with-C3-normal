import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses, optimizers, datasets
# from PIL import Image
import numpy as np
import cv2
import time

def nothing(X):
    pass

def show_xy(event,x,y,flags,param):
    global dots, draw,img_gray                    # 定義全域變數
    if flags == 1:
        if event == 1:
            dots.append([x,y])            # 如果拖曳滑鼠剛開始，記錄第一點座標
        if event == 4:
            dots = []                     # 如果放開滑鼠，清空串列內容
        if event == 0 or event == 4:
            dots.append([x,y])            # 拖曳滑鼠時，不斷記錄座標
            x1 = dots[len(dots)-2][0]     # 取得倒數第二個點的 x 座標
            y1 = dots[len(dots)-2][1]     # 取得倒數第二個點的 y 座標
            x2 = dots[len(dots)-1][0]     # 取得倒數第一個點的 x 座標
            y2 = dots[len(dots)-1][1]     # 取得倒數第一個點的 y 座標
            cv2.line(draw,(x1,y1),(x2,y2),(255,255,255),20)  # 畫直線
        cv2.imshow('img', draw)#draw

class LeNet_Model(tf.keras.Model):
    def __init__(self, output_num=10, **kwargs):
        super(LeNet_Model,self).__init__(**kwargs)
        self.convolution_1 = layers.Conv2D(6,kernel_size=5,strides=1)
        self.convolution_2 = layers.Conv2D(16,kernel_size=5,strides=1)
        self.Dense_1 = layers.Dense(120,activation='relu')
        self.Dense_2 = layers.Dense(84,activation='relu')
        self.output_layer = layers.Dense(10,activation='softmax')
        self.activation = layers.Activation('relu')
        self.averagepooling = layers.AveragePooling2D(pool_size=2,strides=2)
        self.flatten = layers.Flatten()
    def call(self,inputs,**kwargs):
        x = self.convolution_1(inputs)
        x = self.activation(x)
        x = self.averagepooling(x)
        x = self.convolution_2(x)
        x = self.activation(x)
        x = self.averagepooling(x)
        x = self.flatten(x)
        x = self.Dense_1(x)
        x = self.Dense_2(x)
        output_ = self.output_layer(x)
        return output_
    
# model = tf.keras.models.load_model("./CNN_MODEL_40_0.3_0.5_0.5") # tt/.h5
# img = Image.open("D:/Desktop/學校/實驗室/程式/TENSORFLOW/MNIST數據/test_images/6/6.5.jpg")
model = LeNet_Model()
model.load_weights('./SAVE_WEIGHTS/SW')

a = input("輸入使用模式 img or write: ")
if (a == "img"):
    img = cv2.imread("./images/testimg6.png",flags=0)
    show_img = cv2.resize(img,(320,320))
    cv2.imshow("img",show_img)
    cv2.namedWindow('img')
    cv2.createTrackbar('Num','img',0,9,nothing)
    img = np.array(img)/255
    img = np.expand_dims(img,0)
    img = np.expand_dims(img,-1) 
    while True:
        keyboard = cv2.waitKey(5)
        innum = cv2.getTrackbarPos('Num','img')
        if keyboard == ord('n'):
            start = time.time()
            predict = model.call(img)
            end = time.time()
            predict_num = np.argmax(predict, axis=-1)
            print('預測結果:\n',predict_num)
            print(f"預測值: {predict_num[0]}, 正確值: {innum}, 耗時: {end-start}")
        if keyboard == ord('q'):
            break
else:
    dots = []   # 建立空陣列記錄座標
    w = 320
    h = 320
    draw = np.zeros((h,w,3), dtype='uint8')   # 建立 420x240 的 RGBA 黑色畫布
    while True:
        cv2.imshow('img', draw)
        cv2.setMouseCallback('img', show_xy)
        keyboard = cv2.waitKey(5)                    # 每 5 毫秒偵測一次鍵盤事件
        if keyboard == ord('q'):
            break                                    # 按下 q 就跳出

        if keyboard == ord('n'):
            img_gray = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)   # 轉為灰度圖
            img = cv2.resize(img_gray,(32,32))                          # 變更圖片尺寸
            cv2.imwrite(".\images\gray.png",img)
            img = img/255
            img = np.expand_dims(img,0)
            img = np.expand_dims(img,-1)
            # np.savetxt("show_data.txt",img[0,...,0],fmt='%.01f')

            start = time.time()
            # predict = model.predict(img)
            predict = model.call(img)
            end = time.time()
            predict_num = np.argmax(predict, axis=-1)
            print('預測結果: ',predict_num)
            print('花費時間: ',end-start)
            draw = np.zeros((h,w,3), dtype='uint8')

        if keyboard == ord('r'):
            draw = np.zeros((h,w,3), dtype='uint8')  # 按下 r 就變成原本全黑的畫布
            cv2.imshow('img', draw)