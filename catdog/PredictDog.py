from keras.models import load_model #加载模型
import matplotlib.image as processimage #预处理图片库
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class Prediction(object):
    #初始化函数
    def __init__(self,ModelFile,PredictFile,DogType,Width = 100,Height = 100,):
        self.model_file = ModelFile
        self.predict_file = PredictFile
        self.Width = Width
        self.Height = Height
        self.DogType = DogType

    #预测
    def Predict(self):
        #引入model
        model = load_model(self.model_file)

        #处理照片格式和尺寸
        img_open = Image.open(self.predict_file)
        conv_RGB = img_open.convert("RGB")
        new_img = conv_RGB.resize((self.Width,self.Height),Image.BILINEAR)
        new_img.save(self.predict_file) #覆盖掉了
        print("image resized")

        #处理照片shape
        image = processimage.imread(self.predict_file)
        image_to_array = np.array(image)/255.0 #转成float
        image_to_array = image_to_array.reshape(1,100,100,3)
        print("image reshaped")

        #预测照片
        prediction = model.predict(image_to_array)
        Final_Pred = [result.argmax() for result in prediction]
        # print(Final_Pred)
        # print(prediction)
        # print(prediction[0])

        #延伸教程读取概率
        count = 0
        for i in prediction[0]:
            percent = "%.2f%%"%(i*100)
            print(self.DogType[count],"概率",percent)
            count += 1

    def ShowPredImg(self):
        pass


DogType = ["哈士奇", "德国牧羊犬", "拉布拉多", "萨摩耶犬"]

#实例化
Pred = Prediction(PredictFile = "predict_image/4.jpg",ModelFile = "dogfinder.h5",DogType = DogType)
Pred.Predict()

#调用类
