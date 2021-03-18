import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as processimage

#load the train model
model = load_model("model_name.h5")

class MainPredictImg(object):
    def __init__(self):
        pass

    def pred(self,filename):
        #np array
        pred_img = processimage.imread(filename)
        pred_img = np.array(pred_img)
        pred_img = pred_img.reshape(-1,28,28,1)
        prediction = model.predict(pred_img)
        Final_Predict = [result.argmax() for result in prediction]
        a = 0
        for i in prediction[0]:
            print(a)
            print("precent:{:.10%}".format(i))
            a = a+1
        return Final_Predict

    def main(self):
        Predict = MainPredictImg()
        res = Predict.pred("predict_image/7.jpg")
        print("your number is ",res)

if __name__ == "__main__":
    main()