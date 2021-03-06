import os
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Dense, Activation
from keras.optimizers import Adam
from keras.utils import np_utils

#Pre process images
class PreFile(object):
    def __init__(self,FilePath,DogType):
        self.FilePath = FilePath
        self.DogType = DogType

    def FileReName(self):
        type_counter = 0
        for type in self.DogType:
            file_counter = 0
            subfolder = os.listdir(self.FilePath + type)
            for subclass in subfolder:
                file_counter += 1
                # print("file_counter:",file_counter)
                # print("type_counter:",type_counter)
                # print(subclass)
                os.rename(self.FilePath + type + "/" + subclass,self.FilePath + type + "/" + str(type_counter) + "_" + str(file_counter) + "_" + type + ".jpg")
            type_counter += 1
        print("rename finish!")

    def FileResize(self,Width,Height,Output_folder):
        for type in self.DogType:
            subfolder = os.listdir(self.FilePath + type)
            for subclass in subfolder:
                img_open = Image.open(self.FilePath + type + "/" + str(subclass))
                conv_RGB = img_open.convert("RGB") # 统一转化成RGB格式
                Resize_img = conv_RGB.resize((Width, Height), Image.BILINEAR)
                Resize_img.save(os.path.join(Output_folder, os.path.basename(subclass)))
        print("resize finish!")


class Training(object):
    def __init__(self,batch_size,num_batch,categorizes,train_folder):
        self.batch_size = batch_size
        self.number_batch = num_batch
        self.categories = categorizes
        self.train_folder = train_folder

    def read_train_images(self,filename):
        img = Image.open(self.train_folder+filename)
        return np.array(img)

    def train(self):
        train_image_list = []
        train_label_list = []
        for file in os.listdir(self.train_folder):
            files_img_in_array = self.read_train_images(filename = file)
            train_image_list.append(files_img_in_array)
            train_label_list.append(int(file.split("_")[0]))
            # print(Train_list_label)

        train_image_list = np.array(train_image_list)
        train_label_list = np.array(train_label_list)

        print(train_image_list.shape)
        print(train_label_list.shape)

        train_label_list = np_utils.to_categorical(train_label_list,self.categories)
        train_image_list = train_image_list.astype("float32")
        train_image_list /=255.0

        model = Sequential()

        # CNN Layer —— 1
        model.add(Convolution2D( # input shape:(100,100,3)
            input_shape = (100,100,3),
            filters = 32, # next layer output:(100,100,32)
            kernel_size = (5,5), # pixel filtered
            padding = "same", # 外边距处理
        ))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(
            pool_size = (2,2), # next layer output:(50,50,32)
            strides = (2,2),
            padding = "same"
        ))

        # CNN Layer —— 2
        model.add(Convolution2D(
            filters = 64, # next layer output:(50,50,64)
            kernel_size = (2,2), # pixel filtered
            padding = "same", # 外边距处理
        ))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(
            pool_size = (2,2), # next layer output:(25,25,64)
            strides = (2,2),
            padding = "same"))

        # Fully connected Layer —— 1
        model.add(Flatten()) # 降维
        model.add(Dense(1024))
        model.add(Activation("relu"))
        # Fully connected Layer —— 2
        model.add(Dense(512))
        model.add(Activation("relu"))
        # Fully connected Layer —— 3
        model.add(Dense(256))
        model.add(Activation("relu"))
        # Fully connected Layer —— 4
        model.add(Dense(self.categories))
        model.add(Activation("softmax"))

        # Define Optimizer
        adam = Adam(lr = 0.0001)

        # Compile the model
        model.compile(optimizer = adam,
                      loss = "categorical_crossentropy",
                      metrics = ["accuracy"])

        #Fire up the network
        model.fit(
            x = train_image_list,
            y = train_label_list,
            epochs = self.number_batch,
            batch_size = self.batch_size,
            verbose = 1)

        # Save your work model
        model.save("./dogfinder.h5")

def MAIN():

    DogType = ["哈士奇","德国牧羊犬","拉布拉多","萨摩耶犬"]

    # # File pre processing
    # FILE = PreFile(FilePath = "raw_image/",DogType = DogType)
    #
    # # File rename and resize
    # FILE.FileReName()
    # FILE.FileResize(Height = 100, Width = 100, Output_folder = "train_image/")

    # Train the Network
    Train = Training(batch_size = 128, num_batch = 30, categorizes = 4, train_folder = "train_image/")
    Train.train()


if __name__ == "__main__":
    MAIN()





