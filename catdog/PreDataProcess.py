import os
import numpy as np
from PIL import Image

#重新命名
def FileReName(DogType,FilePath):
    type_counter = 0
    for type in DogType:
        file_counter = 0
        subfolder = os.listdir(FilePath + type)
        for subclass in subfolder:
            file_counter += 1
            # print("file_counter:",file_counter)
            # print("type_counter:",type_counter)
            # print(subclass)
            os.rename(FilePath + type + "/" + subclass,FilePath + type + "/" + str(type_counter) + "_" + str(file_counter) + "_" + type + ".jpg")
        type_counter += 1
    print("rename finish!")

#重新定义图片尺寸
def FileResize(Output_folder,DogType,FilePath,Width = 100,Height = 100):
    for type in DogType:
        subfolder = os.listdir(FilePath + type)
        for subclass in subfolder:
            img_open = Image.open(FilePath + type + "/" + str(subclass))
            conv_RGB = img_open.convert("RGB")
            Resize_img = conv_RGB.resize((Width,Height),Image.BILINEAR)
            Resize_img.save(os.path.join(Output_folder,os.path.basename(subclass)))
    print("resize finish!")


#读取图片返回array数据
def ReadImage(filename,train_folder):
    img = Image.open(train_folder+filename)
    #把照片转换成nuupy数组
    return np.array(img)

#图片加载到列表 图像和标签
def DataSet(train_folder):
    Train_list_img = []
    Train_list_label = []

    for file_1 in os.listdir(train_folder):
        file_img_to_array = ReadImage(file_1,train_folder)
        #添加图片数组到主list里
        Train_list_img.append(file_img_to_array)
        #添加标签数组到主list里
        Train_list_label.append(int(file_1.split("_")[0]))
        # print(Train_list_label)

    Train_list_img = np.array(Train_list_img)
    Train_list_label = np.array(Train_list_label)

    print("dataset finish!")
    print(Train_list_img.shape)
    print(Train_list_label.shape)


if __name__ == "__main__":

    DogType = ["哈士奇","德国牧羊犬","拉布拉多","萨摩耶犬"]

    # #修改名字
    # FileReName(DogType = DogType,FilePath = "raw_image/")
    #
    # #修改尺寸
    # FileResize(DogType = DogType,FilePath = "raw_image/",Output_folder = "train_image/")
    #
    # #准备好的数据
    # DataSet(train_folder = "train_image/")
