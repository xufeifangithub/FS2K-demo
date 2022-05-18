import json
import os
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    __init__(): 读取图片、json 文件，制作数据以及标签列表
    __len__(): 获取数据集dataSet的长度
    __getitem__()：打开index对应图片进行预处理后return回处理后的图片和标签
    """
    def __init__(self, data_url, json_file):
        """
        :param data_url: 数据在的文件夹的路径
        :param json_file: 对应数据集的json文件所在的路径
        """

        with open(json_file, 'r') as f:
            json_data = json.loads(f.read())

        attrs = {}
        for attr in json_data[0].keys():
            attrs[attr] = []
        for idx_fs, fs in enumerate(json_data):
            for attr in fs:
                attrs[attr].append(fs[attr])

        file_list = os.listdir(data_url)
        datas = []
        for name in file_list:
            arr = cv2.imread(data_url + name)
            datas.append(arr)

        keys = list(attrs.keys())[4:]

        dataset = []
        print("hair中负样本占比：{:.2f}".format(attrs['hair'].count(0) / len(attrs['hair'])))
        print("gender中负样本占比:{:.2f}".format(attrs['gender'].count(0) / len(attrs['hair'])))
        print("earring中负样本占比{:.2f}".format(attrs['earring'].count(0) / len(attrs['hair'])))
        print("smile中负样本占比{:.2f}".format(attrs['smile'].count(0) / len(attrs['hair'])))
        print("frontal_face中负样本占比{:.2f}".format(attrs['frontal_face'].count(0) / len(attrs['hair'])))

        for i in range(0, len(datas)):
            data = cv2.resize(datas[i], (150, 150), interpolation=cv2.INTER_CUBIC)
            # data = transforms.ToTensor(data)
            label = []
            for key in keys:
                if key != 'style' and key != 'hair_color':
                    label.append(attrs[key][i])
            label = np.array(label)
            label = label.reshape(5, 1)
            label = torch.from_numpy(label)
            # label = attrs["gender"][i]
            # label = np.array(label)
            dataset.append((data, label))

        # dataset = np.array(dataset)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pic, label = self.dataset[index]
        return pic, label


if __name__ == '__main__':
    data = MyDataset('../FS2K/train/photo/', '../FS2K/anno_train.json')
    print(data[100][0].shape)
    print(data[0][1].shape)

    plt.imshow(data[100][0])
    plt.show()
