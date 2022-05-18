import torch
import torch.nn as nn
import torch.optim as optim
from Model.LeNet import LeNet
from Model.Net import Net
from torch.utils.data import DataLoader
from utils import MyDataSet, train, test
from utils.DrawModel import DrawModel

# 此处为彩色图片数据集（训练集、测试集）的加载
# train_set = MyDataSet.MyDataset('./FS2K/train/photo/', './FS2K/anno_train.json')
# train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
# test_set = MyDataSet.MyDataset('./FS2K/test/photo/', './FS2K/anno_test.json')
# test_loader = DataLoader(test_set, batch_size=16, shuffle=True, num_workers=0)

# 此处为素描图片数据集（训练集、测试集）的加载
train_set = MyDataSet.MyDataset('./FS2K/train/sketch/', './FS2K/anno_train.json')
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)
test_set = MyDataSet.MyDataset('./FS2K/test/sketch/', './FS2K/anno_test.json')
test_loader = DataLoader(test_set, batch_size=8, shuffle=True, num_workers=0)


model = LeNet()
# model = Net()

# 输出模型的结构
print(model)

# 定义训练的次数
epoch = 15

# 绘制模型的结构以及计算的过程，保存在pic文件夹中
DrawModel(model, './pic', 'LeNet')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(torch.device('cpu'))
model = model.to(device)

# 使用交叉熵损失函数
loss_func = nn.CrossEntropyLoss()
"""
pkl文件命名规则为
模型_数据集_model.plk
模型为lenet 或 net
数据集为s(sketch) 或 p(photo)
"""
path = './data/lenet_s_model.pkl'

"""
优化器采用了SGD、Adam以及Adagrad通过实验，发现SGD和Adam的收敛效果并不理想，loss一直不会下降，而采用adagrad可以达到快速收敛
"""
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adagrad(model.parameters(), lr=0.001)

# 训练完模型会保存模型的参数，而不是保存整个模型
train.train(model, optimizer, loss_func,  epoch, train_loader, device, path)

# 从保存的模型本地文件读取参数
model = LeNet()
# model = Net()

model.load_state_dict(torch.load(path))
model = model.to(device)
test.model_test(model, test_loader, device, "net_s_")