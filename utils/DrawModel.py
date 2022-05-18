import hiddenlayer as h
import torch
from torchviz import make_dot


def DrawModel(model, path, name):
    """

    :param model: 需要绘制结构和计算过程的模型
    :param path: 保存绘制图片的路径
    :param name: 网络的名称
    :return: 无
    """
    vis_graph = h.build_graph(model, torch.zeros(4, 3, 150, 150))   # 获取绘制图像的对象
    vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
    vis_graph.save(path=path + "/" + name + "_model_structure.png", format='png')

    x = torch.randn(3, 3, 150, 150).requires_grad_(True)  # 定义一个网络的输入值
    y = model(x)  # 获取网络的预测值

    gra = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
    gra.format = "png"
    # 指定文件生成的文件夹
    gra.directory = path
    # 生成文件
    gra.view()