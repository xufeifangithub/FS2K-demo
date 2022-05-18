from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import torch


def plot_confusion(cm, label, title):
    """
    该方法绘制混淆矩阵
    :param cm:混淆矩阵
    :param label: 横坐标纵坐标
    :param title: 对应混淆矩阵的名称
    :return: 无
    """
    plt.figure()
    plot_confusion_matrix(cm, figsize=(12, 8), cmap=plt.cm.Blues) # 参数设置
    plt.xticks(range(len(label)), label, fontsize=14)
    plt.yticks(range(len(label)), label, fontsize=14)
    plt.xlabel('predicted label', fontsize=16)
    plt.ylabel('true label', fontsize=16)
    plt.title(title)
    plt.savefig('./result/' + title + '.png')


def metrics(preds, labels, label, title):
    """
    :param preds: 预测值
    :param labels: 真实值
    :param label: 混淆矩阵的横纵坐标
    :param title: 对应混淆矩阵的名称
    :return: precision， recall， F1
    """
    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    # 绘制混淆矩阵
    plot_confusion(cm, label, title)
    # 获取tn, fp, fn, tp
    tn, fp, fn, tp = cm.ravel()
    # print(tn, fp, fn, tp)
    # 精准率
    precision = tp / (tp + fp)
    # 召回率
    recall = tp / (tp + fn)
    # f1 score
    f1 = 2 * ((precision * recall) / (precision + recall))
    return precision, recall, f1
