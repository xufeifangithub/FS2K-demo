import numpy as np
import torch
from utils.metrics import metrics


def model_test(model, test_load, device, name):

    hair_pre,  gender_pre, earring_pre, smile_pre, front_face_pre = [], [], [], [], []
    hair_label, gender_label, earring_label, smile_label, front_face_label = [], [], [], [], []
    with torch.no_grad():
        model.eval()
        for idx, (data, label) in enumerate(test_load):

            data = data.view((-1, data.shape[3], data.shape[1], data.shape[2]))
            data = data.float()
            data = data.to(device)

            hair, gender, earring, smile, front_face = model(data)

            label = label.to(torch.int64)
            label = label.view(label.shape[0], label.shape[1])
            label = label.T

            label = label.numpy().tolist()
            _, hair = torch.max(hair, dim=1)
            hair = hair.cpu().numpy().tolist()
            _, gender = torch.max(gender, dim=1)
            gender = gender.cpu().numpy().tolist()
            _, earring = torch.max(earring, dim=1)
            earring = earring.cpu().numpy().tolist()
            _, smile = torch.max(smile, dim=1)
            smile = smile.cpu().numpy().tolist()
            _, front_face = torch.max(front_face, dim=1)
            front_face = front_face.cpu().numpy().tolist()

            hair_pre += hair
            gender_pre += gender
            earring_pre += earring
            smile_pre += smile
            front_face_pre += front_face

            hair_label += label[0]
            gender_label += label[1]
            earring_label += label[2]
            smile_label += label[3]
            front_face_label += label[4]

        hair_pre = np.array(hair_pre)
        gender_pre = np.array(gender_pre)
        earring_pre = np.array(earring_pre)
        smile_pre = np.array(smile_pre)
        front_face_pre = np.array(front_face_pre)

        hair_label = np.array(hair_label)
        gender_label = np.array(gender_label)
        earring_label = np.array(earring_label)
        smile_label = np.array(smile_label)
        front_face_label = np.array(front_face_label)

        labels = ['0', '1']
        title = ['hair', 'gender', 'earring', 'smile', 'front_face']
        hair_precision, hair_recall, hair_f1 = metrics(hair_pre, hair_label, labels, name + title[0])
        gender_precision, gender_recall, gender_f1 = metrics(gender_pre, gender_label,labels, name +title[1])
        earring_precision, earring_recall, earring_f1 = metrics(earring_pre, earring_label, labels, name +title[2])
        smile_precision, smile_recall, smile_f1 = metrics(smile_pre, smile_label, labels, name +title[3])
        front_face_precision, front_face_recall, front_face_f1 = metrics(front_face_pre, front_face_label,
                                                                         labels, name +title[4])

        acc_hair = (hair_pre == hair_label).sum() / len(hair_label) * 100
        acc_gender = (gender_pre == gender_label).sum() / len(gender_label) * 100
        acc_earring = (earring_pre == earring_label).sum() / len(earring_label) * 100
        acc_smile = (smile_pre == smile_label).sum() / len(smile_label) * 100
        acc_front_face = (front_face_pre == front_face_label).sum() / len(front_face_label) * 100

        print("{0: >15} | precision:{1: >15.2f} | recall:{2:>15.2f} | f1:{3:<15.2f} | acc:{4:>15.2f}".format(
            'hair', hair_precision, hair_recall, hair_f1, acc_hair))
        print("{0: >15} | precision:{1: >15.2f} | recall:{2:>15.2f} | f1:{3:<15.2f} | acc:{4:>15.2f}".format(
            'gender', gender_precision, gender_recall, gender_f1, acc_gender))
        print("{0: >15} | precision:{1: >15.2f} | recall:{2:>15.2f} | f1:{3:<15.2f} | acc:{4:>15.2f}".format(
            'earring', earring_precision, earring_recall, earring_f1, acc_earring))
        print("{0: >15} | precision:{1: >15.2f} | recall:{2:>15.2f} | f1:{3:<15.2f} | acc:{4:>15.2f}".format(
            'smile', smile_precision, smile_recall, smile_f1, acc_smile))
        print("{0: >15} | precision:{1: >15.2f} | recall:{2:>15.2f} | f1:{3:<15.2f} | acc:{4:>15.2f}".format(
            'front_face', front_face_precision, front_face_recall, front_face_f1, acc_front_face))

