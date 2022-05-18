import torch
import numpy as np


def train(model, optimizer, loss_func, epoch, train_loader, device, path):

    print("{0: >15} | {1: >15} | {2:>15} | {3: >15} | {4: >15} | {5:>15} | {6: >15}".format(
        'epoch', 'loss', 'acc_hair', 'acc_gender', 'acc_earring', 'acc_smile', 'acc_front_face'))

    model.train()
    pre_loss = float('inf')
    for i in range(epoch):
        total_loss = 0.0
        hair_pre, gender_pre, earring_pre, smile_pre, front_face_pre = [], [], [], [], []
        hair_label, gender_label, earring_label, smile_label, front_face_label = [], [], [], [], []
        for idx, (data, label) in enumerate(train_loader):

            optimizer.zero_grad()
            data = data.float()
            data, label = data.to(device), label.to(device)
            data = data.view((-1, data.shape[3], data.shape[1], data.shape[2]))

            hair, gender, earring, smile, front_face = model(data)
            label = label.to(torch.int64)
            label = label.view(label.shape[0], label.shape[1])
            label = label.T

            loss_hair = loss_func(hair, label[0])
            loss_gender = loss_func(gender, label[1])
            loss_earring = loss_func(earring, label[2])
            loss_smile = loss_func(smile, label[3])
            loss_front_face = loss_func(front_face, label[4])

            loss = loss_hair + loss_gender + loss_earring + loss_smile + loss_front_face

            loss = loss / 5

            loss.backward()
            optimizer.step()

            label = label.cpu().numpy().tolist()
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

        acc_hair = (hair_pre == hair_label).sum() / len(hair_label) * 100
        acc_gender = (gender_pre == gender_label).sum() / len(gender_label) * 100
        acc_earring = (earring_pre == earring_label).sum() / len(earring_label) * 100
        acc_smile = (smile_pre == smile_label).sum() / len(smile_label) * 100
        acc_front_face = (front_face_pre == front_face_label).sum() / len(front_face_label) * 100

        total_loss += loss.item() * data.size(0)
        if total_loss < pre_loss:

            pre_loss = total_loss

        print("{0: >15} | {1: >15.10f} | {2:>15.2f} | {3: >15.2f} | {4: >15.2f} | {5:>15.2f} | {6: >15.2f}".format(
            i + 1, total_loss, acc_hair, acc_gender, acc_earring, acc_smile, acc_front_face))

    torch.save(model.state_dict(), path)