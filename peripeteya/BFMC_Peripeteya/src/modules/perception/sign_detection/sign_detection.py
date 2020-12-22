import math
import os
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16 * 58 * 58, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 24)
        self.fc4 = nn.Linear(24, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        sizes = x.size()[1:]
        num_features = 1
        for s in sizes:
            num_features *= s
        return num_features


ZEROS = np.zeros(10)

LABELS = {
    "none": 0,
    "enter_highway": 1,
    "exit_highway": 2,
    "no_entry": 3,
    "one_way": 4,
    "parking": 5,
    "pedestrians": 6,
    "priority": 7,
    "roundabout": 8,
    "stop": 9,
}


SIGNS = {v: k for k, v in LABELS.items()}


def get_data_set(path):
    images = []
    labels = []
    sub_dir = [*os.walk(path)][0][1]
    for dir in sub_dir:
        for img in glob.glob(path+dir + "/*.png"):
            img_read = cv2.imread(img)
            img_read = cv2.resize(img_read, (240, 240))
            gray = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
            _, single_channel = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            images.append(single_channel)
            copy_zeros = deepcopy(ZEROS)
            copy_zeros[LABELS[dir]] = 1.
            label = np.asarray(copy_zeros)
            labels.append(label)
        for img in glob.glob(path+dir + "/*.jpg"):
            img_read = cv2.imread(img)
            img_read = cv2.resize(img_read, (240, 240))
            gray = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
            _, single_channel = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            images.append(single_channel)
            copy_zeros = deepcopy(ZEROS)
            copy_zeros[LABELS[dir]] = 1.
            label = np.asarray(copy_zeros)
            labels.append(label)
    images = np.asarray(images, dtype=float)
    x = torch.from_numpy(images)
    x = x.unsqueeze(1)
    y = torch.from_numpy(np.asarray(labels))
    return x, y, labels


def get_images_labels(path):
    images = []
    labels = []
    sub_dir = [*os.walk(path)][0][1]
    for dir in sub_dir:
        for img in glob.glob(path + dir + "/*.png"):
            img_read = cv2.imread(img)
            images.append(img_read)
            copy_zeros = deepcopy(ZEROS)
            copy_zeros[LABELS[dir]] = 1.
            label = np.asarray(copy_zeros)
            labels.append(label)
        for img in glob.glob(path + dir + "/*.jpg"):
            img_read = cv2.imread(img)
            images.append(img_read)
            copy_zeros = deepcopy(ZEROS)
            copy_zeros[LABELS[dir]] = 1.
            label = np.asarray(copy_zeros)
            labels.append(label)
    return images, labels


def train_model(net, x, y, iterations, optimizer, criterion=nn.MSELoss()):
    costs = []
    for i in range(iterations):
        # forward
        y_pred = net(x)
        # backward + loss
        optimizer.zero_grad()
        loss = criterion(y_pred, y)
        if (i + 1) % 100 == 0 and i != 0:
            print(f"Cost after iteration {i}: {loss.item()}")
        costs.append(loss.item())
        loss.backward()
        optimizer.step()
    return costs


def main():
    net = load_model()
    # retrieve training set
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
    iterations = 1000
    x, y, labels = get_data_set("data_set/")
    x = x.cuda()
    y = y.cuda()
    for i in range(10):
        print(f"Current epoch: {i+1}")
        # train
        costs = train_model(net, x, y, iterations, optimizer)
        # see results of the training
        plt.plot(costs)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.show()

    # save model
    torch.save(net, "model.pt")


def load_model():
    net = Net()
    # load the model if it was previously saved so we don't need to train it again
    try:
        net = torch.load("model.pt")
        net.eval()
    except Exception:
        pass
    net = net.double()
    device = torch.device("cuda:0")
    net = net.to(device=device)
    return net


def predict(image: np.ndarray, cnn: Net):
    img_read = cv2.resize(image, (240, 240))
    gray = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
    _, single_channel = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # need a (240, 120) image of a binary threshold lane view
    images = np.asarray([single_channel], dtype=float)
    x = torch.from_numpy(images)
    x = x.unsqueeze(1)
    x = x.cuda()
    prediction = cnn(x).cpu().detach().numpy()[0]
    return prediction


def run_simulation():
    net = load_model()
    images, labels = get_images_labels("data_set/")
    labels = list(labels)
    for i in range(len(images)):
        image = images[i]
        prediction = list(predict(image, net))
        image = cv2.resize(image, (240, 240))
        max_confidence = max(prediction)
        max_conf_index = prediction.index(max_confidence)
        predicted_sign = SIGNS[max_conf_index]
        print(predicted_sign, max_confidence)
        cv2.imshow("image", image)
        cv2.waitKey(0)
