import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from src.modules.perception.lane_detection.test_images.image_processing import generate_images


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16 * 58 * 28, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 24)
        self.fc4 = nn.Linear(24, 3)

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


def load_model():
    net = Net()
    # load the model if it was previously saved so we don't need to train it again
    try:
        net = torch.load("model.pt")
        net.eval()
    except Exception:
        pass
    # net = net.double()
    device = torch.device("cpu")
    net.double()
    return net.to(device)


def predict(image: np.ndarray, cnn: Net):
    image = cv2.resize(image, (240, 240))
    image = image[120:]
    # gray = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
    # _, single_channel = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # single_channel = single_channel[120:]
    # need a (240, 120) image of a binary threshold lane view
    images = np.asarray([image], dtype=np.float)
    print(type(images[0][0][0]))
    x = torch.from_numpy(images).double()
    x = x.unsqueeze(1)
    x = x.to(torch.device("cpu"))
    print(x.size())
    prediction = cnn(x).cpu().detach().numpy()[0]
    return prediction


def test():
    model = load_model()
    image = cv2.imread("image_c0_b40.jpg")
    result = generate_images(image)
    print(predict(result["warped_thresh"], model))
