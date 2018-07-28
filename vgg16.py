import torch.nn as nn
import torch.nn.functional as F


class Vgg(nn.Module):
    def __init__(self):
        super(Vgg, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool(F.relu(self.conv6(F.relu(self.conv6(F.relu(self.conv5(x)))))))
        x = self.pool(F.relu(self.conv8(F.relu(self.conv8(F.relu(self.conv7(x)))))))
        x = self.pool(F.relu(self.conv8(F.relu(self.conv8(F.relu(self.conv8(x)))))))

        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class VggBN(nn.Module):
    def __init__(self):
        super(VggBN, self).__init__()
        self.conv = nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            # Stage 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            # Stage 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            # Stage 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            # Stage 5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            # FC
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


class VggDropout(nn.Module):
    def __init__(self):
        super(VggDropout, self).__init__()
        self.conv = nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Stage 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Stage 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Stage 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Stage 5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            # FC
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


class VggDropBN(nn.Module):
    def __init__(self):
        super(VggDropBN, self).__init__()
        self.conv = nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            # Stage 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            # Stage 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            # Stage 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            # Stage 5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(p=0.4),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            # FC
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x