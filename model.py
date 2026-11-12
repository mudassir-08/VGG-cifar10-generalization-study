import torch
import torch.nn as nn

class VGGClassifier(nn.Module):
  def __init__(self):
    super().__init__()

    self.block1= nn.Sequential(
      nn.Conv2d(3, 64, 3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),


      nn.Conv2d(64, 64, 3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),

      nn.MaxPool2d(2)
    )

    self.block2= nn.Sequential(
      nn.Conv2d(64, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),

      nn.Conv2d(128, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),

      nn.MaxPool2d(2)
    )

    self.block3=nn.Sequential(
      nn.Conv2d(128, 256, 3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),

      nn.Conv2d(256, 256, 3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),

      nn.Conv2d(256, 256, 3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),

    )

    self.block4=nn.Sequential(
      nn.Conv2d(256, 512, 3, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),

      nn.Conv2d(512, 512, 3, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),

      nn.Conv2d(512, 512, 3, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),

      nn.MaxPool2d(2)
    )

    self.block5=nn.Sequential(
      nn.Conv2d(512,512, 3, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),

      nn.Conv2d(512, 512, 3, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),

      nn.Conv2d(512, 512, 3, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),

      nn.MaxPool2d(2)
    )

    self.classifier=nn.Sequential(
      nn.AdaptiveAvgPool2d(output_size=(1,1)),
      nn.Flatten(),

      nn.Linear(512, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(),
      nn.Dropout(0.2),

      nn.Linear(256, 10)
    )

  def forward(self, x):
      x= self.block1(x)
      x= self.block2(x)
      x= self.block3(x)
      x= self.block4(x)
      x= self.block5(x)

      x= self.classifier(x)

      return x

