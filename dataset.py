from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def get_dataloaders(batch_size):
  train_transforms= transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
      mean=[0.4914, 0.4822, 0.4465],
      std=[0.2470, 0.2435, 0.2616]
    )
  ])

  val_transforms= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
      mean=[0.4914, 0.4822, 0.4465],
      std=[0.2470, 0.2435, 0.2616]
    )
  ])

  train_dataset= datasets.CIFER10(
    root="./data",
    train=True,
    download=True,
    transform=train_transforms
  )

  val_dataset= datasets.CIFER10(
    root="./data",
    train=False,
    download=True,
    transform=val_transforms
  )

  train_loader= DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
  )

  val_loader= DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
  )

  return train_loader, val_loader