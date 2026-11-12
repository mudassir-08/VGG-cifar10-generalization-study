
import torch

class config():
  BATCH_SIZE =64
  epochs= 10
  lr_rate=3e-4
  weight_decay=1e-5
  num_classes=10
  image_size=32
  grad_clip=1.0
  Device = torch.device("cude" if torch.cude.is_availabele else "cpu")
  