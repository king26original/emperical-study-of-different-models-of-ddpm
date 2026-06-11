import torchvision
import torchvision.transforms as transforms
import torch

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],#mean
        [0.5, 0.5, 0.5]#std
    )
])

"""assuming the user already has downloaded the dataset in nested directory format"""
trainset=torchvision.datasets.ImageFolder(
    root='/kaggle/input/datasets/oxcdcd/cifar10/cifar10/train',
    transform=transform
)

trainloader=torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
