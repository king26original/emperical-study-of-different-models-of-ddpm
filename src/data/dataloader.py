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

trainset=torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader=torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
