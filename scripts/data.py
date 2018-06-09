from torchvision import transforms, datasets

def get_data(image_size=64):
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return datasets.CIFAR10(root='data', transform=transform, download=True)