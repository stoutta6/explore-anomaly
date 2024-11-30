# adapted from https://github.com/talreiss/Mean-Shifted-Anomaly-Detection/blob/main/utils.py
# @article{reiss2021mean,
#   title={Mean-Shifted Contrastive Loss for Anomaly Detection},
#   author={Reiss, Tal and Hoshen, Yedid},
#   journal={arXiv preprint arXiv:2106.03844},
#   year={2021}
# }

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import torchvision.models as models
import torch.nn.functional as F
from PIL import ImageFilter
from skimage import io
import random
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
# from torchvision.datasets.folder import IMG_EXTENSIONS

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_resnet18 = transforms.Compose([
    transforms.Resize(224, interpolation=BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_xview3_basic = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224, interpolation=BICUBIC),
    transforms.CenterCrop(224),
    transforms.Normalize([67.69, 86.55, 105.37], [6.528, 5.427, 8.328])
])

class Transform:
    def __init__(self):
        self.moco_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, x):
        x_1 = self.moco_transform(x)
        x_2 = self.moco_transform(x)
        return x_1, x_2
    
class Transform_xview3:
    def __init__(self, transform_set_num=4):
        if transform_set_num == 1:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224, scale=(0.765, 1.)),
                transforms.Normalize([67.69, 86.55, 105.37], [6.528, 5.427, 8.328])])
        elif transform_set_num == 2:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224, scale=(0.765, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize([67.69, 86.55, 105.37], [6.528, 5.427, 8.328])])
        elif transform_set_num == 3:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224, scale=(0.765, 1.)),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3))], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize([67.69, 86.55, 105.37], [6.528, 5.427, 8.328])])
        elif transform_set_num == 4:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224, scale=(0.765, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3))], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize([67.69, 86.55, 105.37], [6.528, 5.427, 8.328])])
        elif transform_set_num == 5:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224, scale=(0.765, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomErasing(p=1.0, scale=(0.05, 0.15)),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3))], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize([67.69, 86.55, 105.37], [6.528, 5.427, 8.328])])

    def __call__(self, x):
        x_1 = self.transform(x)
        x_2 = self.transform(x)
        return x_1, x_2

def xview3_loader(path, clone_channel=False):
    img = io.imread(path) / 256 # normalize to 8-bit range
    if clone_channel:
        img = np.moveaxis(np.tile(img, [3,1,1]), 0, 2) # clone single channel to 3 channels
    return img

class Model(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        if backbone.lower() == 'resnet152':
            self.backbone = models.resnet152(pretrained=True)
        elif backbone.lower() == 'resnet50':
            self.backbone = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        elif backbone.lower() == 'resnet18':
            self.backbone = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1') #pretrained=True)
        elif backbone.lower() == 'convnext_tiny':
            self.backbone = models.convnext_tiny(weights='ConvNeXt_Tiny_Weights.IMAGENET1K_V1')
        elif backbone.lower() == 'vit_b_16':
            self.backbone = models.vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_V1')
        
        
        self.backbone.fc = torch.nn.Identity()
        freeze_parameters(self.backbone, backbone, train_fc=False)

    def forward(self, x):
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n

def freeze_parameters(model, backbone, train_fc=False):
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False
    if backbone.lower() == 'resnet152':
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.bn1.parameters():
            p.requires_grad = False
        for p in model.layer1.parameters():
            p.requires_grad = False
        for p in model.layer2.parameters():
            p.requires_grad = False



def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def get_loaders(dataset, training_data_root, testing_data_root, label_class, batch_size, backbone, num_workers=0):
    if dataset == "cifar10":
        ds = torchvision.datasets.CIFAR10
        transform = transform_color if backbone == 152 else transform_resnet18
        coarse = {}
        trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
        testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        trainset_1 = ds(root='data', train=True, download=True, transform=Transform(), **coarse)
        idx = np.array(trainset.targets) == label_class
        testset.targets = [int(t != label_class) for t in testset.targets]
        trainset.data = trainset.data[idx]
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        trainset_1.data = trainset_1.data[idx]
        trainset_1.targets = [trainset_1.targets[i] for i, flag in enumerate(idx, 0) if flag]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                                  drop_last=False)
        return train_loader, test_loader, torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
                                                                      shuffle=True, num_workers=num_workers, drop_last=False)
    elif dataset == "xview3":
        transform = transform_xview3_basic
        trainset = torchvision.datasets.ImageFolder(root=training_data_root, transform=transform, loader=xview3_loader)
        testset = torchvision.datasets.ImageFolder(root=testing_data_root, transform=transform, loader=xview3_loader)
        trainset_1 = torchvision.datasets.ImageFolder(root=training_data_root, transform=Transform_xview3(), loader=xview3_loader)
        idx = np.array(trainset.targets) == label_class
        idx_remove = np.array(trainset.targets) != label_class        
        idx = idx[np.invert(idx_remove)] # remove non-label class from training
        testset.targets = [int(t != label_class) for t in testset.targets]
        for ii in np.flip(np.where(idx_remove)[0]):
            trainset.imgs.remove(trainset.imgs[ii])
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        for ii in np.flip(np.where(idx_remove)[0]):
            trainset_1.imgs.remove(trainset_1.imgs[ii])
        trainset_1.targets = [trainset_1.targets[i] for i, flag in enumerate(idx, 0) if flag]
        
        # shuffle dataset once for the train and test loaders
        # shuffle dataset at each epoch (shuffle=True) for the train with augmentations loader
        ShuffledTrainset = torch.utils.data.Subset(trainset, torch.randperm(len(trainset)))
        train_loader = torch.utils.data.DataLoader(ShuffledTrainset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                                   drop_last=False)
        ShuffledTestset = torch.utils.data.Subset(testset, torch.randperm(len(testset)))
        test_loader = torch.utils.data.DataLoader(ShuffledTestset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                                  drop_last=False)
        train_loader_aug = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
                                                                      shuffle=True, num_workers=num_workers, drop_last=False)
        return train_loader, test_loader, train_loader_aug
    else:
        print('Unsupported Dataset')
        exit()