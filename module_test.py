#test customdata_dtyle
import torch
from image_folder import customData, customData_one, customData_style, ImageFolder_iaa, ImageFolder_iaa_multi_weather, ImageFolder_iaa_selectID
from torchvision import datasets, models, transforms
import imgaug.augmenters as iaa
from PIL import Image
import numpy as np
import imageio
import random
import os
from collections import defaultdict

# trained_model = torch.load('/home/wangtyu/university-ibn/model/three_view_long_share_d0.75_256_s1_google_lr0.005_ada-ibn_v21.5_210ep_weather/net_209.pth')
# print(trained_model.keys())
# print(trained_model['model_1.model.bn1.running_var'])

dir = '/home/wangtyu/datasets/University-Release/test/query_drone'

data_transforms = transforms.Compose([
        # transforms.Resize((256, 256), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

seq = iaa.Sequential(
        [
            iaa.BlendAlpha(0.5, foreground=iaa.Add(100), background=iaa.Multiply(0.2), seed=31),
            iaa.MultiplyAndAddToBrightness(mul=0.2, add=-15, seed=1991),  # qianbao
            iaa.Resize({"height":256, "width":256}),
            iaa.Pad(px=10, pad_mode="edge", keep_size=False),
            iaa.CropToFixedSize(width=256, height=256),
            iaa.Fliplr(0.5),
        ]
)

iaa_weather_list = [
    None,
    iaa.Sequential([
        iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
                       alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
                       density_multiplier=0.5, seed=35)

    ]),
    iaa.Sequential([
        iaa.Rain(drop_size=(0.005, 0.015), speed=(0.04, 0.06), seed=38),
        iaa.Rain(drop_size=(0.005, 0.015), speed=(0.04, 0.06), seed=35)
    ]),
    iaa.Sequential([
        # iaa.Snowflakes(flake_size=(0.2, 0.7), speed=(0.007, 0.03), seed=38),
        # iaa.Snowflakes(flake_size=(0.2, 0.7), speed=(0.007, 0.03), seed=35)
        iaa.SnowflakesLayer(density=0.075, density_uniformity=0.5, flake_size=(0.6, 0.8), flake_size_uniformity=0.5,
                            angle=(-30, 30), speed=(0.007, 0.03), blur_sigma_fraction=(0.0001, 0.001), seed=38),
        iaa.SnowflakesLayer(density=0.075, density_uniformity=0.5, flake_size=(0.6, 0.8), flake_size_uniformity=0.5,
                            angle=(-30, 30), speed=(0.007, 0.03), blur_sigma_fraction=(0.0001, 0.001), seed=35)
    ]),
    iaa.Sequential([
        iaa.MultiplyAndAddToBrightness(mul=0.2, add=-15, seed=1991)  # qianbao
    ]),
    iaa.Sequential([
        iaa.MultiplyAndAddToBrightness(mul=1.6, add=(0, 30), seed=1992)  # guobao
    ]),
    iaa.Sequential([
        iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
                       alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
                       density_multiplier=0.5, seed=35),
        iaa.Rain(drop_size=(0.005, 0.015), speed=(0.04, 0.06), seed=35)

    ]),
    iaa.Sequential([
        iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
                       alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
                       density_multiplier=0.5, seed=35),
        iaa.SnowflakesLayer(density=0.075, density_uniformity=0.5, flake_size=(0.6, 0.8), flake_size_uniformity=0.5,
                            angle=(-30, 30), speed=(0.007, 0.03), blur_sigma_fraction=(0.0001, 0.001), seed=38)
    ]),
    iaa.Sequential([
        iaa.SnowflakesLayer(density=0.075, density_uniformity=0.5, flake_size=(0.6, 0.8), flake_size_uniformity=0.5,
                            angle=(-30, 30), speed=(0.007, 0.03), blur_sigma_fraction=(0.0001, 0.001), seed=38),
        iaa.Rain(drop_size=(0.005, 0.015), speed=(0.04, 0.06), seed=35)
    ])
]
# image_dataset={}
# data_dir = '/home/wangtyu/datasets/University-Release/train'
# image_dataset['drone'] = datasets.ImageFolder(os.path.join(data_dir, 'drone'), data_transforms)
# image_dataset['satellite'] = datasets.ImageFolder(os.path.join(data_dir, 'satellite'), data_transforms)
# dataloaders = {x: torch.utils.data.DataLoader(image_dataset[x], batch_size=8,
#                                                 shuffle=True, num_workers=0, pin_memory=False)
#                 for x in ['satellite', 'drone']}

image_datasets = ImageFolder_iaa_selectID(dir, transform=data_transforms, iaa_transform=seq)
# image_datasets = ImageFolder_iaa_multi_weather(dir, transform=data_transforms, iaa_transform=seq, iaa_weather_list=iaa_weather_list, batchsize=8, shuffle=True)

# data = next(iter(image_datasets))

dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=8,
                                             shuffle=False, num_workers=0, pin_memory=False)

# for i in range(5):
#     s = set()
#     for data,data2 in zip(dataloaders['satellite'], dataloader):
#         _, label = data
#         _, label2 = data2
#         for l in label2:
#             s.add(int(l))
#     print(len(s))

for epoch in range(2):
    s = set()
    for i, data in enumerate(dataloader):
        imput, label = data
    #     for l in label:
    #         s.add(int(l))
    # print(len(s))
# model = model = three_view_net(701, droprate = 0.75, stride = 1, pool = 'avg', share_weight = True, norm = 'ada-ibn')
# # load params of trained ibn
# trained_dict = torch.load('/home/wangtingyu/LPN-ibn/model/three_view_long_share_d0.75_256_s1_google_lr0.01_ibn/net_199.pth')
# model_dict = model.state_dict()
# trained_dict = {k: v for k, v in trained_dict.items() if k in model_dict}
# model_dict.update(trained_dict)
# model.load_state_dict(model_dict)
#
# # fix the params of main model
# for subnet in [model.model_1, model.model_2, model.model_3, model.classifier]:
#     for p in subnet.parameters():
#         p.requires_grad = False

# check the params
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)