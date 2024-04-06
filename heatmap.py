import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from model import three_view_net
from utils import load_network
import yaml
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import imgaug.augmenters as iaa
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--data_dir',default='/home/wangtyu/datasets/University-Release/test',type=str, help='./test_data')
parser.add_argument('--name', default='three_view_long_share_d0.75_256_s1_google_lr0.005_bn_v21.9_210ep_weather', type=str, help='save model path')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')

opt = parser.parse_args()

config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.fp16 = config['fp16']
opt.stride = config['stride']
opt.block = config['block']
opt.views = config['views']
opt.norm = config['norm']
opt.adain = config['adain']
if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']
if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 751


def heatmap2d(img, arr):
    # fig = plt.figure()
    # ax0 = fig.add_subplot(121, title="Image")
    # ax1 = fig.add_subplot(122, title="Heatmap")
    # fig, ax = plt.subplots(）
    # ax[0].imshow(Image.open(img))
    # plt.figure()
    heatmap = plt.imshow(arr, cmap='viridis')
    plt.axis('off')
    # fig.colorbar(heatmap, fraction=0.046, pad=0.04)
    #plt.show()
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    plt.savefig('bn_heatmap_normal', bbox_inches='tight', pad_inches=0)

# data_transforms = transforms.Compose([
#         transforms.Resize((opt.h, opt.w), interpolation=3),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

def assign_adain_params(adain_params_w, adain_params_b, model, dim=32, init_w=None, init_b=None):
    # assign the adain_params to the AdaIN layers in model
    # dim = self.output_dim
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params_b[:, :dim].contiguous()
            std = adain_params_w[:, :dim].contiguous()
            m.bias = mean.view(-1)
            m.weight = std.view(-1)
            if adain_params_w.size(1) > dim:  # Pop the parameters
                adain_params_b = adain_params_b[:, dim:]
                adain_params_w = adain_params_w[:, dim:]


iaa_transform = iaa.Sequential(
        [
            # iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
            #                alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
            #                density_multiplier=0.5, seed=35),

            # iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38),
            # iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            # iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=73),
            # iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=93),
            # iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95),

            # iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=38),
            # iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            # iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
            # iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=94),
            # iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=96),

            # iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
            #                alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
            #                density_multiplier=0.5, seed=35),
            # iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),
            # iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36),

            # iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
            #                alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
            #                density_multiplier=0.5, seed=35),
            # iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
            # iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36),

            # iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            # iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            # iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=92),
            # iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91),
            # iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),

            # iaa.BlendAlpha(0.5, foreground=iaa.Add(100), background=iaa.Multiply(0.2), seed=31),
            # iaa.MultiplyAndAddToBrightness(mul=0.2, add=(-30, -15), seed=1991),  # qianbao

            # iaa.MultiplyAndAddToBrightness(mul=1.6, add=(0, 30), seed=1992),  # guobao

            # iaa.MotionBlur(15, seed=17),
            iaa.Resize({"height": opt.h, "width": opt.w}, interpolation=3),
        ]
    )
data_transforms_iaa = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# image_datasets = {x: datasets.ImageFolder( os.path.join(opt.data_dir,x) ,data_transforms) for x in ['satellite']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
#                                              shuffle=False, num_workers=1) for x in ['satellite']}

# imgpath = image_datasets['satellite'].imgs
# print(imgpath)
# imgname = 'gallery_drone/0721/image-28.jpeg'
# imgname = 'query_satellite/0721/0721.jpg'
imgname = 'query_drone/0561/image-24.jpeg'
imgpath = os.path.join(opt.data_dir,imgname)
img = Image.open(imgpath)
# img = data_transforms(img)
img = np.array(img)
img = iaa_transform(image=img)
img = data_transforms_iaa(img)
img = torch.unsqueeze(img,0)
print(img.shape)
model, _, epoch = load_network(opt.name, opt)
# print(model)
model = model.eval().cuda()

# data = next(iter(dataloaders['satellite']))
# img, label = data
with torch.no_grad():
    # dw1, db1, dw2, db2, dw3, db3, dout_w = model.pt_model(img.cuda())
    # assign_adain_params(dw1[0] + model._w1[0], db1[0] + model._b1[0], model.model_3.model.layer1[0], 32)
    # assign_adain_params(dw1[2] + model._w1[2], db1[2] + model._b1[2], model.model_3.model.layer1[2], 32)
    x = model.model_3.model.conv1(img.cuda())
    x = model.model_3.model.bn1(x)
    x = model.model_3.model.relu(x)
    x = model.model_3.model.maxpool(x)
    x = model.model_3.model.layer1(x)
    x = model.model_3.model.layer2(x)
    x = model.model_3.model.layer3(x)
    output = model.model_3.model.layer4(x)
print(output.shape)
heatmap = output.squeeze().sum(dim=0).cpu().numpy()

#test_array = np.arange(100 * 100).reshape(100, 100)
# Result is saved tas `heatmap.png`
heatmap = cv2.resize(heatmap, (512, 512))
print(heatmap.shape)
heatmap2d(imgpath,heatmap)