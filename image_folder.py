import torch.utils.data as Data
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from os.path import join as ospj
from PIL import Image
import os
import numpy as np
from collections import defaultdict
import random

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images
#getting one image of a folder.
def make_dataset_one(dir, class_to_idx, extensions, reverse=False):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            index = 0
            for fname in sorted(fnames, reverse=reverse):
                index += 1
                if has_file_allowed_extension(fname, extensions) and index == 36:
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    break

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class customData(Data.Dataset):
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader, rotate = 0, pad = 0):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform 
        self.loader = loader
        self.rotate = rotate
        self.pad = pad
    def __getitem__(self, index):
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, target = self.imgs[index]
            img = self.loader(path)
            img = transforms.functional.rotate(img,self.rotate)
            if self.pad > 0:
                img = transforms.functional.resize(img,(256,256),interpolation=3)
                img = transforms.functional.pad(img,(self.pad,0,0,0))
                img = transforms.functional.five_crop(img,(256,256))[0]
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

    def __len__(self):
        return len(self.imgs)

class customData_one(Data.Dataset):
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader, rotate = 0, pad = 0, reverse=False):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset_one(root, class_to_idx, IMG_EXTENSIONS, reverse)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform 
        self.loader = loader
        self.rotate = rotate
        self.pad = pad
    def __getitem__(self, index):
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, target = self.imgs[index]
            img = self.loader(path)
            img = transforms.functional.rotate(img,self.rotate)
            if self.pad > 0:
                img = transforms.functional.resize(img,(256,256),interpolation=3)
                img = transforms.functional.pad(img,(self.pad,0,0,0))
                img = transforms.functional.five_crop(img,(256,256))[0]
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

    def __len__(self):
        return len(self.imgs)

def make_dataset_style(dir, class_to_idx, extensions, style='all'):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    if style == 'all':
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
                    else:
                        fstyle = fname.split('_')[2].split('.')[0]
                        if fstyle == style:
                            path = os.path.join(root, fname)
                            item = (path, class_to_idx[target])
                            images.append(item)

    return images
class customData_style(Data.Dataset):
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader, rotate = 0, pad = 0, style = 'all'):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset_style(root, class_to_idx, IMG_EXTENSIONS, style=style)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform 
        self.loader = loader
        self.rotate = rotate
        self.pad = pad
    def __getitem__(self, index):
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, target = self.imgs[index]
            img = self.loader(path)
            img = transforms.functional.rotate(img,self.rotate)
            if self.pad > 0:
                img = transforms.functional.resize(img,(256,256),interpolation=3)
                img = transforms.functional.pad(img,(self.pad,0,0,0))
                img = transforms.functional.five_crop(img,(256,256))[0]
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

    def __len__(self):
        return len(self.imgs)


class ImageFolder_iaa(Data.Dataset):
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader, iaa_transform = None):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform 
        self.loader = loader
        self.iaa_trans = iaa_transform
    def __getitem__(self, index):
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, target = self.imgs[index]
            img = self.loader(path)
            if self.iaa_trans is not None:
                img = np.array(img)
                img = self.iaa_trans(image = img)
                # img = Image.fromarray(img)
                # img_aug.save('/home/wangtingyu/%s'%path.split('/')[-1])
            # else:
            #     img = np.array(img)
            #     # img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

    def __len__(self):
        return len(self.imgs)

def make_dataset_selectID(dir, class_to_idx, extensions):
    images = defaultdict(list)
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images[class_to_idx[target]].append(item)
    return images

class ImageFolder_iaa_selectID(Data.Dataset):
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader, iaa_transform = None, norm='bn'):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset_selectID(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs.keys()) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.iaa_trans = iaa_transform
        self.norm = norm
    def __getitem__(self, index):
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, target = random.choice(self.imgs[index])
            img = self.loader(path)
            if self.iaa_trans is not None:
                img = np.array(img)
                img = self.iaa_trans(image = img)
                # img_aug = Image.fromarray(img)
                # img_aug.save('/home/wangtyu/test_img/%s'%path.split('/')[-1])
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            if self.norm == 'ada-ibn' or self.norm == 'spade':
                return img, target, 1
            else:
                return img, target

    def __len__(self):
        return len(self.imgs)

class ImageFolder_iaa_multi_weather(Data.Dataset):
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader, iaa_transform = None, iaa_weather_list=[], batchsize=8, shuffle=False, norm='bn', select=False):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        if select:
            imgs = make_dataset_selectID(root, class_to_idx, IMG_EXTENSIONS)
        else:
            imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.iaa_trans = iaa_transform
        self.iaa_weather_list = iaa_weather_list
        self.batch = batchsize
        self.img_num = 0
        self.shuffle = shuffle
        self.norm = norm
        self.select = select
        # random.seed(1)
    def __getitem__(self, index):
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            if self.select:
                path, target = random.choice(self.imgs[index])
            else:
                path, target = self.imgs[index]
            img = self.loader(path)
            # if self.iaa_trans is not None:
            #     img = np.array(img)
            #     img = self.iaa_trans(image = img)
            if self.iaa_weather_list:
                img = np.array(img)
                if self.shuffle:
                    idx = random.choice(range(len(self.iaa_weather_list)))
                    if idx == 0:
                        img = img
                    else:
                        img = self.iaa_weather_list[idx](image=img)
                        # img_aug = Image.fromarray(img)
                        # img_aug.save('/home/wangtyu/test_img/%s'%path.split('/')[-1])

                else:
                    idx = self.img_num // self.batch % (len(self.iaa_weather_list)+1)
                    if idx == 0:
                        img = img
                        # img_aug = Image.fromarray(img)
                        # img_aug.save('/Users/wongtyu/Downloads/University-Release/train/%s'%path.split('/')[-1])
                    else:
                        img = self.iaa_weather_list[idx-1](image = img)
                        # img_aug = Image.fromarray(img)
                        # img_aug.save('/Users/wongtyu/Downloads/University-Release/train/%s'%path.split('/')[-1])
                    self.img_num += 1
                    if self.img_num == len(self):
                        self.img_num = 0
            if self.iaa_trans is not None:
                img = self.iaa_trans(image = img)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            if self.norm == 'ada-ibn' or self.norm == 'spade':
                return img, target, idx+1
            else:
                return img, target

    def __len__(self):
        return len(self.imgs)