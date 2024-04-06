# satellite images have been transformed

import os
from shutil import copyfile
import numpy as np
from scipy.misc import imread, imsave



######################### prepare cvusa dataset ###########################

download_path = '/home/wangtyu/datasets/CVUSA/'
# train_split = download_path + 'splits/train-19zl.csv'
# train_save_path = download_path + 'train_pt/'  # polar transform satellite images
#
# if not os.path.isdir(train_save_path):
#     os.mkdir(train_save_path)
#     # os.mkdir(train_save_path + 'street')
#     os.mkdir(train_save_path + 'satellite')
# num = 0
# with open(train_split) as fp:
#     line = fp.readline()
#     while line:
#         filename = line.split(',')
#         #print(filename[0])
#         src_path = download_path + 'polarmap/' + filename[0][8:-4] + '.png'
#         dst_path = train_save_path + 'satellite/' + os.path.basename(filename[0][:-4])
#         if not os.path.isdir(dst_path):
#             os.mkdir(dst_path)
#         copyfile(src_path, dst_path + '/' + os.path.basename(filename[0]))
#
#         # apply_aerial_polar_transform(src_path, dst_path, os.path.basename(filename[0]))
#
#         # src_path = download_path + '/' + filename[1]
#         # dst_path = train_save_path + '/street/' + os.path.basename(filename[1][:-4])
#         # if not os.path.isdir(dst_path):
#         #     os.mkdir(dst_path)
#         # copyfile(src_path, dst_path + '/' + os.path.basename(filename[1]))
#         num += 1
#         line = fp.readline()
# print('file_num:', num)

val_split = download_path + 'splits/val-19zl.csv'
val_save_path = download_path + 'val_pt/'

if not os.path.isdir(val_save_path):
    os.mkdir(val_save_path)
    # os.mkdir(val_save_path + 'street')
    os.mkdir(val_save_path + 'satellite')
num = 0
with open(val_split) as fp:
    line = fp.readline()
    while line:
        filename = line.split(',')
        #print(filename[0])
        src_path = download_path + 'polarmap/' + filename[0][8:-4] + '.png'
        dst_path = val_save_path + 'satellite/' + os.path.basename(filename[0][:-4])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + os.path.basename(filename[0]))
        # apply_aerial_polar_transform(src_path, dst_path, os.path.basename(filename[0]))

        # src_path = download_path + '/' + filename[1]
        # dst_path = val_save_path + '/street/' + os.path.basename(filename[1][:-4])
        # if not os.path.isdir(dst_path):
        #     os.mkdir(dst_path)
        # copyfile(src_path, dst_path + '/' + os.path.basename(filename[1]))
        num = num + 1
        line = fp.readline()

print('file_num:', num)

