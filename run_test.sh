# python test_cvusa.py \
# --name='usa_vgg_noshare_warm5_4PCBv_lr0.05' \
# --test_dir='/home/wangtyu/datasets/CVUSA/val' \
# --gpu_ids='1'
#
# python test.py \
# --name='three_view_long_share_d0.75_256_s1_google_lr0.005_bn_v18_210ep_r' \
# --test_dir='/home/wangtyu/datasets/University-Release/test' \
# --batchsize=128 \
# --gpu_ids='3'

# python test_cvusa.py \
# --name='usa_res50_share_warm5_lr0.02_spade_210ep_weather_h112_110' \
# --test_dir='/home/wangtyu/datasets/CVUSA/val_pt' \
# --iaa \
# --gpu_ids='3'
#

#python test_iaa.py \
#--name='three_view_long_share_d0.5_256_s1_google_lr0.005_vgg_v24.11_210ep_weather1' \
#--test_dir='/home/wangtingyu/datasets/University-Release/test' \
#--batchsize=128 \
#--gpu_ids='3' \
#--iaa

# python test_iaa.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.005_ibn_v24.8_210ep_weather1' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --batchsize=128 \
# --gpu_ids='0' \
# --iaa

python test_iaa.py \
--name='three_view_long_share_d0.5_256_s1_google_lr0.005_spade_v24.11_210ep_weather_0110000_r2' \
--test_dir='/home/wangtingyu/datasets/University-Release/test' \
--batchsize=128 \
--gpu_ids='1' \
--iaa

#python test_cvusa.py \
#--name='usa_res50_noshare_warm5_lr0.01' \
#--test_dir='/home/wangtyu/datasets/CVUSA/val' \
#--gpu_ids='3'
