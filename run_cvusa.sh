# train bn baseline
#python train_cvusa.py \
#--name='usa_res50_share_warm5_lr0.02_bn_120ep_h112' \
#--data_dir='/home/wangtyu/datasets/CVUSA/train_pt' \
#--warm_epoch=5 \
#--batchsize=16 \
#--h=112 \
#--w=616 \
#--droprate=0.5 \
#--share \
#--stride=1 \
#--lr=0.02 \
#--norm='bn' \
#--adain='a' \
#--iaa \
#--gpu_ids='0'
#
#python test_cvusa.py \
#--name='usa_res50_share_warm5_lr0.02_bn_120ep_h112' \
#--test_dir='/home/wangtyu/datasets/CVUSA/val_pt' \
#--iaa \
#--gpu_ids='0'

# train multi-weather cvusa bn
#python train_cvusa.py \
#--name='usa_res50_share_warm5_lr0.02_bn_210ep_weather_h112' \
#--data_dir='/home/wangtyu/datasets/CVUSA/train_pt' \
#--warm_epoch=5 \
#--batchsize=16 \
#--h=112 \
#--w=616 \
#--droprate=0.5 \
#--share \
#--stride=1 \
#--lr=0.02 \
#--norm='ada-ibn' \
#--adain='a' \
#--iaa \
#--multi_weather \
#--gpu_ids='3'
#
#python test_cvusa.py \
#--name='usa_res50_share_warm5_lr0.02_bn_210ep_weather_h112' \
#--test_dir='/home/wangtyu/datasets/CVUSA/val_pt' \
#--iaa \
#--gpu_ids='3'

# train multi-weather cvusa ibn
#python train_cvusa.py \
#--name='usa_res50_share_warm5_lr0.02_bn_210ep_weather_h112' \
#--data_dir='/home/wangtyu/datasets/CVUSA/train_pt' \
#--warm_epoch=5 \
#--batchsize=16 \
#--h=112 \
#--w=616 \
#--droprate=0.5 \
#--share \
#--stride=1 \
#--lr=0.02 \
#--norm='bn' \
#--adain='a' \
#--iaa \
#--multi_weather \
#--gpu_ids='3'
#
#python test_cvusa.py \
#--name='usa_res50_share_warm5_lr0.02_bn_210ep_weather_h112' \
#--test_dir='/home/wangtyu/datasets/CVUSA/val_pt' \
#--iaa \
#--gpu_ids='3'


# spade
python train_cvusa.py \
--name='usa_res50_share_warm5_lr0.02_spade_210ep_weather_h112_110' \
--data_dir='/home/wangtyu/datasets/CVUSA/train_pt' \
--warm_epoch=5 \
--batchsize=16 \
--h=112 \
--w=616 \
--droprate=0.5 \
--share \
--stride=1 \
--lr=0.02 \
--norm='spade' \
--adain='a' \
--iaa \
--multi_weather \
--btnk 1 1 0 \
--gpu_ids='0'

python test_cvusa.py \
--name='usa_res50_share_warm5_lr0.02_spade_210ep_weather_h112_110' \
--test_dir='/home/wangtyu/datasets/CVUSA/val_pt' \
--iaa \
--gpu_ids='0'
