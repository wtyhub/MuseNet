# train spade net
python train.py \
--name='three_view_long_share_d0.5_256_s1_google_lr0.005_spade_v24.11_210ep_weather_0110000_alpha1_test' \
--experiment_name='three_view_long_share_d0.5_256_s1_google_lr0.005_spade_v24.11_210ep_weather_0110000_alpha1_test' \
--data_dir='/home/wangtingyu/datasets/University-Release/train' \
--views=3 \
--droprate=0.5 \
--extra \
--share \
--stride=1 \
--h=256 \
--w=256 \
--lr=0.005 \
--gpu_ids='0' \
--norm='spade' \
--iaa \
--multi_weather \
--btnk 0 1 1 0 0 0 0 \
--conv_norm='none' \
--alpha=1 \
--adain='a'

python test_iaa.py \
--name='three_view_long_share_d0.5_256_s1_google_lr0.005_spade_v24.11_210ep_weather_0110000_alpha1_test' \
--test_dir='/home/wangtingyu/datasets/University-Release/test' \
--iaa \
--gpu_ids='2'


# training ibn net
# python train.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.005_ibn_v24.11_210ep_weather' \
# --experiment_name='three_view_long_share_d0.5_256_s1_google_lr0.005_ibn_v24.11_210ep_weather' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.5 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.005 \
# --gpu_ids='4' \
# --norm='ibn' \
# --iaa \
# --multi_weather \
# --btnk 1 0 0 0 0 0 0 \
# --conv_norm='none' \
# --adain='a'

# python test_iaa.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.005_ibn_v24.11_210ep_weather' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --iaa \
# --gpu_ids='4'

# training ibn net
#python train.py \
#--name='three_view_long_share_d0.75_256_s1_google_lr0.005_ibn_v21.9_210ep_weather_r6' \
#--experiment_name='three_view_long_share_d0.75_256_s1_google_lr0.005_ibn_v21.9_210ep_weather_r6' \
#--data_dir='/home/wangtyu/datasets/University-Release/train' \
#--views=3 \
#--droprate=0.75 \
#--extra \
#--share \
#--stride=1 \
#--h=256 \
#--w=256 \
#--lr=0.005 \
#--gpu_ids='2' \
#--norm='ibn' \
#--iaa \
#--multi_weather \
#--adain='a'
#
#python test_iaa.py \
#--name='three_view_long_share_d0.75_256_s1_google_lr0.005_ibn_v21.9_210ep_weather_r6' \
#--test_dir='/home/wangtyu/datasets/University-Release/test' \
#--iaa \
#--gpu_ids='2'

# train original university1652 ada-ibn
#python train.py \
#--name='three_view_long_share_d0.75_256_s1_google_lr0.005_ada-ibn_v21.9_210ep_no-style-loss' \
#--experiment_name='three_view_long_share_d0.75_256_s1_google_lr0.005_ada-ibn_v21.9_210ep_no-style-loss' \
#--data_dir='/home/wangtyu/datasets/University-Release/train' \
#--views=3 \
#--droprate=0.75 \
#--extra \
#--share \
#--stride=1 \
#--h=256 \
#--w=256 \
#--lr=0.005 \
#--gpu_ids='2' \
#--norm='ada-ibn' \
#--iaa \
#--adain='a'
#
#python test_iaa.py \
#--name='three_view_long_share_d0.75_256_s1_google_lr0.005_ada-ibn_v21.9_210ep_no-style-loss' \
#--test_dir='/home/wangtyu/datasets/University-Release/test' \
#--iaa \
#--gpu_ids='2'

# train original university1652 ibn
#python train.py \
#--name='three_view_long_share_d0.75_256_s1_google_lr0.005_ibn_v21.9_210ep' \
#--experiment_name='three_view_long_share_d0.75_256_s1_google_lr0.005_ibn_v21.9_210ep' \
#--data_dir='/home/wangtyu/datasets/University-Release/train' \
#--views=3 \
#--droprate=0.75 \
#--extra \
#--share \
#--stride=1 \
#--h=256 \
#--w=256 \
#--lr=0.005 \
#--gpu_ids='1' \
#--norm='ibn' \
#--iaa \
#--adain='a'
#
#python test_iaa.py \
#--name='three_view_long_share_d0.75_256_s1_google_lr0.005_ibn_v21.9_210ep' \
#--test_dir='/home/wangtyu/datasets/University-Release/test' \
#--iaa \
#--gpu_ids='1'

# train LPN weather
# python train.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.001_LPN_v24.11_210ep_weather' \
# --experiment_name='three_view_long_share_d0.5_256_s1_google_lr0.001_LPN_v24.11_210ep_weather' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.5 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.001 \
# --gpu_ids='5' \
# --LPN \
# --iaa \
# --multi_weather \
# --block=4 \
# --conv_norm='none' \
# --adain='a'

# python test_iaa.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.001_LPN_v24.11_210ep_weather' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --iaa \
# --gpu_ids='5'

# LPN + Spade
# python train.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.001_LPN_Spade_v24.11_210ep_weather' \
# --experiment_name='three_view_long_share_d0.5_256_s1_google_lr0.001_LPN_Spade_v24.11_210ep_weather' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.5 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.001 \
# --gpu_ids='0' \
# --LPN \
# --iaa \
# --multi_weather \
# --btnk 0 1 1 0 0 0 0 \
# --norm='spade' \
# --block=4 \
# --conv_norm='none' \
# --adain='a'

# python test_iaa.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.001_LPN_Spade_v24.11_210ep_weather' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --iaa \
# --gpu_ids='4'

# train vgg16 weather
# python train.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.005_vgg_v24.11_210ep_weather1' \
# --experiment_name='three_view_long_share_d0.5_256_s1_google_lr0.005_vgg_v24.11_210ep_weather1' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.5 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.005 \
# --gpu_ids='3' \
# --iaa \
# --use_vgg \
# --multi_weather \
# --btnk 1 0 0 0 0 0 0 \
# --conv_norm='none' \
# --adain='a'

# python test_iaa.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.005_vgg_v24.11_210ep_weather1' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --iaa \
# --gpu_ids='3'

# train ResnNet101 weather
# python train.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.005_Res101_v24.11_210ep_weather' \
# --experiment_name='three_view_long_share_d0.5_256_s1_google_lr0.005_Res101_v24.11_210ep_weather' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.5 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.005 \
# --gpu_ids='0' \
# --iaa \
# --use_res101 \
# --multi_weather \
# --btnk 0 0 0 0 0 0 0 \
# --conv_norm='none' \
# --adain='a'

# python test_iaa.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.005_Res101_v24.11_210ep_weather' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --iaa \
# --gpu_ids='0'
