import os
import torch
import yaml
import torch.nn as nn
import parser
from model import ft_net, two_view_net, three_view_net

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1 # count the image number in every class
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        print('no dir: %s'%dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

######################################################################
# Save model
#---------------------------
def save_network(network, dirname, epoch_label):
    if not os.path.isdir('./model/'+dirname):
        os.mkdir('./model/'+dirname)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth'% epoch_label
    else:
        save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',dirname,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


######################################################################
#  Load model for resume
#---------------------------
def load_network(name, opt):
    # Load config
    dirname = os.path.join('./model',name)
    last_model_name = os.path.basename(get_model_list(dirname, 'net'))
    epoch = last_model_name.split('_')[1]
    epoch = epoch.split('.')[0]
    if not epoch=='last':
       epoch = int(epoch)
    config_path = os.path.join(dirname,'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)

    opt.name = config['name']
    opt.data_dir = config['data_dir']
    opt.train_all = config['train_all']
    opt.droprate = config['droprate']
    opt.color_jitter = config['color_jitter']
    opt.batchsize = config['batchsize']
    opt.h = config['h']
    opt.w = config['w']
    opt.share = config['share']
    opt.stride = config['stride']
    opt.LPN = config['LPN']
    opt.norm = config['norm']
    opt.adain = config['adain']

    if 'pool' in config:
        opt.pool = config['pool']
    if 'h' in config:
        opt.h = config['h']
        opt.w = config['w']
    if 'gpu_ids' in config:
        opt.gpu_ids = config['gpu_ids']
    opt.erasing_p = config['erasing_p']
    opt.lr = config['lr']
    opt.nclasses = config['nclasses']
    opt.erasing_p = config['erasing_p']
    opt.use_dense = config['use_dense']
    opt.fp16 = config['fp16']
    opt.views = config['views']
    try:
        config['use_vgg']
    except:
        opt.use_vgg = False
    else:
        opt.use_vgg = config['use_vgg']

    if opt.norm == 'spade':
        opt.conv_norm = config['conv_norm']
    else:
        opt.conv_norm = 'none'
    opt.block = config['block']
    if 'btnk' in config:
        opt.btnk = config['btnk']
        if len(opt.btnk) < 7:
            opt.btnk = opt.btnk + [0]*(7-len(opt.btnk))
        print('btnk------------:', opt.btnk)
    else:
        opt.btnk = [1,0,1]
    # if opt.use_dense:
    #     model = ft_net_dense(opt.nclasses, opt.droprate, opt.stride, None, opt.pool)
    # if opt.LPN:
    #     model = LPN(opt.nclasses)
    if 'use_res101' in config:
        print('--------------res101 in the config----------------')
        opt.use_res101 = config['use_res101']

    if opt.views == 3:
        if opt.LPN:
            model = three_view_net(opt.nclasses, opt.droprate, stride = opt.stride, pool = opt.pool, share_weight = opt.share, LPN=True, block=opt.block, norm=opt.norm, btnk=opt.btnk)
        else:
            print('btnk------------:', opt.btnk)
            if 'use_res101' in config and opt.use_res101:
                model = three_view_net(opt.nclasses, opt.droprate, stride = opt.stride, pool = opt.pool, share_weight = opt.share, norm = opt.norm, adain = opt.adain, btnk=opt.btnk, conv_norm=opt.conv_norm, VGG16=opt.use_vgg, Dense=opt.use_dense, ResNet101=opt.use_res101)
            else:
                model = three_view_net(opt.nclasses, opt.droprate, stride = opt.stride, pool = opt.pool, share_weight = opt.share, norm = opt.norm, adain = opt.adain, btnk=opt.btnk, conv_norm=opt.conv_norm, VGG16=opt.use_vgg, Dense=opt.use_dense)
    if 'use_vgg16' in config:
        print('--------------vgg16 in the config----------------')
        opt.use_vgg16 = config['use_vgg16']
        if opt.views == 2:
            model = two_view_net(opt.nclasses, opt.droprate, stride = opt.stride, pool = opt.pool, share_weight = opt.share, VGG16 = opt.use_vgg16, norm = opt.norm, adain = opt.adain, btnk=opt.btnk)
            if opt.LPN:
                model = two_view_net(opt.nclasses, opt.droprate, stride = opt.stride, pool = opt.pool, share_weight = opt.share, VGG16 = opt.use_vgg16, LPN = True, block=opt.block)
            # elif opt.views == 3:
            #     model = three_view_net(opt.nclasses, opt.droprate, stride = opt.stride, pool = opt.pool, share_weight = opt.share, VGG16 = opt.use_vgg16)


    # load model
    if isinstance(epoch, int):
        save_filename = 'net_%03d.pth'% epoch
    else:
        save_filename = 'net_%s.pth'% epoch
    # save_filename = 'net_099.pth'
    save_path = os.path.join('./model',name,save_filename)
    print('Load the model from %s'%save_path)
    network = model
    network_dict = network.state_dict()
    trained_dict = torch.load(save_path)
    print('different keys---------------:', (network_dict.keys()^trained_dict.keys()))
    trained_dict = {k: v for k, v in trained_dict.items() if k in network_dict}
    network_dict.update(trained_dict)
    network.load_state_dict(network_dict)
    return network, opt, epoch

def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

    toogle_grad(model_src, True)

