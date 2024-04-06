from thop import profile
from thop import clever_format
from model import three_view_net
from torch.autograd import Variable
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = Variable(torch.FloatTensor(1, 3, 256, 256)).cuda()
# net = three_view_net(701, droprate=0.75, pool='avg', stride=1, share_weight=True, LPN=False, block=2, norm='bn', ResNet101=True).cuda()
net = three_view_net(701, droprate=0.75, pool='avg', stride=1, share_weight=True, LPN=True, block=4, norm='bn').cuda()
print(net)
total_ops, total_params = profile(net, (input, input, input), verbose=False)
macs, params = clever_format([total_ops, total_params], "%.3f")
print('MACs:',macs)
print('Paras:',params)

