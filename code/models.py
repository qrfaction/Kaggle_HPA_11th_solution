from torchvision.models import *
from torch import nn
from config import *
from collections import OrderedDict
import torch.nn.functional as F
import torch
import pretrainedmodels
from Nadam import Nadam
from utils import *
from sync_models import sysc_model,DataParallelWithCallback,patch_replication_callback
import types

def resnet18_model(cfg):
    model = resnet18(pretrained=True)
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
    model.conv1.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Sequential(
                nn.Linear(512,28),
                nn.Sigmoid()
    )
    if cfg['half']:
        model.half()
        bn_to_float(model)
        print('---half')
    main_params  = list(map(id, model.fc.parameters()))
    main_params += list(map(id, model.conv1.parameters()))
    main_params += list(map(id, model.bn1.parameters()))
    base_params  = filter(lambda p: id(p) not in main_params, model.parameters())
    opt = Nadam([
        {'params': base_params, 'lr': cfg['backbone_lr']},
        {'params': model.fc.parameters()},
        {'params': model.conv1.parameters()},
        {'params': model.bn1.parameters()},
    ], lr=cfg['lr'])
    return model,opt


def resnet34_model(cfg):
    model = resnet34(pretrained=True)
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
    model.conv1.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Sequential(
                nn.Linear(512,28),
                nn.Sigmoid()
    )
    if cfg['half']:
        model.half()
        bn_to_float(model)
        print('---half')
    main_params  = list(map(id, model.fc.parameters()))
    main_params += list(map(id, model.conv1.parameters()))
    main_params += list(map(id, model.bn1.parameters()))
    base_params  = filter(lambda p: id(p) not in main_params, model.parameters())
    opt = Nadam([
        {'params': base_params, 'lr': cfg['backbone_lr']},
        {'params': model.fc.parameters()},
        {'params': model.conv1.parameters()},
        {'params': model.bn1.parameters()},
    ], lr=cfg['lr'])
    return model,opt

def resnet50_model(cfg):
    model = resnet50(pretrained=True)
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
    model.conv1.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Sequential(
                nn.Linear(2048,28),
                nn.Sigmoid()
    )
    if cfg['half']:
        model.half()
        bn_to_float(model)
        print('---half')
    if len(cfg['device']) > 1:
        model = sysc_model(model)
        model = DataParallelWithCallback(model, device_ids=cfg['device'])

        main_params = list(map(id, model.module.fc.parameters()))
        main_params += list(map(id, model.module.conv1.parameters()))
        main_params += list(map(id, model.module.bn1.parameters()))
        base_params = filter(lambda p: id(p) not in main_params, model.parameters())
        opt = Nadam([
            {'params': base_params, 'lr': cfg['backbone_lr']},
            {'params': model.module.fc.parameters()},
            {'params': model.module.conv1.parameters()},
            {'params': model.module.bn1.parameters()},
        ], lr=cfg['lr'])
    else:
        main_params  = list(map(id, model.fc.parameters()))
        main_params += list(map(id, model.conv1.parameters()))
        main_params += list(map(id, model.bn1.parameters()))
        base_params  = filter(lambda p: id(p) not in main_params, model.parameters())
        opt = Nadam([
            {'params': base_params, 'lr': cfg['backbone_lr']},
            {'params': model.fc.parameters()},
            {'params': model.conv1.parameters()},
            {'params': model.bn1.parameters()},
        ], lr=cfg['lr'])
    return model,opt

def se_resnext50(cfg):
    model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet', num_classes=1000)
    w = model.layer0.conv1.weight
    model.layer0.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.layer0.conv1.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.Linear(2048, 28),
        nn.Sigmoid()
    )
    if cfg['half']:
        model.half()
        bn_to_float(model)
        print('---half')
    if len(cfg['device']) > 1:
        model = sysc_model(model)
        model = DataParallelWithCallback(model, device_ids=cfg['device'])
        # model = nn.DataParallel(model, device_ids=cfg['device'])
        main_params = list(map(id, model.module.last_linear.parameters()))
        main_params += list(map(id, model.module.layer0.conv1.parameters()))
        main_params += list(map(id, model.module.layer0.bn1.parameters()))
        base_params = filter(lambda p: id(p) not in main_params, model.parameters())
        opt = Nadam([
            {'params': base_params, 'lr': cfg['backbone_lr']},
            {'params': model.module.last_linear.parameters()},
            {'params': model.module.layer0.conv1.parameters()},
            {'params': model.module.layer0.bn1.parameters()},
        ], lr=cfg['lr'])
    else:
        main_params =  list(map(id, model.last_linear.parameters()))
        main_params += list(map(id, model.layer0.conv1.parameters()))
        main_params += list(map(id, model.layer0.bn1.parameters()))
        base_params  = filter(lambda p: id(p) not in main_params, model.parameters())
        opt = Nadam([
            {'params': base_params, 'lr': cfg['backbone_lr']},
            {'params': model.last_linear.parameters()},
            {'params': model.layer0.conv1.parameters()},
            {'params': model.layer0.bn1.parameters()},
        ], lr=cfg['lr'])
    return model, opt

def bninception_model(cfg):
    model = pretrainedmodels.models.bninception(pretrained='imagenet')
    w = model.conv1_7x7_s2.weight
    model.conv1_7x7_s2 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1_7x7_s2.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
                nn.Linear(1024, 28),
                nn.Sigmoid()
            )
    if cfg['half']:
        model.half()
        bn_to_float(model)
        print('---half')
    main_params = list(map(id, model.last_linear.parameters()))
    main_params += list(map(id, model.conv1_7x7_s2.parameters()))
    main_params += list(map(id, model.conv1_7x7_s2_bn.parameters()))
    base_params = filter(lambda p: id(p) not in main_params, model.parameters())
    opt = Nadam([
        {'params': base_params, 'lr': cfg['backbone_lr']},
        {'params': model.last_linear.parameters()},
        {'params': model.conv1_7x7_s2.parameters()},
        {'params': model.conv1_7x7_s2_bn.parameters()},
    ], lr=cfg['lr'])
    return model, opt

def inceptionv3_model(cfg):
    model = pretrainedmodels.models.inceptionv3(pretrained='imagenet')
    w = model.Conv2d_1a_3x3.conv.weight
    model.Conv2d_1a_3x3.conv = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    model.Conv2d_1a_3x3.conv.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))
    model.last_linear = nn.Sequential(
        nn.Linear(2048, 28),
        nn.Sigmoid()
    )

    def features(self, input):
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(input) # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x) # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x) # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2) # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x) # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x) # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2) # 35 x 35 x 192
        x = self.Mixed_5b(x) # 35 x 35 x 256
        x = self.Mixed_5c(x) # 35 x 35 x 288
        x = self.Mixed_5d(x) # 35 x 35 x 288
        x = self.Mixed_6a(x) # 17 x 17 x 768
        x = self.Mixed_6b(x) # 17 x 17 x 768
        x = self.Mixed_6c(x) # 17 x 17 x 768
        x = self.Mixed_6d(x) # 17 x 17 x 768
        x = self.Mixed_6e(x) # 17 x 17 x 768
        x = self.Mixed_7a(x) # 8 x 8 x 1280
        x = self.Mixed_7b(x) # 8 x 8 x 2048
        x = self.Mixed_7c(x) # 8 x 8 x 2048
        return x
    def logits(self, features):
        x = F.adaptive_avg_pool2d(features,1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x
    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    if cfg['half']:
        model.half()
        bn_to_float(model)
        print('---half')
    main_params = list(map(id, model.last_linear.parameters()))
    main_params += list(map(id, model.Conv2d_1a_3x3.parameters()))
    base_params = filter(lambda p: id(p) not in main_params, model.parameters())

    opt = Nadam([
        {'params': base_params, 'lr': cfg['backbone_lr']},
        {'params': model.last_linear.parameters()},
        {'params': model.Conv2d_1a_3x3.parameters()},
    ], lr=cfg['lr'])
    return model, opt


def nasnetmobile_model(cfg):
    model = pretrainedmodels.models.nasnetamobile(pretrained='imagenet',num_classes=1000)
    w = model.conv0.conv.weight
    model.conv0.conv = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    model.conv0.conv.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))
    model.last_linear = nn.Sequential(
        nn.Linear(1056, 28),
        nn.Sigmoid()
    )
    def features(self, input):
        x_conv0 = self.conv0(input)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)

        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)

        x_reduction_cell_0 = self.reduction_cell_0(x_cell_3, x_cell_2)

        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_3)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)

        x_reduction_cell_1 = self.reduction_cell_1(x_cell_9, x_cell_8)

        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_9)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        return x_cell_15

    def logits(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x,1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    if cfg['half']:
        model.half()
        bn_to_float(model)
        print('---half')
    if len(cfg['device']) > 1:
        model = sysc_model(model)
        model = DataParallelWithCallback(model, device_ids=cfg['device'])

        main_params = list(map(id, model.module.last_linear.parameters()))
        main_params += list(map(id, model.module.conv0.parameters()))
        base_params = filter(lambda p: id(p) not in main_params, model.parameters())
        opt = Nadam([
            {'params': base_params, 'lr': cfg['backbone_lr']},
            {'params': model.module.last_linear.parameters()},
            {'params': model.module.conv0.parameters()},
        ], lr=cfg['lr'])
    else:
        main_params = list(map(id, model.last_linear.parameters()))
        main_params += list(map(id, model.conv0.parameters()))
        base_params = filter(lambda p: id(p) not in main_params, model.parameters())

        opt = Nadam([
            {'params': base_params, 'lr': cfg['backbone_lr']},
            {'params': model.last_linear.parameters()},
            {'params': model.conv0.parameters()},
        ], lr=cfg['lr'])
    return model, opt


def xception_model(cfg):
    model = pretrainedmodels.models.xception(pretrained='imagenet')
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    model.conv1.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))
    model.last_linear = nn.Sequential(
                nn.Linear(2048, 28),
                nn.Sigmoid()
            )
    if cfg['half']:
        model.half()
        bn_to_float(model)
        print('---half')
    main_params = list(map(id, model.last_linear.parameters()))
    main_params += list(map(id, model.bn1.parameters()))
    main_params += list(map(id, model.conv1.parameters()))
    base_params = filter(lambda p: id(p) not in main_params, model.parameters())
    opt = Nadam([
        {'params': base_params, 'lr': cfg['backbone_lr']},
        {'params': model.last_linear.parameters()},
        {'params': model.conv1.parameters()},
        {'params': model.bn1.parameters()},
    ], lr=cfg['lr'])
    return model, opt

def dpnb_model(cfg):
    model = pretrainedmodels.models.dpn68b(pretrained='imagenet+5k')
    w = model.features.conv1_1.conv.weight
    model.features.conv1_1.conv = nn.Conv2d(4, 10, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),bias=False)
    model.features.conv1_1.conv.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))
    model.last_linear = nn.Sequential(
        nn.Conv2d(832, 28, kernel_size=(1, 1), stride=(1, 1)),
        nn.Sigmoid()
    )
    if cfg['half']:
        model.half()
        bn_to_float(model)
        print('---half')
    if len(cfg['device']) > 1:
        model = sysc_model(model)
        model = DataParallelWithCallback(model, device_ids=cfg['device'])

        main_params = list(map(id, model.module.last_linear.parameters()))
        main_params += list(map(id, model.module.features.conv1_1.parameters()))
        base_params = filter(lambda p: id(p) not in main_params, model.parameters())
        opt = Nadam([
            {'params': base_params, 'lr': cfg['backbone_lr']},
            {'params': model.module.last_linear.parameters()},
            {'params': model.module.features.conv1_1.parameters()},
        ], lr=cfg['lr'])
    else:
        main_params = list(map(id, model.last_linear.parameters()))
        main_params += list(map(id, model.features.conv1_1.parameters()))
        base_params = filter(lambda p: id(p) not in main_params, model.parameters())
        opt = Nadam([
            {'params': base_params, 'lr': cfg['backbone_lr']},
            {'params': model.last_linear.parameters()},
            {'params': model.features.conv1_1.parameters()},
        ], lr=cfg['lr'])
    return model, opt

class weightedBCELoss(nn.Module):

    def __init__(self,alpha):
        super(weightedBCELoss, self).__init__()
        self.alpha = alpha
    def forward(self, input, target, dim=None):
        loss = F.binary_cross_entropy(input, target, weight=None, reduction='none')
        loss = self.alpha * target *loss + (2-self.alpha) * (1-target) *loss
        if dim:
            return torch.mean(loss,dim)
        else:
            return torch.mean(loss)



if __name__ == '__main__':
    # print(resnet_model(None))
    cfg = {'device':[0,1]}
    m = nasnetmobile_model(cfg)
    print(m)


