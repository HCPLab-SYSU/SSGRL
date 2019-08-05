import torch
import pickle
import os
import gc

def load_pretrain_model(model, args):
    model_dict = model.resnet_101.state_dict()
    print('loading pretrained model from imagenet:')
    resnet_pretrained = torch.load(args.pretrain_model)
    pretrain_dict = {k:v for k, v in resnet_pretrained.items() if not k.startswith('fc')}
    model_dict.update(pretrain_dict)
    model.resnet_101.load_state_dict(model_dict)
    del resnet_pretrained
    del pretrain_dict
    gc.collect()
    return model
