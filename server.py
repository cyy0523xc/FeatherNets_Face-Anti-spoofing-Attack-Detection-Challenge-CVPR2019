# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2019年12月25日 星期三 14时12分37秒
import os
import time
import yaml
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd.variable import Variable
import models

gpus = '0'
USE_GPU = torch.cuda.is_available()
device = torch.device('cuda:%s' % gpus
                      if torch.cuda.is_available() else "cpu")
model = None
trans = None


class Args:
    config = 'cfgs/FeatherNet54-se-64.yaml'
    resume = './checkpoints/FeatherNet54-se/_68_best.pth.tar'
    input_size = 224
    image_size = 224
    lr = 0.1
    momentum = 0.9
    arch = ''
    gpus = gpus
    weight_decay = 1e-4


def init(args):
    global model, trans
    init_time = time.time()
    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in config['common'].items():
        setattr(args, k, v)

    if "model" in config.keys():
        model = models.__dict__[args.arch](**config['model'])
    else:
        model = models.__dict__[args.arch]()

    if USE_GPU:
        # cudnn.benchmark = True
        # torch.cuda.manual_seed_all(args.random_seed)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        model = torch.nn.DataParallel(model, device_ids=args.gpus)
        model.to(device)

    # optionally resume from a checkpoint
    print(os.getcwd())
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    model.eval()
    normalize = transforms.Normalize(mean=[0.14300402, 0.1434545, 0.14277956],
                                     std=[0.10050353, 0.100842826, 0.10034215])
    img_size = args.input_size
    ratio = 224.0 / float(img_size)
    trans = transforms.Compose([
        transforms.Resize(int(256 * ratio)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    print('Init Time: %.2f' % (time.time() - init_time))
    return args


def predict(image, args):
    """
    :param image Image.open and RGB
    """
    pred_time = time.time()
    image = trans(image)
    with torch.no_grad():
        input_var = Variable(image).float().to(device)
        output = model(input_var)
        soft_output = torch.softmax(output, dim=-1)
        _, predicted = torch.max(soft_output.data, 1)
        predicted = predicted.to('cpu').detach().numpy()

    print('Predict Time: %.2f' % (time.time() - pred_time))
    return predicted


if __name__ == '__main__':
    import sys
    args = init(Args())
    image = Image.open(sys.argv[1])
    predicted = predict(image.convert('RGB'), args)
    print(predicted)
