# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2019年12月25日 星期三 16时23分59秒
# see: https://github.com/SoftwareGift/FeatherNets_Face-Anti-spoofing-Attack-Detection-Challenge-CVPR2019/issues/72#issuecomment-550123281
import torch
from models import FeatherNet
from PIL import Image
import torchvision.transforms as transforms

transform = None
model_path = './checkpoints/FeatherNetB_bs32/_47_best.pth.tar'


def check_spoofing(image):
    image = transform(image).unsqueeze(0)
    output = model(image)
    soft_output = torch.softmax(output, dim=-1)
    _, predicted = torch.max(soft_output.data, 1)
    predicted = predicted.to('cpu').detach().numpy()
    return predicted


def get_model():
    global transform

    # Data loading code
    normalize = transforms.Normalize(mean=[0.14300402, 0.1434545, 0.14277956],
                                     std=[0.10050353, 0.100842826, 0.10034215])
    model = FeatherNet(se=True, avgdown=True)
    checkpoint = torch.load(model_path, map_location='cpu')
    print('load model:', model_path)
    model_dict = {}
    state_dict = model.state_dict()
    for (k, v) in checkpoint['state_dict'].items():
        print(k)
        if k[7:] in state_dict:
            model_dict[k[7:]] = v

    img_size = 224
    ratio = 224.0 / float(img_size)
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(int(256 * ratio)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])


if __name__ == '__main__':
    model = get_model()
    check_spoofing(Image.open('./images/fake.jpg'))
    check_spoofing(Image.open('./images/real.jpg'))
