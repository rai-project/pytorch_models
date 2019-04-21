import argparse

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from time import time
import numpy as np

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_path', default='./pretrain/netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_PATH = opt.model_path

model = torch.load(MODEL_PATH)
model.eval()


image = Image.open(IMAGE_NAME)
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
if TEST_MODE:
    image = image.cuda()

traced_script_module = torch.jit.trace(model, image)
traced_script_module.save("./python_free_models/netG_epoch_4_100.pt")


jit_model = torch.jit.load("./python_free_models/netG_epoch_4_100.pt")

output = jit_model(image)
out_img = ToPILImage()(output[0].data.cpu())
out_img.save(str(UPSCALE_FACTOR) + '_tranced_' + IMAGE_NAME)