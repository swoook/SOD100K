import os
import math
import argparse
import importlib
import time

import numpy as np
import torch
from torch.nn import functional as F
import skimage
from configs import cfg
from skimage import io
from skimage.transform import resize
from model.utils.simplesum_octconv import simplesum
# from CSNet.model.utils.simplesum_octconv import simplesum
from tqdm import tqdm, trange


parser = argparse.ArgumentParser(description='PyTorch SOD')
parser.add_argument(
    "--config",
    default="",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument('--cpu', dest='cuda', action='store_false')
args = parser.parse_args()
assert os.path.isfile(args.config)
cfg.merge_from_file(args.config)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
if args.cuda is False: os.environ["CUDA_VISIBLE_DEVICES"] = ''
else: os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU)
if cfg.TASK == '': cfg.TASK = cfg.MODEL.ARCH


def main():
    device = torch.device('cuda' if args.cuda else 'cpu')
    model_lib = importlib.import_module("model.csnet")
    predefine_file = cfg.TEST.MODEL_CONFIG
    model = model_lib.build_model(predefine=predefine_file)
    model.to(device)
    # prams, flops = simplesum(model, inputsize=(3, 224, 224), device=0)
    # print('  + Number of params: %.4fM' % (prams / 1e6))
    # print('  + Number of FLOPs: %.4fG' % (flops / 1e9))
    ckpt_no = cfg.TEST.CHECKPOINT
    if not os.path.isfile(ckpt_no): 
        print('{} not found.'.format(ckpt_no))
        return -1
    print("=> loading checkpoint '{}'".format(ckpt_no))
    if args.cuda is True: checkpoint = torch.load(ckpt_no)
    else: checkpoint = torch.load(ckpt_no, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(
        ckpt_no, checkpoint['epoch']))
    model.eval()

    num_repet = 30
    h, w = 224, 224
    dump_input = torch.rand((1, 3, h, w)).to(device)
    print("start warm up")
    for _ in range(num_repet): model(dump_input)
    print("warm up done")

    num_repet = 100
    if args.cuda: torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_repet):
        pred = F.interpolate(input=torch.sigmoid(model(dump_input)[0].unsqueeze(0)), size=(h, w), 
                             mode='bilinear', align_corners=False)
    if args.cuda: torch.cuda.synchronize()
    t1 = time.perf_counter()
    inference_time = (t1 - t0) / num_repet
    print('FPS: {}'.format((1/inference_time)))
    

if __name__ == "__main__":
    main()