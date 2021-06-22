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
from tqdm import tqdm, trange
from model.csnet import build_model

class Inspector():
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.device = torch.device('cuda' if self.args.cuda else 'cpu')
        self.net = build_model(predefine=self.cfg.TEST.MODEL_CONFIG)
        self.net.to(self.device)
        self.net.load_state_dict(torch.load(self.cfg.TEST.CHECKPOINT, map_location=self.device)['state_dict'])
        self.net.eval()
        # importlib.import_module('model.csnet').build_model(predefine=self.cfg.TEST.MODEL_CONFIG)

    @torch.no_grad()
    def measure_fps(self):
        num_repet = 10
        inputs = torch.rand((1, 3, self.args.input_size, self.args.input_size)).to(self.device)
        print('start warm up')
        for idx in range(30 if num_repet>=30 else num_repet): self.net(inputs)
        print('warm up done')

        if self.args.cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        for idx in range(num_repet):
            pred = F.interpolate(input=torch.sigmoid(self.net(inputs)[0].unsqueeze(0)), 
                                 size=(self.args.input_size, self.args.input_size), 
                                 mode='bilinear', align_corners=False)
        if self.args.cuda: torch.cuda.synchronize()
        t1 = time.perf_counter()
        inference_time = (t1 - t0) / num_repet
        print('FPS: {}'.format((1/inference_time)))


def parse_args():
    parser = argparse.ArgumentParser(description='CSNet')
    parser.add_argument(
        '--config',
        required=True,
        metavar='FILE',
        help='config .yml file path',
        type=str,
    )
    parser.add_argument('--runmode', type=str, choices=['fps'],
    help='fps: measure FPS of given model')    
    parser.add_argument('--input_size', type=int, help='size of one side (Assume the input is square)')
    parser.add_argument('--cpu', dest='cuda', action='store_false')
    args = parser.parse_args()
    assert os.path.isfile(args.config)
    cfg.merge_from_file(args.config)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
    if args.cuda is False: os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else: os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU)
    if cfg.TASK == '': cfg.TASK = cfg.MODEL.ARCH
    return args


def main():
    args = parse_args()
    inspector = Inspector(cfg, args)
    if args.runmode == 'fps': inspector.measure_fps()
    

if __name__ == "__main__":
    main()