import os
import math
import argparse
import time

import numpy as np
import torch
from torch.nn import functional as F
from skimage import io
from skimage.transform import resize
import cv2

from networks.csf_res2net import build_model, weights_init



class Inspector():
    def __init__(self, args):
        self.args = args
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.device = torch.device('cuda' if self.args.cuda else 'cpu')
        self.build_model()
        if self.args.cuda:
            self.net.load_state_dict(torch.load(self.args.pretrained_path), strict=False)
        else:
            self.net.load_state_dict(torch.load(self.args.pretrained_path, map_location='cpu'), strict=False)
        self.net.eval()

    # build the network
    def build_model(self):
        self.net = build_model()
        if self.args.cuda:
            self.net = self.net.cuda()
        self.net.apply(weights_init)

    @torch.no_grad()
    def infer_single_img(self, input_path, output_path):
        img = io.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.float32)
        img = img/255
        img -= np.array((0.485, 0.456, 0.406))
        img /= np.array((0.229, 0.224, 0.225))
        img = img.transpose((2,0,1))
        img = torch.unsqueeze(torch.FloatTensor(img), 0)
        input_var = torch.Tensor(img).to(self.device)
    
        # modified: scaling -> sigmoid
        pred = self.net(input_var)
        pred = np.squeeze(torch.sigmoid(pred).cpu().data.numpy())
        pred = 255 * pred

        output_img_dir = os.path.dirname(output_path)
        if not os.path.exists(output_img_dir): os.makedirs(output_img_dir)

        io.imsave(output_path, pred)

    def infer(self):
        '''
        1. Load an image
        2. Scale
        3. Normalize
        4. Infer
        5. Scale
        '''
        if (self.args.input_imgs_dir is None) and (self.args.output_imgs_dir is None):
            self.infer_single_img(self.args.input_img_path, self.args.output_img_path)
        elif (self.args.input_img_path is None) and (self.args.output_img_path is None):
            filenames = os.listdir(self.args.input_imgs_dir)
            input_paths = [os.path.join(self.args.input_imgs_dir, filename) 
            for filename in filenames]
            for input_path in input_paths:
                input_name = os.path.basename(input_path)
                output_path = os.path.join(self.args.output_imgs_dir, input_name)
                self.infer_single_img(input_path, output_path)

    @torch.no_grad()
    def measure_fps(self):
        num_repet = 1000
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--runmode', type=str, 
    choices=['infer', 'fps'], default='infer', 
    help='infer: infer from single image and write a result as a .jpg file \n fps: measure a FPS of given model')
    parser.add_argument('--pretrained_path', type=str, metavar='PATH',
    help='path of model file such as .pth', default=None) # Snapshot

    parser.add_argument('--input_img_path', metavar='PATH', help='Input image path', 
    default=None)
    parser.add_argument('--output_img_path', metavar='PATH', 
    help='Output image path, i.e. It visualizes an inference result', default=None)

    parser.add_argument('--input_imgs_dir', metavar='DIR', help='directory of input images', 
    default=None)
    parser.add_argument('--output_imgs_dir', metavar='DIR', help='directory of output images', 
    default=None)

    parser.add_argument('--cpu', dest='cuda', action='store_false')
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    inspector = Inspector(args)
    if args.runmode == 'fps': inspector.measure_fps()
    elif args.runmode == 'infer': inspector.infer()
    

if __name__ == "__main__":
    main()