import os
import os.path as osp
import numpy as np

import glob
from utils import *
from dataset.data_help import *

def init_dataset(args, ext='.npy'):
    dlr_list = []
    lr_list = []
    hr_list = []

    if ext=='.yuv':
        dlr_list = sorted(glob.glob(osp.join(args.vin, args.decoder_folder, '*.yuv')))
        videos = [osp.basename(k).replace('.yuv', '') for k in dlr_list]
    else:
        videos = sorted(glob.glob(osp.join(args.vin, args.decoder_folder, '*')))
        videos = [osp.basename(k) for k in videos]
        for video_name in videos:
            dlr_list += sorted(glob.glob(osp.join(args.vin, args.decoder_folder, video_name, '*' + ext)))
    lr_list = [k.replace(args.decoder_folder, 'LR') for k in dlr_list]
    hr_list = [k.replace(args.decoder_folder, 'HR') for k in dlr_list]

    return dlr_list, lr_list, hr_list, videos

class HRLRDLR_VideoLoader:
    def __init__(self, args, video_size=(None, None)):

        self.vin = args.vin
        self.video_size = video_size
        self.args = args

        # Initialize the per sequence images for online training
        self.dlr_list, self.lr_list, self.hr_list, self.videos = init_dataset(args, ext='.yuv')
        print('Done initializing Dataset')

    def __len__(self):
        return len(self.dlr_list)

    def get_item(self, idx):
        
        dlr_size = [self.video_size[0], self.video_size[1]]
        dlr_cap = VideoProcessor(self.dlr_list[idx], self.video_size[0], self.video_size[1])

        lr_size = [self.video_size[0], self.video_size[1]]

        lr_cap = VideoProcessor(self.lr_list[idx], lr_size[0], lr_size[1])
        hr_cap = VideoProcessor(self.hr_list[idx], lr_size[0]*2, lr_size[1]*2)

        sample = {
            'DLR': dlr_cap.vid_arr,
            'LR': lr_cap.vid_arr,
            'HR': hr_cap.vid_arr,
        }

        return sample, self.videos[idx]
