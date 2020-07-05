import os
import numpy as np
import torch.nn as nn
import csv

def preprocess_args(in_args):
    args = in_args
    if args.out is None:
        args.out = os.path.join(args.vin, os.path.basename(args.ckpt).split('.')[0] + '_S' + args.decoder_folder)
    args.size = [int(k) for k in args.size.split('x')]
    if not os.path.exists(args.out):
        os.mkdir(args.out)
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    print(args)
    return args

class YUVWriter(object):
    def __init__(self, video_out, video_size):
        self.size = video_size
        self.out = video_out
        self.down_sampler = nn.AvgPool2d(kernel_size=2, stride=2).cuda()
        self.writer = open(video_out, 'wb')

    def write(self, _tensor):
        y_tensor = _tensor[:,0,:,:].squeeze(0).cpu().detach().numpy() * 255.0
        uv_tensor = self.down_sampler(_tensor[:,1:,:,:]).squeeze(0).cpu().detach().numpy() * 255.0

        u_tensor = np.reshape(uv_tensor[0, :, :], (self.size[0]//4, self.size[1]))
        v_tensor = np.reshape(uv_tensor[1, :, :], (self.size[0]//4, self.size[1]))
        yuv_tensor = np.concatenate((y_tensor,u_tensor,v_tensor), axis=0).astype(np.uint8)

        yuv_frame = yuv_tensor.reshape((yuv_tensor.shape[0]*yuv_tensor.shape[1], ))
        binary_frame = yuv_frame.tostring()
        self.writer.write(binary_frame)

    def close(self):
        self.writer.close()

class AverageMeters(object):
    def __init__(self, args, _vname, names=['HR-Y-PSNR', 'HR-Y-SSIM', 'LR-Y-PSNR', 'LR-Y-SSIM']):
        self.names = names
        self.info = [os.path.basename(args.ckpt).split('.')[0], _vname, args.decoder_folder]
        self.log_dir = args.log_dir
        self.reset()
    
    def update(self, loss_list):
        self.values += np.array(loss_list)
        self.nof_samples += 1

    def get_mean_values(self):
        return self.values/self.nof_samples
    
    def reset(self):
        self.values = np.zeros(len(self.names))
        self.nof_samples = 0
    
    def write_result(self):
        csv_content = ['_'.join(self.info)]
        mean_vals = self.get_mean_values()
        print('Video: ' + self.info[1])
        for i in range(len(self.names)):
            print("===> Avg. {}: {:.4f} dB".format(self.names[i], mean_vals[i]))
            csv_content.append('{:.4f}'.format(mean_vals[i]))

        csv_out = os.path.join(self.log_dir, '{}_{}.csv'.format(self.info[2].split('_')[1], self.info[0]))
        print('Writing the result to ' + csv_out)
        with open(csv_out, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(csv_content)
