import torch
import torch.nn as nn
import os
import argparse

from dataset import dataloader as db
from network import get_model
from utils import *
from math import log10
import pytorch_ssim

def run_eval(args):
    ckpt = torch.load(args.ckpt)
    ckpt_opt = ckpt['opts']

    transformer = get_model(ckpt_opt.net)(ckpt_opt).cuda()
    transformer.load_state_dict(ckpt['transformer'])
    transformer.eval()

    up_sampler = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True).cuda()

    video_loader = db.HRLRDLR_VideoLoader(args, video_size=(args.size[0],args.size[1]))
    print('Number of samples: {}'.format(len(video_loader)))

    criterionMSE = nn.MSELoss().cuda()
    counter = 0

    with torch.no_grad():
        for i in range(len(video_loader)):
            data_sample, _vname = video_loader.get_item(i)
            print('Processing ' + _vname)

            DLR_frames = data_sample['DLR']
            LR_frames = data_sample['LR']
            HR_frames = data_sample['HR']

            hr_writer = YUVWriter(os.path.join(args.out, _vname + '_HR.yuv'), (args.size[0]*2, args.size[1]*2))
            lr_writer = YUVWriter(os.path.join(args.out, _vname + '_LR.yuv'), (args.size[0], args.size[1]))

            errors = AverageMeters(args, _vname, ['HR-Y-PSNR', 'HR-Y-SSIM', 'LR-Y-PSNR', 'LR-Y-SSIM'])

            for frame_id in range(len(DLR_frames)):
                # Init data
                DLR = torch.autograd.Variable(torch.from_numpy(DLR_frames[frame_id].transpose((2, 0, 1))).float()).unsqueeze(0).cuda() / 255.0
                LR = torch.autograd.Variable(torch.from_numpy(LR_frames[frame_id].transpose((2, 0, 1))).float()).unsqueeze(0).cuda() / 255.0
                HR = torch.autograd.Variable(torch.from_numpy(HR_frames[frame_id].transpose((2, 0, 1))).float()).unsqueeze(0).cuda() / 255.0

                # Inference
                res_residual_out, rec_residual_out, upsampled_lr = transformer(DLR[:,0,:,:].unsqueeze(1))

                HR_out = upsampled_lr + rec_residual_out
                HR_out = torch.cat([HR_out, up_sampler(DLR[:,1:,:,:])], 1)

                LR_out = DLR[:,0,:,:].unsqueeze(1) + res_residual_out
                LR_out = torch.cat([LR_out, DLR[:,1:,:,:]], 1)

                # Evaluation
                hr_y_mse = criterionMSE(HR_out[:,0,:,:].unsqueeze(1), HR[:,0,:,:].unsqueeze(1))
                hr_y_psnr = 10 * log10(1 / hr_y_mse.item())
                hr_y_ssim = pytorch_ssim.ssim(HR_out[:,0,:,:].unsqueeze(1), HR[:,0,:,:].unsqueeze(1)).item()

                lr_y_mse = criterionMSE(LR_out[:,0,:,:].unsqueeze(1), LR[:,0,:,:].unsqueeze(1))
                lr_y_psnr = 10 * log10(1 / lr_y_mse.item())
                lr_y_ssim = pytorch_ssim.ssim(LR_out[:,0,:,:].unsqueeze(1), LR[:,0,:,:].unsqueeze(1)).item()

                errors.update([
                    hr_y_psnr, hr_y_ssim, lr_y_psnr, lr_y_ssim
                ])

                # CPU
                res_residual_out = res_residual_out.cpu()
                rec_residual_out = rec_residual_out.cpu()
                upsampled_lr = upsampled_lr.cpu()
                HR_out = HR_out.cpu()
                DLR = DLR.cpu()
                LR = LR.cpu()
                HR = HR.cpu()

                # Write frames
                hr_writer.write(HR_out)
                lr_writer.write(LR_out)
            # Finish
            hr_writer.close()
            lr_writer.close()

            errors.write_result()

    print('Test Completed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vin", type=str, default="./data/class_e/", help='path to class folder')
    parser.add_argument("--ckpt", type=str, default="./models/rrdncnn_v2.pth.tar", help='path to ckpt')
    parser.add_argument("--out", type=str, default=None, help='default is the input')
    parser.add_argument("--decoder_folder", type=str, default='DLR_LDP_QP37', help='subfolder containing videos')
    parser.add_argument("--size", type=str, default='360x640', help='360x640 - h x w')
    parser.add_argument("--log_dir", type=str, default="./logs/", help='path to ckpt')

    args = preprocess_args(parser.parse_args())
    run_eval(args)
