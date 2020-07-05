import torch
import torch.nn as nn

def get_model(name):
    return {
        "v1.0": RRDnCNN,
        "v2.0": RRDnCNN_v2,
    }[name]

class RRDnCNN(nn.Module):
    def __init__(self, args):
        super(RRDnCNN, self).__init__()

        print('Init RR-DnCNN')

        self.restore = self.make_layers(args.img_channel, args.n_channels, None, args.res_depth, customize=[1, 0])
        self.conv_1 = nn.Conv2d(in_channels=args.n_channels, out_channels=args.img_channel, kernel_size=3, padding=1, bias=True)
        self.upsampler_1 = nn.ConvTranspose2d(in_channels=args.img_channel, out_channels=args.img_channel, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.upsampler_2 = nn.ConvTranspose2d(in_channels=args.n_channels, out_channels=args.n_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

        self.reconstruct = self.make_layers(args.n_channels, args.n_channels, args.img_channel, args.rec_depth, customize=[0, 1])

    def forward(self, x):
        y = x
        out_1 = self.restore(x)
        res_residual_out = self.conv_1(out_1)
        upsampled_lr = self.upsampler_1(y+res_residual_out)
        rec_residual_out = self.reconstruct(self.upsampler_2(out_1))

        return res_residual_out, rec_residual_out, upsampled_lr

    def make_layers(self, in_channels, n_channels, out_channels, depth, customize=[False, False]):
        layers = []

        if customize[0]:
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=n_channels, kernel_size=5, stride=1, padding=2, bias=True))
            layers.append(nn.LeakyReLU(0.1))
        for i in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1, bias=True))
            layers.append(nn.LeakyReLU(0.1))
        if customize[1]:
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True))
        return nn.Sequential(*layers)

class convLayer(nn.Module):
    def __init__(self, nInputs, nOutputs, kernel_size=3, stride=1, padding=1):
        super(convLayer, self).__init__()

        self.conv1 = nn.Conv2d(nInputs, nOutputs, kernel_size, stride, padding, bias=True)
        self.conv2 = nn.Conv2d(nOutputs, nOutputs, kernel_size, stride, padding, bias=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, _input, _skip_feat=None):
        X = _input
        X = self.conv1(X)
        X = self.relu(X)
        if _skip_feat is not None:
            X = X + _skip_feat
        X = self.conv2(X)
        X = self.relu(X)
        return X

class RRDnCNN_v2(nn.Module):
    def __init__(self, args):
        super(RRDnCNN_v2, self).__init__()

        print('Init RR-DnCNN v2.0')
        if args.res_depth != args.rec_depth:
            print('Restoration and Reconstruction should have the same length in version 2.0.\nTheir length is set as %d automatically.' % args.res_depth)
            args.rec_depth = args.res_depth
        n_channels=args.n_channels
        self.depth = args.res_depth

        self.res_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=args.img_channel, out_channels=n_channels, kernel_size=5, stride=1, padding=2, bias=True),
            nn.LeakyReLU(0.1)
        )
        self.res_final = nn.Conv2d(in_channels=n_channels, out_channels=args.img_channel, kernel_size=3, padding=1, bias=True)

        for i in range(1,self.depth):
            setattr(self, 'res_conv_{}'.format(i+1), convLayer(n_channels, n_channels, kernel_size=3, stride=1, padding=1))

        for i in range(self.depth):
            setattr(self, 'rec_conv_{}'.format(i+1), convLayer(n_channels, n_channels, kernel_size=3, stride=1, padding=1))
        
        self.upsampler_bi = nn.ConvTranspose2d(in_channels=args.img_channel, out_channels=args.img_channel, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

        for i in range(self.depth):
            setattr(self, 'upsampler_{}'.format(i+1), nn.ConvTranspose2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True))

        self.res_final_conv = nn.Conv2d(n_channels,args.img_channel,3,1,1,bias=True)
        self.rec_final_conv = nn.Conv2d(n_channels,args.img_channel,3,1,1,bias=True)

    def forward(self, _input):

        skips = []
        y = _input
        X1 = self.res_conv_1(_input)
        skips.append(X1)

        for i in range(1, self.depth):
            X1 = getattr(self,'res_conv_{}'.format(i+1))(X1)
            skips.append(X1)
        res_residual_out = self.res_final_conv(X1)

        upsampled_lr = self.upsampler_bi(y + res_residual_out)

        X2 = skips.pop()
        skips.append(None) # skip the first one
        X2 = self.rec_conv_1(self.upsampler_1(X2))

        for i in range(1, self.depth):
            X2 = getattr(self,'rec_conv_{}'.format(i+1))(
                X2,
                getattr(self,'upsampler_{}'.format(i+1))(skips[-i-1])
            )

        rec_residual_out = self.rec_final_conv(X2)
        return res_residual_out, rec_residual_out, upsampled_lr
