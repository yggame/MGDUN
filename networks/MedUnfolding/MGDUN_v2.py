
import torch
import torch.nn as nn
import torch.nn.functional as F

from .edsr import EDSR
from .common_2 import *


###################################  #  MAIN ARCH   #########################################################
class MGDUN_V2(nn.Module):
    def __init__(self, inchannel, n_feats, n_resblocks, rgb_range, n_colors, res_scale, scale=2, iter_num=4) -> None:
        super().__init__()
        self.iter_num = iter_num
        self.scale = scale

        self.base_sr = EDSR(n_resblocks, n_feats, scale, rgb_range, n_colors, res_scale)
        self.prox1 = BasicUnit(in_channels=inchannel, mid_channels=n_feats, out_channels=inchannel)
        self.prox2 = BasicUnit(in_channels=inchannel, mid_channels=n_feats, out_channels=inchannel)

        self.down = Conv_down(inchannel, n_feats, self.scale)
        self.up = Conv_up(inchannel, n_feats, self.scale)
        
        self.pixelshuffle_c_down = nn.PixelShuffle(2)
        self.pixelshuffle_c_up = nn.PixelUnshuffle(2)
        self.trans = InvBlock(subnet('HinResnet'), inchannel*2, inchannel*2)

        self.u = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(self.iter_num)])
        self.v = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(self.iter_num)])
        self.delta = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(self.iter_num)])
        self.eta = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(self.iter_num)])
        self.belta1 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(self.iter_num)])
        self.belta2 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(self.iter_num)])
        

    def forward(self, x, y):

        # hx = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)   # B 4 256 256
        hx = self.base_sr(x)
        u_list = []
        v_list = []
        for i in range(self.iter_num):
            if i != 0:
                ut = u_list[-1]
                vt = v_list[-1]
            else:
                ut = torch.zeros_like(hx)
                vt = torch.zeros_like(hx)
            
            ut = self.prox1(ut + self.u[i]*hx)
            vt = self.prox2(vt + self.v[i]*hx)
            hx = hx - self.delta[i]*(self.up(self.down(hx) - x) + \
                self.eta[i]*self.pixelshuffle_c_down(self.trans(self.pixelshuffle_c_up((self.pixelshuffle_c_down(self.trans(self.pixelshuffle_c_up(hx), rev=False))-y)), rev=True)) \
                    + self.belta1[i]*(hx-ut) + self.belta2[i]*(hx-vt))

            u_list.append(ut)
            v_list.append(vt)

        return hx


    def test(self, device='cpu'):
        total_params = sum(p.numel() for p in self.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        input1_tensor = torch.rand(1, 1, 128, 128)
        input2_tensor = torch.rand(1, 1, 256, 256)
        ideal_out = torch.rand(1, 1, 256, 256)
        out = self.forward(input1_tensor, input2_tensor)
        assert out.shape == ideal_out.shape

        import torchsummaryX
        torchsummaryX.summary(self, input1_tensor.to(device),input2_tensor.to(device))

if __name__=='__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # Model Hyperparameters
    parser.add_argument('--model', default='EDSR',
                    help='model name')
    parser.add_argument('--stage', type=int, default=4,
                    help='the number of stage')

    parser.add_argument('--scale', type=str, default='2',
                    help='super resolution scale')
    parser.add_argument('--color_range', default=255, type=int)
    parser.add_argument('--n_colors', type=int, default=1,
                    help='number of color channels to use')

    parser.add_argument('--act', type=str, default='relu',
                        help='activation function')
    parser.add_argument('--pre_train', type=str, default='',
                        help='pre-trained model directory')
    parser.add_argument('--extend', type=str, default='.',
                        help='pre-trained model directory')
    parser.add_argument('--n_resblocks', type=int, default=16,
                        help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parser.add_argument('--shift_mean', default=True,
                        help='subtract pixel mean from the input')
    parser.add_argument('--dilation', action='store_true',
                        help='use dilated convolution')
    parser.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'),
                        help='FP precision for test (single | half)')

    args = parser.parse_args()
    net = MGDUN_V2(1,
                args.n_feats, 
                args.n_resblocks,  
                args.color_range, 
                args.n_colors, 
                args.res_scale,
                scale = int(args.scale),
                iter_num=args.stage)
    net.test()