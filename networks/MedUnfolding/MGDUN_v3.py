
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .edsr import EDSR
from .common import *

################################ Unet-like denoise module    ###################################################
class Encoding_Block(torch.nn.Module):
    def __init__(self, c_in, n_feat):
        super(Encoding_Block, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=c_in, out_channels=n_feat, kernel_size=3, padding=3 // 2)
        # self.conv2 = torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=3 // 2)
        # self.conv3 = torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=3 // 2)
        self.conv4 = torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=3 // 2)
        self.conv5 = torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, stride = 2, padding=3 // 2)

        self.act =  torch.nn.PReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)


    def forward(self, input):

        out1 = self.act(self.conv1(input))
        f_e = self.conv4(out1)
        down = self.act(self.conv5(f_e))
        return f_e, down


class Encoding_Block_End(torch.nn.Module):
    def __init__(self, c_in = 64, n_feat=64 ):
        super(Encoding_Block_End, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=c_in, out_channels=n_feat, kernel_size=3, padding=3 // 2)
        self.conv4 = torch.nn.Conv2d(in_channels=n_feat, out_channels=c_in, kernel_size=3, padding=3 // 2)
        self.act =  torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input):
        out1 = self.act(self.conv1(input))
        f_e = self.conv4(out1)
        return f_e

class Decoding_Block(torch.nn.Module):
    def __init__(self,c_in, n_feat):
        super(Decoding_Block, self).__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=n_feat*2, out_channels=n_feat*2, kernel_size=3, padding=3 // 2)

        self.conv3 = torch.nn.Conv2d(in_channels=n_feat*2, out_channels=n_feat, kernel_size=1)
        self.batch = 1
        self.up = torch.nn.ConvTranspose2d(c_in, n_feat, kernel_size=3, stride=2,padding=3 // 2)

        self.act =  torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def up_sampling(self, input,label, kernel_size=3, out_features=64):

        batch_size = self.batch

        label_h1 = int(label.shape[2])
        label_h2 = int(label.shape[3])

        in_features = int(input.shape[1])
        Deconv = self.up(input)

        return Deconv
    def forward(self, input, map):

        up = self.up(input, output_size = [input.shape[0],input.shape[1],map.shape[2],map.shape[3]] )
        cat = torch.cat((up, map), 1)
        cat = self.act(self.conv0(cat))

        out3 = self.conv3(cat)

        return out3

class Feature_Decoding_End(torch.nn.Module):
    def __init__(self, n_feat, c_out):
        super(Feature_Decoding_End, self).__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=n_feat*2, out_channels=n_feat, kernel_size=3, padding=3 // 2)

        self.conv3 = torch.nn.Conv2d(in_channels=n_feat, out_channels=c_out, kernel_size=3, padding=3 // 2)
        self.batch = 1
        self.up = torch.nn.ConvTranspose2d(n_feat, n_feat, kernel_size=3, stride=2, padding=3 // 2)
        self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
    def up_sampling(self, input,label, kernel_size=3, out_features=64):

        batch_size = self.batch

        label_h1 = int(label.shape[2])
        label_h2 = int(label.shape[3])

        in_features = int(input.shape[1])

        Deconv = self.up(input)

        return Deconv
    def forward(self, input,map):

        up = self.up(input,  output_size = [input.shape[0],input.shape[1],map.shape[2],map.shape[3]] )
        cat = torch.cat((up, map), 1)
        cat = self.act(self.conv0(cat))

        out3 = self.conv3(cat)

        return out3

class Unet_Spatial(torch.nn.Module):
    def __init__(self, cin, n_feat):
        super(Unet_Spatial, self).__init__()

        self.Encoding_block1 = Encoding_Block(cin, n_feat)
        self.Encoding_block2 = Encoding_Block(n_feat, n_feat)
        self.Encoding_block3 = Encoding_Block(n_feat, n_feat)
        self.Encoding_block4 = Encoding_Block(n_feat, n_feat)
        self.Encoding_block_end = Encoding_Block_End(n_feat, n_feat)

        self.Decoding_block1 = Decoding_Block(n_feat, n_feat)
        self.Decoding_block2 = Decoding_Block(n_feat, n_feat)
        self.Decoding_block3 = Decoding_Block(n_feat, n_feat)
        self.Decoding_block_End = Feature_Decoding_End(n_feat, cin)

        self.acti = torch.nn.PReLU()
        self.reset_parameters()
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        sz = x.shape

        encode0, down0 = self.Encoding_block1(x)
        encode1, down1 = self.Encoding_block2(down0)
        encode2, down2 = self.Encoding_block3(down1)
        encode3, down3 = self.Encoding_block4(down2)

        media_end = self.Encoding_block_end(down3)

        decode3 = self.Decoding_block1(media_end, encode3)
        decode2 = self.Decoding_block2(decode3, encode2)
        decode1 = self.Decoding_block3(decode2, encode1)
        decode0 = self.Decoding_block_End(decode1, encode0)
        return decode0 + x

###################################  #  MAIN ARCH   #########################################################
class MGDUN_V3(nn.Module):
    def __init__(self, inchannel, n_feats, n_resblocks, rgb_range, n_colors, res_scale, scale=2, iter_num=4) -> None:
        super().__init__()
        self.iter_num = iter_num
        self.scale = scale

        self.prox = Unet_Spatial(1, 16)

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

        hx = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)   # B 4 256 256
        # hx = self.base_sr(x)
        u_list = []
        v_list = []
        for i in range(self.iter_num):
            if i != 0:
                ut = u_list[-1]
                vt = v_list[-1]
            else:
                ut = torch.zeros_like(hx)
                vt = torch.zeros_like(hx)
            
            ut = self.prox((ut + self.u[i]*(hx))) 
            vt = self.prox((vt + self.v[i]*(hx))) 

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
    net = MGDUN_V3(1,
                args.n_feats, 
                args.n_resblocks,  
                args.color_range, 
                args.n_colors, 
                args.res_scale,
                scale = int(args.scale),
                iter_num=args.stage)
    net.test()
