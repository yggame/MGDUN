
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_2 import *


###################################  #  MAIN ARCH   #########################################################
class MGDUN(nn.Module):
    def __init__(self, inchannel, n_feats, scale=2, iter_num=4) -> None:
        super().__init__()
        self.iter_num = iter_num
        self.scale = scale

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

        hx = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)   # B 4 256 256

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
    net = MGDUN(1, 64)
    net.test()