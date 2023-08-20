import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
#import functools
from torch.optim import lr_scheduler
import math
import time
from torch.autograd import Variable
import torch.nn.functional as F

def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).cuda()
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

def gradient_norm(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).cuda()
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return dydx_l2norm

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)
    
class Encoder(nn.Module):
    def __init__(self, zdim, cdim, nc=1):
        super(Encoder, self).__init__()
        self.zdim, self.cdim = zdim, cdim
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.Conv2d(nc + self.cdim, nf, 3, 1, 1),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf) x 64 x 64
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c5 = dcgan_conv(nf * 8, nf * 16)
        # state size. (nf*16) x 4 x 4
        self.mu_net = nn.Conv2d(nf * 16, self.zdim, 4, 1, 0)
        self.logvar_net = nn.Conv2d(nf * 16, self.zdim, 4, 1, 0)
    
    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)
    
    def forward(self, x_src, c_src, return_mean_logvar = False):
        h, w = np.shape(x_src)[2], np.shape(x_src)[3]
        c_src = torch.tensor(torch.unsqueeze(c_src, dim=2), dtype=torch.float32).cuda()
        c_src = torch.reshape(c_src @ torch.ones((np.shape(c_src)[0], 1, h*w)).cuda(),
                              (np.shape(c_src)[0], self.cdim, h, w))
        h1 = self.c1(torch.cat((x_src, c_src), 1))
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        mu, logvar = self.mu_net(h5).view(-1, self.zdim), self.logvar_net(h5).view(-1, self.zdim)
        if return_mean_logvar:
            return mu, logvar
        else:
            z = self.reparameterize(mu, logvar)
            return z
    
class Decoder(nn.Module):
    def __init__(self, zdim, cdim, nc=1):
        super(Decoder, self).__init__()
        self.zdim, self.cdim = zdim, cdim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(self.zdim + self.cdim, nf * 16, 4, 1, 0),
                nn.BatchNorm2d(nf * 16),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf*16) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 16, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 8, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 4, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.upc5 = dcgan_upconv(nf * 2, nf)
        # state size. (nf) x 64 x 64
        self.final = nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 3, 1, 1),
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
                )

    def forward(self, z, c):
        input = torch.cat((z, c), 1)
        d1 = self.upc1(input.view(-1, self.zdim+self.cdim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        d5 = self.upc5(d4)
        output = self.final(d5)
        return output
    
class Translator(nn.Module):
    def __init__(self, zdim, cdim, nc=1):
        super(Translator, self).__init__()
        self.cdim = cdim
        conv_dim=128
        repeat_num=6
        
        layers = []
        layers.append(nn.Conv2d(nc+2*self.cdim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, nc, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, x_src, c_src, c_tar):
        h, w = np.shape(x_src)[2], np.shape(x_src)[3]
        c_src = torch.tensor(torch.unsqueeze(c_src, dim=2),
                             dtype=torch.float32).cuda()
        c_tar = torch.tensor(torch.unsqueeze(c_tar, dim=2),
                             dtype=torch.float32).cuda()
        c_src = torch.reshape(c_src @ torch.ones((np.shape(c_src)[0], 1, h*w)).cuda(),
                              (np.shape(c_src)[0], self.cdim, h, w))
        c_tar = torch.reshape(c_tar @ torch.ones((np.shape(c_tar)[0], 1, h*w)).cuda(),
                              (np.shape(c_tar)[0], self.cdim, h, w))
        input = torch.cat([x_src, c_src, c_tar], dim=1)
        return self.main(input)
    
class Discriminator_zc(nn.Module):
    def __init__(self, zdim, cdim, nc=1):
        super(Discriminator_zc, self).__init__()
        self.zdim, self.cdim = zdim, cdim
        nf = 512
        self.fc1 = nn.Sequential(
                    nn.Linear(self.zdim + self.cdim, nf),
                    nn.ReLU()
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(nf, nf),
                    nn.ReLU()
                    )
        self.fc3 = nn.Sequential(
                    nn.Linear(nf, nf),
                    nn.ReLU()
                    )
        self.fc4 = nn.Sequential(
                    nn.Linear(nf, nf),
                    nn.ReLU()
                    )
        self.fc5 = nn.Sequential(
                    nn.Linear(nf, 1),
                    nn.Sigmoid()
                    )
        
    def forward(self, z_src, c_src):
        input = torch.cat((z_src, c_src), 1)
        h1 = self.fc1(input)
        h2 = self.fc2(h1)
        h3 = self.fc3(h2)
        h4 = self.fc4(h3)
        o = self.fc5(h4)
        return o
    
class Discriminator_xcc(nn.Module):
    def __init__(self, cdim, nc=1):
        # modified from PatchGAN.
        super(Discriminator_xcc, self).__init__()
        self.cdim = cdim
        conv_dim=64
        repeat_num=5
        
        layers = []
        layers.append(nn.Conv2d(nc, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        
        self.feature_extractor = nn.Sequential(*layers)
        self.final_label_predictor = nn.Sequential(
            nn.Conv2d(curr_dim, cdim, kernel_size=2, stride=1, padding=0, bias=False),
            )
        self.final_critic = nn.Sequential(
            nn.Conv2d(curr_dim+2*cdim, 1, kernel_size=2, stride=1, padding=0, bias=False),
            )
        
    def forward(self, x_tar, c_src, c_tar, feature_extract = False):
        extracted_feature = self.feature_extractor(x_tar)
        if feature_extract:
            return extracted_feature
        else:
            pred_label = self.final_label_predictor(extracted_feature).view(-1, self.cdim)
            input_critic = torch.cat([extracted_feature, c_src, c_tar], dim=1)
            critic_value = self.final_critic(input_critic).view(-1, 1)
            return critic_value, pred_label