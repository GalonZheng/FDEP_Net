import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal



class Conv_block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.Norm_1 = nn.InstanceNorm3d(out_channels)

    def forward(self, x_in):
        x = self.Conv_1(x_in)
        x = self.Norm_1(x)
        x_out = self.LeakyReLU(x)

        return x_out


class Flower(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Conv_1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.Norm_1 = nn.InstanceNorm3d(out_channels)

        self.flower = nn.Conv3d(in_channels, 3, kernel_size=3, stride=1, padding=1)
        self.flower.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flower.weight.shape))
        self.flower.bias = nn.Parameter(torch.zeros(self.flower.bias.shape))

    def forward(self, x_in):
        x = self.Conv_1(x_in)
        x = self.Norm_1(x)
        x = self.LeakyReLU(x)
        x = self.flower(x)

        return x

class FeatureDiffusion2(nn.Module):

    def __init__(self, channel_num: int):
        super().__init__()
        self.Conv_fix = nn.Conv3d(channel_num + 1, channel_num, kernel_size=1, stride=1)
        self.Conv_mov = nn.Conv3d(channel_num + 1, channel_num, kernel_size=1, stride=1)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Conv_in1 = nn.Conv3d(channel_num, channel_num, kernel_size=1, stride=1)
        self.Conv_in2 = nn.Conv3d(channel_num, channel_num, kernel_size=1, stride=1)
        nn.init.zeros_(self.Conv_in1.weight)
        nn.init.zeros_(self.Conv_in2.weight)

    def forward(self, x_in, fix, mov):
        x_fix = self.Conv_fix(torch.cat([x_in, mov], dim=1))
        x_mov = self.Conv_mov(torch.cat([x_in, fix], dim=1))
        x_fix = self.LeakyReLU(x_fix)
        x_mov = self.LeakyReLU(x_mov)
        w_1 = self.Conv_in1(x_fix)
        w_2 = self.Conv_in2(x_mov)

        # Softmax
        exp_1 = torch.exp(w_1)
        exp_2 = torch.exp(w_2)
        w_1 = exp_1 / (exp_1 + exp_2)
        w_2 = exp_2 / (exp_1 + exp_2)

        x_out1 = torch.add(torch.mul(x_in, w_1), x_in)
        x_out2 = torch.add(torch.mul(x_in, w_2), x_in)

        return x_out1, x_out2


class FeatureFusion2(nn.Module):

    def __init__(self, channel_num: int):
        super().__init__()
        self.Conv_in0 = nn.Conv3d(channel_num * 2 + 2, channel_num * 2, kernel_size=3, stride=1, padding=1)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Conv_in1 = nn.Conv3d(channel_num * 2, channel_num, kernel_size=1, stride=1)
        self.Conv_in2 = nn.Conv3d(channel_num * 2, channel_num, kernel_size=1, stride=1)
        nn.init.zeros_(self.Conv_in1.weight)
        nn.init.zeros_(self.Conv_in2.weight)

    def forward(self, flow1, flow2, fix, mov):
        x = torch.cat([flow1, flow2, fix, mov], dim=1)
        x = self.Conv_in0(x)
        x = self.LeakyReLU(x)
        w_1 = self.Conv_in1(x)
        w_2 = self.Conv_in2(x)

        # Softmax
        exp_1 = torch.exp(w_1)
        exp_2 = torch.exp(w_2)
        w_1 = exp_1 / (exp_1 + exp_2)
        w_2 = exp_2 / (exp_1 + exp_2)

        flow = torch.add(torch.mul(flow1, w_1), torch.mul(flow2, w_2))
        return flow



class Fusion2(nn.Module):
    def __init__(self, in_channels, block_num, channel_num):
        super(Fusion2, self).__init__()
        self.data_dim = in_channels
        self.conv1 = nn.Conv3d(block_num * self.data_dim, channel_num * 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(channel_num * 2, channel_num * 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(channel_num * 2, block_num * self.data_dim, kernel_size=3, stride=1, padding=1)
        self.act_layer = nn.LeakyReLU()

    def forward(self, x_in):
        x = self.act_layer(self.conv1(x_in))
        x = self.act_layer(self.conv2(x))
        k = self.conv3(x)
        B, _, D, H, W = x_in.shape
        flow = (x_in.view(B, -1, self.data_dim, D, H, W) * nnf.softmax(k.view(B, -1, self.data_dim, D, H, W),1)).sum(1)
        return flow

class Encoder(nn.Module):
    def __init__(self, in_channels:int, channel_num:int):
        super(Encoder, self).__init__()
        self.d_init = Conv_block(in_channels, channel_num)
        self.d_e1 = Conv_block(channel_num, channel_num)
        self.d_e2 = Conv_block(channel_num, channel_num * 2)
        self.d_e3 = Conv_block(channel_num * 2, channel_num * 4)
        self.d_e4 = Conv_block(channel_num * 4, channel_num * 8)

        self.downsample1 = nn.Conv3d(channel_num, channel_num, kernel_size=3, stride=2, padding=1)
        self.downsample2 = nn.Conv3d(channel_num, channel_num, kernel_size=3, stride=2, padding=1)
        self.downsample3 = nn.Conv3d(channel_num * 2, channel_num * 2, kernel_size=3, stride=2, padding=1)
        self.downsample4 = nn.Conv3d(channel_num * 4, channel_num * 4, kernel_size=3, stride=2, padding=1)
    def forward(self, dvol):
        x_in = self.d_init(dvol)
        x = self.downsample1(x_in)
        x_1 = self.d_e1(x)
        x = self.downsample2(x_1)
        x_2 = self.d_e2(x)
        x = self.downsample3(x_2)
        x_3 = self.d_e3(x)
        x = self.downsample4(x_3)
        x_4 = self.d_e4(x)

        return x_1, x_2, x_3, x_4


class RegHead_block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.Conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Norm = nn.InstanceNorm3d(in_channels)

        # 对mean_head和sigma_head这两个卷积操作的卷积参数指定其初始化数值
        # 可见两个卷积的参数初始的数值都指定为接近于0的正态分布，这样做的意义可能是为了使模型加快收敛，而设置为0附近的正态分布应该是经验值
        self.mean_head = nn.Conv3d(in_channels, 3, kernel_size=3, stride=1, padding=1)
        self.mean_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.mean_head.weight.shape))
        self.mean_head.bias = nn.Parameter(torch.zeros(self.mean_head.bias.shape))

        self.sigma_head = nn.Conv3d(in_channels, 3, kernel_size=3, stride=1, padding=1)
        self.sigma_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.sigma_head.weight.shape))
        self.sigma_head.bias = nn.Parameter(torch.zeros(self.sigma_head.bias.shape))

    def Reg_forward(self, x_in):

        x = self.Conv(x_in)
        x = self.Norm(x)
        x = self.LeakyReLU(x)

        SVF_mean = self.mean_head(x)
        SVF_log_sigma = self.sigma_head(x)

        return SVF_mean, SVF_log_sigma

    def forward(self, x_in):

        if self.use_checkpoint and x_in.requires_grad:
            SVF_mean, SVF_log_sigma = checkpoint.checkpoint(self.Reg_forward, x_in)
        else:
            SVF_mean, SVF_log_sigma = self.Reg_forward(x_in)

        return SVF_mean, SVF_log_sigma

class Sample_block(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, mean, log_sigma):
        noise = torch.normal(torch.zeros(mean.shape), torch.ones(mean.shape)).to(mean.device)
        x_out = mean + torch.exp(log_sigma / 2.0) * noise

        return x_out

class Integration_block(nn.Module):

    def __init__(self, int_steps=7):
        super().__init__()
        self.int_steps = int_steps

    def forward(self, SVF):
        shape = SVF.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(SVF.device)

        flow = SVF * (1 / (2.0 ** self.int_steps))
        for _ in range(self.int_steps):
            new_locs = grid + flow
            for i in range(len(shape)):
                new_locs[:, i] = 2 * (new_locs[:, i] / (shape[i] - 1) - 0.5)

            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
            flow = flow + nnf.grid_sample(flow, new_locs, align_corners=True, mode='bilinear')

        return flow

class SpatialTransformer_block(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)
        new_locs = grid + flow

        for i in range(len(shape)):
            new_locs[:, i] = 2 * (new_locs[:, i] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class YPath(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channel_num: int,
                 out_channels: int):
        super().__init__()

        self.encoder = Encoder(in_channels, channel_num)

        self.t_conv4 = Conv_block(channel_num * 8, channel_num * 4)    # 128
        self.t_conv3 = Conv_block(channel_num * 4, channel_num * 2)    # 64
        self.t_conv2 = Conv_block(channel_num * 2, channel_num * 1)    # 32
        self.t_conv1 = Conv_block(channel_num, channel_num)    # 16

        self.skip_m_conv3 = Conv_block(channel_num * 4 + channel_num * 4, channel_num * 4)
        self.skip_m_conv2 = Conv_block(channel_num * 2 + channel_num * 2, channel_num * 2)
        self.skip_m_conv1 = Conv_block(channel_num + channel_num, channel_num)

        self.skip_f_conv3 = Conv_block(channel_num * 4 + channel_num * 4, channel_num * 4)
        self.skip_f_conv2 = Conv_block(channel_num * 2 + channel_num * 2, channel_num * 2)
        self.skip_f_conv1 = Conv_block(channel_num + channel_num, channel_num)

        self.fu4 = FeatureFusion2(channel_num * 8)
        self.fu3 = FeatureFusion2(channel_num * 4)
        self.fu2 = FeatureFusion2(channel_num * 2)
        self.fu1 = FeatureFusion2(channel_num * 1)

        self.diffu3 = FeatureDiffusion2(channel_num * 4)
        self.diffu2 = FeatureDiffusion2(channel_num * 2)
        self.diffu1 = FeatureDiffusion2(channel_num * 1)

        self.ff3 = Fusion2(channel_num * 4, 2, channel_num * 4)
        self.ff2 = Fusion2(channel_num * 2, 2, channel_num * 2)
        self.ff1 = Fusion2(channel_num, 2, channel_num)

        self.flower3 = Conv_block(channel_num * 4, 3)
        self.flower2 = Conv_block(channel_num * 2, 3)
        self.flower1 = Conv_block(channel_num * 1, 3)
        self.flower0 = Conv_block(channel_num, out_channels)
        self.stn = SpatialTransformer_block()

        self.t_upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)


    def dataprocess(self, image):
        downsample = nn.MaxPool3d(2, stride=2)
        pool1 = downsample(image)
        pool2 = downsample(pool1)
        pool3 = downsample(pool2)
        pool4 = downsample(pool3)

        return [pool4, pool3, pool2, pool1]

    def forward(self, mov, fix):
            pool_mov = self.dataprocess(mov)
            pool_fix = self.dataprocess(fix)

            f1, f2, f3, f4 = self.encoder(fix)
            m1, m2, m3, m4 = self.encoder(mov)

            fus4 = self.fu4(f4, m4, pool_fix[0], pool_mov[0])
            x = self.t_upsample(fus4)
            x = self.t_conv4(x)
            x1, x2 = self.diffu3(x, pool_fix[1], pool_mov[1])
            f3 = torch.cat([f3, x1], dim=1)
            f3 = self.skip_f_conv3(f3)
            m3 = torch.cat([m3, x2], dim=1)
            m3 = self.skip_m_conv3(m3)
            flow = self.flower3(x)
            m3 = self.stn(m3, flow)

            fus3 = self.fu3(f3, m3, pool_fix[1], pool_mov[1])
            fus3 = self.ff3(torch.cat([fus3, x], dim=1))
            # fus3 = self.ff3(fus3, x)
            x = self.t_upsample(fus3)
            x = self.t_conv3(x)
            x1, x2 = self.diffu2(x, pool_fix[2], pool_mov[2])
            f2 = torch.cat([f2, x1], dim=1)
            f2 = self.skip_f_conv2(f2)
            m2 = torch.cat([m2, x2], dim=1)
            m2 = self.skip_m_conv2(m2)
            flow = self.flower2(x)
            m2 = self.stn(m2, flow)

            fus2 = self.fu2(f2, m2, pool_fix[2], pool_mov[2])
            fus2 = self.ff2(torch.cat([fus2, x], dim=1))
            # fus2 = self.ff2(fus2, x)
            x = self.t_upsample(fus2)
            x = self.t_conv2(x)
            x1, x2 = self.diffu1(x, pool_fix[3], pool_mov[3])
            f1 = torch.cat([f1, x1], dim=1)
            f1 = self.skip_f_conv1(f1)
            m1 = torch.cat([m1, x2], dim=1)
            m1 = self.skip_m_conv1(m1)
            flow = self.flower1(x)
            m1 = self.stn(m1, flow)

            fus1 = self.fu1(f1, m1, pool_fix[3], pool_mov[3])
            fus1 = self.ff1(torch.cat([fus1, x], dim=1))
            x = self.t_upsample(fus1)
            x = self.t_conv1(x)
            x = self.flower0(x)

            return x


class FDNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 channel_num: int = 16):
        super().__init__()
        self.net = YPath(in_channels, channel_num, channel_num)
        self.reghead = RegHead_block(channel_num)
        self.Sample = Sample_block()
        self.Integration = Integration_block(int_steps=7)
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, inputs):

        flow = self.net(inputs[0], inputs[1])

        SVF_mean, SVF_log_sigma = self.reghead(flow)
        SVF = self.Sample(SVF_mean, SVF_log_sigma)
        flow = self.Integration(SVF)
        warp_mov = self.SpatialTransformer(inputs[0], flow)
        SVF_params = torch.cat([SVF_mean, SVF_log_sigma], dim=1)

        return warp_mov, flow, SVF_params
