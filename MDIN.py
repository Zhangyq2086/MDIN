import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from mamba_ssm.modules.mamba_simple import Mamba

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def conv2d(x, device):
    b, c, h, w = x.shape
    conv = nn.Conv2d(c, c // 2, kernel_size=1).to(device)
    return conv(x)

def dwt_init(x):
    # 输入检查
    assert x.dim() == 4, "Input tensor must be 4-dimensional"
    assert x.size(2) % 2 == 0 and x.size(3) % 2 == 0, "Input height and width must be even"

    # 分解操作
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    # 四个子带
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    # 返回连接的结果和各个子带
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1), (x_LL, x_HL, x_LH, x_HH)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()

    # 输入形状检查
    assert in_channel % 4 == 0, "Input channel must be a multiple of 4"

    out_batch = in_batch
    out_channel = in_channel // 4
    out_height = r * in_height
    out_width = r * in_width

    # 提取子带
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    # 初始化输出张量
    h = torch.zeros([out_batch, out_channel, out_height, out_width], device=x.device)

    # 进行逆变换
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        res = self.process(x)
        y = self.avg_pool(res)
        z = self.conv_du(y)
        return z * res + x
class Refine(nn.Module):

    def __init__(self, n_feat, out_channel):
        super(Refine, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
            ChannelAttention(n_feat, 4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
# ---------------------------------------------------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)


class PatchUnEmbed(nn.Module):#
    def __init__(self,basefilter) -> None:
        super().__init__()
        self.nc = basefilter
    def forward(self, x,x_size):
        B,HW,C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x
class PatchEmbed(nn.Module):#
    """ 2D Image to Patch Embedding
    """
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')

    def forward(self, x):
        #（b,c,h,w)->(b,c*s*p,h//s,w//s)
        #(b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x
class SingleMambaBlock(nn.Module): #
    def __init__(self, dim):
        super(SingleMambaBlock, self).__init__()
        self.encoder = Mamba(dim,bimamba_type=None)
        self.norm = LayerNorm(dim,'with_bias')
        # self.PatchEmbe=PatchEmbed(patch_size=4, stride=4,in_chans=dim, embed_dim=dim*16)
    def forward(self,ipt):
        x,residual = ipt
        residual = x+residual
        x = self.norm(residual)
        return (self.encoder(x),residual)

class CrossMamba(nn.Module):#
    def __init__(self, dim):
        super(CrossMamba, self).__init__()
        self.cross_mamba = Mamba(dim,bimamba_type="v3")
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
    def forward(self,ms,ms_resi,pan):
        ms_resi = ms+ms_resi
        ms = self.norm1(ms_resi)
        pan = self.norm2(pan)
        global_f = self.cross_mamba(self.norm1(ms),extra_emb1=pan) ##  extra_emb=self.norm2(pan),
        B,HW,C = global_f.shape
        ms = global_f.transpose(1, 2).view(B, C, int(HW ** 0.5), int(HW ** 0.5)) ## 128*8, 128*8
        ms = (self.dwconv(ms)+ms).flatten(2).transpose(1, 2)
        return ms,ms_resi
class HinResBlock(nn.Module): #
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x+resi
class TokenSwapMamba(nn.Module): #
    def __init__(self, dim):
        super(TokenSwapMamba, self).__init__()
        self.msencoder = Mamba(dim,bimamba_type=None)
        self.panencoder = Mamba(dim,bimamba_type=None)
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
    def forward(self, ms,pan
                ,ms_residual,pan_residual):
        # ms (B,N,C)
        #pan (B,N,C)
        ms_residual = ms+ms_residual
        pan_residual = pan+pan_residual
        ms = self.norm1(ms_residual)
        pan = self.norm2(pan_residual)
        B,N,C = ms.shape
        ms_first_half = ms[:, :, :C//2]
        pan_first_half = pan[:, :, :C//2]
        ms_swap= torch.cat([pan_first_half,ms[:,:,C//2:]],dim=2)
        pan_swap= torch.cat([ms_first_half,pan[:,:,C//2:]],dim=2)
        ms_swap = self.msencoder(ms_swap)
        pan_swap = self.panencoder(pan_swap)
        return ms_swap,pan_swap,ms_residual,pan_residual

def DWTwork(pan_input):
    # 对全色输入进行小波变换
    pan_dwt, (pan_LL, pan_HL, pan_LH, pan_HH) = dwt_init(pan_input)

    # 继续处理，pan_dwt 现在是一个张量
    C = pan_dwt.shape[1]  # 获取通道数

    # 进行后续操作，例如提取低频分量
    pan_LL = pan_dwt[:, :C // 4, :, :]  # 确保 pan_LL 是从 pan_dwt 中提取的张量
    # 在这里可以进行其他操作，例如传递到其他层
    pan_HL = pan_dwt[:, C // 4:2 * C // 4, :, :]  # HL分量
    pan_LH = pan_dwt[:, 2 * C // 4:3 * C // 4, :, :]  # LH分量
    # 返回最终结果
    pan_HH = pan_dwt[:, C - C//4:, :, :] #提取HH
    return pan_LL,pan_HH,pan_HL,pan_LH
#-----------------------------------------------------------------
def Test(ms, pan):
    ms_bic = F.interpolate(ms, scale_factor=4)
    # print(ms_bic.shape)
    ms_LL, ms_HH, ms_HL, ms_LH = DWTwork(ms_bic)
    # print(ms_LL.shape)
    pan_LL, pan_HH, pan_HL, pan_LH = DWTwork(pan)
    hrms = torch.concat((ms_LL, ms_HL, ms_LH, ms_HH), dim=1)
    hrms = iwt_init(hrms)
    return hrms
#------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, num_channels=None, base_filter=None, args=None):
        super(Net, self).__init__()
        base_filter = 32
        self.base_filter = base_filter
        self.stride = 1
        self.patch_size = 1
        self.pan_encoder = nn.Sequential(nn.Conv2d(31, base_filter, 3, 1, 1), HinResBlock(base_filter, base_filter), ## 之前是1-->4
                                         HinResBlock(base_filter, base_filter), HinResBlock(base_filter, base_filter))
        self.hs_encoder = nn.Sequential(nn.Conv2d(32, base_filter, 3, 1, 1), HinResBlock(base_filter, base_filter),  ## 输入通道数
                                        HinResBlock(base_filter, base_filter), HinResBlock(base_filter, base_filter))
        self.pan_encoder1 = nn.Sequential(nn.Conv2d(1, base_filter, 3, 1, 1), HinResBlock(base_filter, base_filter), ## 之前是1-->4
                                         HinResBlock(base_filter, base_filter), HinResBlock(base_filter, base_filter))
        self.hs_encoder1 = nn.Sequential(nn.Conv2d(31, base_filter, 3, 1, 1), HinResBlock(base_filter, base_filter),  ## 输入通道数
                                         HinResBlock(base_filter, base_filter), HinResBlock(base_filter, base_filter))
        self.embed_dim = base_filter * self.stride * self.patch_size
        self.shallow_fusion1 = nn.Conv2d(base_filter * 2, base_filter, 3, 1, 1)
        self.shallow_fusion2 = nn.Conv2d(base_filter * 2, base_filter, 3, 1, 1)
        self.hs_to_token = PatchEmbed(in_chans=base_filter, embed_dim=self.embed_dim, patch_size=self.patch_size,
                                      stride=self.stride)
        self.pan_to_token = PatchEmbed(in_chans=base_filter, embed_dim=self.embed_dim, patch_size=self.patch_size,
                                       stride=self.stride)
        self.deep_fusion1 = CrossMamba(self.embed_dim)
        self.deep_fusion2 = CrossMamba(self.embed_dim)
        self.deep_fusion3 = CrossMamba(self.embed_dim)
        self.deep_fusion4 = CrossMamba(self.embed_dim)
        self.deep_fusion5 = CrossMamba(self.embed_dim)

        self.pan_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(8)])
        self.ms_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(8)])
        self.patchunembe = PatchUnEmbed(base_filter)
        self.output = Refine(base_filter, 31)  # # 输出通道

    def forward(self, hs, pan):

        hs_bic = F.interpolate(hs, scale_factor=4)
        hs_LL, hs_HH, hs_HL,hs_LH = DWTwork(hs_bic)
        pan_LL, pan_HH, pan_HL, pan_LH = DWTwork(pan)
        fusion_LL = torch.cat((hs_LL, pan_LL), dim=1)
        hs_f = self.hs_encoder(fusion_LL)
        pan_f = self.pan_encoder(hs_HH)
        b, c, h, w = hs_f.shape
        hs_f = self.hs_to_token(hs_f)
        pan_f = self.pan_to_token(pan_f)

        residual_hs_f = 0
        hs_f, residual_hs_f = self.deep_fusion1(pan_f, residual_hs_f, hs_f)
        hs_f, residual_hs_f = self.deep_fusion2(pan_f, residual_hs_f, hs_f)
        hs_f, residual_hs_f = self.deep_fusion3(pan_f, residual_hs_f, hs_f)
        hs_f, residual_hs_f = self.deep_fusion4(pan_f, residual_hs_f, hs_f)
        hs_f, residual_hs_f = self.deep_fusion5(pan_f, residual_hs_f, hs_f)

        hs_f = self.patchunembe(hs_f, (h, w))
        hs_HH = self.output(hs_f)
        hrhs = torch.concat((hs_LL, hs_HL, hs_LH, hs_HH), dim=1)
        hrhs = iwt_init(hrhs)
#-------------------------------------------------------------------------------------------
        hs_f1 = self.hs_encoder1(hrhs)
        b,c,h,w = hs_f1.shape
        pan_f1 = self.pan_encoder1(pan)
        hs_f1 = self.hs_to_token(hs_f1)
        pan_f1 = self.pan_to_token(pan_f1)

        residual_hs_f1 = 0
        residual_pan_f1 = 0
        hs_f1, residual_hs_f1 = self.ms_feature_extraction([hs_f1, residual_hs_f1])
        pan_f1, residual_pan_f1 = self.pan_feature_extraction([pan_f1, residual_pan_f1])

        # # # Mamba*8
        hs_f1 = self.patchunembe(hs_f1, (h, w))
        pan_f1 = self.patchunembe(pan_f1, (h, w))
        hs_f1 = self.shallow_fusion1(torch.concat([hs_f1, pan_f1], dim=1)) + hs_f1
        pan_f1 = self.shallow_fusion2(torch.concat([pan_f1, hs_f1], dim=1)) + pan_f1
        hs_f1 = self.hs_to_token(hs_f1)
        pan_f1 = self.pan_to_token(pan_f1)
        residual_hs_f1 = 0
        hs_f1, residual_hs_f1 = self.deep_fusion1(hs_f1, residual_hs_f1, pan_f1)
        hs_f1, residual_hs_f1 = self.deep_fusion2(hs_f1, residual_hs_f1, pan_f1)
        hs_f1, residual_hs_f1 = self.deep_fusion3(hs_f1, residual_hs_f1, pan_f1)
        hs_f1, residual_hs_f = self.deep_fusion4(hs_f1, residual_hs_f1, pan_f1)
        hs_f1, residual_hs_f = self.deep_fusion5(hs_f1, residual_hs_f1, pan_f1)
        hs_f1 = self.patchunembe(hs_f1, (h, w))
        hrhs = self.output(hs_f1) + hrhs
        return hrhs

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on:", device)
    # 创建示例输入张量，代表多光谱和全色图像
    hs_input = torch.rand(1, 31, 32, 32).to(device)  # 高光谱输入示例形状
    pan_input = torch.rand(1, 1, 128, 128).to(device)  # 全色输入示例形状
    residual_ms_f = 0
    # 实例化网络
    net = Net().to(device)
    # 前向传播
    output = net(hs_input,pan_input)  #
    # 输出将是高分辨率融合图像
    print(output.shape)