import torch


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

if __name__ == '__main__':

    x =torch.randn((1,144,160,160))
    a,_ = dwt_init(x)
    print(a.size())
    print(iwt_init(a).size())
