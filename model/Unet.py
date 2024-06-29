import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

class ConvolutionalBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvolutionalBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

class UpConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvolution, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class RecurrentBlock(nn.Module):
    def __init__(self, out_channels, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)  # Corrected typo from "x + x" to "x + out" for recurrent connection
        return out
        

class RRCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super(RRCNNBlock, self).__init__()

        self.recurrent_block1 = RecurrentBlock(out_channels, t=t)
        self.recurrent_block2 = RecurrentBlock(out_channels, t=t)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.recurrent_block1(x1)
        x3 = self.recurrent_block2(x2)
        out = x1 + x3
        return out


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()

        self.W_g_conv = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_g_batchnorm = nn.BatchNorm2d(F_int)

        self.W_x_conv = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x_batchnorm = nn.BatchNorm2d(F_int)

        self.psi_conv = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi_batchnorm = nn.BatchNorm2d(1)
        self.psi_sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # print(g.shape)
        # print(x.shape)
        g1 = self.W_g_batchnorm(self.W_g_conv(g))
        x1 = self.W_x_batchnorm(self.W_x_conv(x))
        psi = self.relu(g1 + x1)
        psi = self.psi_sigmoid(self.psi_batchnorm(self.psi_conv(psi)))
        out = x * psi
        return out


class PositionAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        b = self.conv_b(x)
        c = self.conv_c(x)
        d = self.conv_d(x)
        # # print("shape of b,c,d")
        # # print(b.shape, c.shape, d.shape)

        b = b.view(b.size(0), b.size(1), -1).permute(0,2,1)
        c = c.view(c.size(0), c.size(1), -1)
        # # print("shape of b,c")
        # # print(b.shape, c.shape)
        s = torch.bmm(b, c)
        s = torch.softmax(s, dim=-1)
        
        # # print(s.shape)

        d = d.view(d.size(0), d.size(1), -1)
        # # print(d.shape)
        y = torch.bmm(d, s)
        y = y.view(y.size(0), y.size(1), x.size(2), x.size(3))

        # # print(y.shape)

        return x + y

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionModule, self).__init__()
        self.conv_e = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_f = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_g = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        e = self.conv_e(x)
        f = self.conv_f(x)
        g = self.conv_g(x)

        # # print("shape of e,f,g")
        # # print(e.shape, f.shape,g.shape)


        e = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        f = x.view(x.size(0), x.size(1), -1)
        g = x.view(x.size(0), x.size(1), -1)
        # # print("shape of e,f,g")
        # # print(e.shape, f.shape,g.shape)


        y = torch.bmm(f,e)
        y = torch.softmax(y, dim=-1)
        # # print("shape y")
        # # print(y.shape)


        y = torch.bmm(y, g)
        y = y.view(x.size(0), x.size(1), *x.size()[2:])
        # # print("shape y")
        # # print(y.shape)

        return y + x

class MultiResolutionAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(MultiResolutionAttentionModule, self).__init__()
        self.position_attention = PositionAttentionModule(in_channels)
        self.channel_attention = ChannelAttentionModule(in_channels)

    def forward(self, x):
        position_attention = self.position_attention(x)
        channel_attention = self.channel_attention(x)
        attention_map = position_attention + channel_attention

        return attention_map

class MultiScaleConvolutionModule(nn.Module):
    def __init__(self, in_channels, out_channels,output_size):
        super(MultiScaleConvolutionModule, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_final = nn.Conv2d(out_channels * 4, out_channels, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)
        self.conv_final = nn.Conv2d(in_channels + out_channels * 4, out_channels, kernel_size=3, padding=1)


    def forward(self, x):
        x1 = self.conv1(self.pool1(x))
        x1 = self.upsample(x1)
        x2 = self.conv2(self.pool2(x))
        # # print(x2.shape)
        x2 = self.upsample(x2)
        # # print(x2.shape)
        x3 = self.conv3(self.pool3(x))
        # # print(x3.shape)
        x3 = self.upsample(x3)
        # # print(x3.shape)
        x4 = self.conv4(self.pool4(x))
        # # print(x4.shape)
        x4 = self.upsample(x4)
        # # print(x4.shape)

        x = self.upsample(x)
        
        # # print(x.shape)
        
        x = torch.cat([x,x1, x2, x3, x4], dim=1)
        x = self.conv_final(x)
        
        return x

class R2AttU_Net(nn.Module): #run1 and run2
    def __init__(self, in_channels=1, out_channels=1, t=2):
        super(R2AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4]

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.rrcnn1 = RRCNNBlock(in_channels, filters[0], t=t)
        self.rrcnn2 = RRCNNBlock(filters[0]+1, filters[1], t=t)
        self.rrcnn3 = RRCNNBlock(filters[1]+1, filters[2], t=t)

        self.up3 = UpConvolution(filters[2], filters[1])
        self.att3 = AttentionBlock(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.up_rrcnn3 = RRCNNBlock(filters[2], filters[1], t=t)

        self.up2 = UpConvolution(filters[1], filters[0])
        self.att2 = AttentionBlock(F_g=filters[0], F_l=filters[0], F_int=32)
        self.up_rrcnn2 = RRCNNBlock(filters[1], filters[0], t=t)

        self.conv = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1, padding=0)
        self.scale = [1/2,1/4]

    def forward(self, x):

        # # print(x.shape)
        e1 = self.rrcnn1(x)
        
        e2 = self.maxpool1(e1)

        x_res = F.interpolate(x, scale_factor=self.scale[0], mode='bilinear',align_corners=True)
        
        # # print(x_res.shape)
        # # print(e2.shape)
        e2 = torch.cat((x_res,e2),1)
        # # print(e2.shape)

        e2 = self.rrcnn2(e2)

        e3 = self.maxpool2(e2)

        x_res = F.interpolate(x, scale_factor=self.scale[1], mode='bilinear',align_corners=True)
        
        # # print(x_res.shape)
        # # print(e3.shape)
        e3 = torch.cat((x_res,e3),1)
        # # print(e3.shape)
        
        e3 = self.rrcnn3(e3)

        

        d3 = self.up3(e3)
        e2 = self.att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.up_rrcnn3(d3)

        d2 = self.up2(d3)
        e1 = self.att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.up_rrcnn2(d2)

        out = self.conv(d2)
        out = torch.sigmoid(out)

        return out



def build_model():
    unet = R2AttU_Net()
    return unet