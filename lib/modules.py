import numbers
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels=3, kernel_size=3, sigma=1.0, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.kernel_size = kernel_size[0]
        self.dim = dim
        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        pad_size = self.kernel_size//2
        if self.dim == 1:
            pad = F.pad(input, (pad_size, pad_size), mode='reflect')
            # pad = F.pad(input, (pad_size, pad_size, pad_size), mode='reflect')
        elif self.dim == 2:
            pad = F.pad(input, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        elif self.dim == 3:
            # pad = F.pad(input, (pad_size, pad_size, pad_size, pad_size, pad_size), mode='reflect')
            pad = F.pad(input, (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size), mode='reflect')

        return self.conv(pad, weight=self.weight.type_as(input), groups=self.groups)


class CBatchNorm2d(nn.Module):
    ''' Conditional batch normalization layer class.
        Borrowed from Occupancy Network repo: https://github.com/autonomousvision/occupancy_networks
    Args:
        c_dim (int): dimension of latent conditioned code c
        f_channels (int): number of channels of the feature maps
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_channels, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_channels = f_channels
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_channels, 1) # match the cond dim to num of feature channels
        self.conv_beta = nn.Conv1d(c_dim, f_channels, 1)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm2d(f_channels, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm2d(f_channels, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm2d(f_channels, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert(x.size(0) == c.size(0))
        assert(c.size(1) == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x 1 (conv1d needs the 3rd dim)
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c).unsqueeze(-1) # make gamma be of shape [batch, f_dim, 1, 1]
        beta = self.conv_beta(c).unsqueeze(-1)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class Conv2DBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=4, stride=2, padding=1, use_bias=False, use_bn=True, use_relu=True):
        super(Conv2DBlock, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        if use_bn:
            self.bn = nn.BatchNorm2d(output_nc, affine=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.use_relu:
            x = self.relu(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class UpConv2DBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=4, stride=2, padding=1,
                 use_bias=False, use_bn=True, up_mode='upconv', use_dropout=False):
        super(UpConv2DBlock, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.relu = nn.ReLU()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride,
                                           padding=padding, bias=use_bias)
        else:
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1, stride=1),
            )
        if use_bn:
            self.bn = nn.BatchNorm2d(output_nc, affine=False)
        if use_dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, x, skip_input=None):
        x = self.relu(x)
        x = self.up(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_dropout:
            x = self.drop(x)
        if skip_input is not None:
            x = torch.cat([x, skip_input], 1)
        return x


class GeomConvLayers(nn.Module):
    '''
    A few convolutional layers to smooth the geometric feature tensor
    '''
    def __init__(self, input_nc=16, hidden_nc=16, output_nc=16, use_relu=False):
        super().__init__()
        self.use_relu = use_relu

        self.conv1 = nn.Conv2d(input_nc, hidden_nc, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv2d(hidden_nc, hidden_nc, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(hidden_nc, output_nc, kernel_size=5, stride=1, padding=2, bias=False)
        if use_relu:
            self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.conv2(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.conv3(x)

        return x


class GeomConvBottleneckLayers(nn.Module):
    '''
    A u-net-like small bottleneck network for smoothing the geometric feature tensor
    '''
    def __init__(self, input_nc=16, hidden_nc=16, output_nc=16, use_relu=False):
        super().__init__()
        self.use_relu = use_relu

        self.conv1 = nn.Conv2d(input_nc, hidden_nc, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(hidden_nc, hidden_nc*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(hidden_nc*2, hidden_nc*4, kernel_size=4, stride=2, padding=1, bias=False)

        self.up1 = nn.ConvTranspose2d(hidden_nc*4, hidden_nc*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.up2 = nn.ConvTranspose2d(hidden_nc*2, hidden_nc, kernel_size=4, stride=2, padding=1, bias=False)
        self.up3 = nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=False)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        return x


class GaussianSmoothingLayers(nn.Module):
    '''
    use a fixed, not-trainable gaussian smoother layers for smoothing the geometric feature tensor
    '''
    def __init__(self, channels=16, kernel_size=5, sigma=1.0):
        super().__init__()
        self.conv1 = GaussianSmoothing(channels, kernel_size=kernel_size, sigma=1.0, dim=2)
        self.conv2 = GaussianSmoothing(channels, kernel_size=kernel_size, sigma=1.0, dim=2)
        self.conv3 = GaussianSmoothing(channels, kernel_size=kernel_size, sigma=1.0, dim=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class UnetNoCond5DS(nn.Module):
    # 5DS: downsample 5 times, for posmap size=32
    def __init__(self, input_nc=3, output_nc=3, nf=64, up_mode='upconv', use_dropout=False, 
                return_lowres=False, return_2branches=False):
        super().__init__()
        assert up_mode in ('upconv', 'upsample')

        self.return_lowres = return_lowres
        self.return_2branches = return_2branches

        self.conv1 = Conv2DBlock(input_nc, nf, 4, 2, 1, use_bias=False, use_bn=False, use_relu=False)
        self.conv2 = Conv2DBlock(1 * nf, 2 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv3 = Conv2DBlock(2 * nf, 4 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv4 = Conv2DBlock(4 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv5 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=False)

        self.upconv1 = UpConv2DBlock(8 * nf, 8 * nf, 4, 2, 1, up_mode=up_mode) #2x2, 512
        self.upconv2 = UpConv2DBlock(8 * nf * 2, 4 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 4x4, 512
        self.upconv3 = UpConv2DBlock(4 * nf * 2, 2 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512

        # Coord regressor
        self.upconv4 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode=up_mode) # 16
        self.upconv5 = UpConv2DBlock(1 * nf * 2, output_nc, 4, 2, 1, use_bn=False, use_bias=True, up_mode=up_mode) # 32

        if return_2branches:
            self.upconvN4 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode=up_mode) # 16
            self.upconvN5 = UpConv2DBlock(1 * nf * 2, output_nc, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upconv') # 32

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)

        u1 = self.upconv1(d5, d4)
        u2 = self.upconv2(u1, d3)
        u3 = self.upconv3(u2, d2)

        u4 = self.upconv4(u3, d1)
        u5 = self.upconv5(u4)

        if self.return_2branches:
            uN4 = self.upconvN4(u3, d1)
            uN5 = self.upconvN5(uN4)
            return u5, uN5

        return u5


class UnetNoCond6DS(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, nf=64, up_mode='upconv', use_dropout=False, return_lowres=False, return_2branches=False):
        super(UnetNoCond6DS, self).__init__()
        assert up_mode in ('upconv', 'upsample')

        self.return_lowres = return_lowres
        self.return_2branches = return_2branches

        self.conv1 = Conv2DBlock(input_nc, nf, 4, 2, 1, use_bias=False, use_bn=False, use_relu=False)
        self.conv2 = Conv2DBlock(1 * nf, 2 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv3 = Conv2DBlock(2 * nf, 4 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv4 = Conv2DBlock(4 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv5 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv6 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=False)

        self.upconv1 = UpConv2DBlock(8 * nf, 8 * nf, 4, 2, 1, up_mode=up_mode) #2x2, 512
        self.upconv2 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 4x4, 512
        self.upconv3 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512
        self.upconv4 = UpConv2DBlock(4 * nf * 3, 4 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512

        # Coord regressor
        self.upconvC5 = UpConv2DBlock(2 * nf * 3, 2 * nf, 4, 2, 1, up_mode='upsample') # 16
        self.upconvC6 = UpConv2DBlock(1 * nf * 3, output_nc, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upsample') # 64x64, 128

        if return_2branches:
            # Normal regressor
            self.upconvN5 = UpConv2DBlock(2 * nf * 3, 2 * nf, 4, 2, 1, up_mode='upconv') # 32x32, 256
            self.upconvN6 = UpConv2DBlock(1 * nf * 3, 3, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upconv') # 64x64, 128

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)
        d6 = self.conv6(d5)

        # shared decoder layers
        u1 = self.upconv1(d6, d5)
        u2 = self.upconv2(u1, d4)
        u3 = self.upconv3(u2, d3)
        u4 = self.upconv4(u3, d2)

        # coord regressor
        uc5 = self.upconvC5(u4, d1)
        uc6 = self.upconvC6(uc5)

        if self.return_2branches:
            # normal regressor
            un5 = self.upconvN5(u4, d1)
            un6 = self.upconvN6(un5)

            return uc6, un6

        return uc6


class UnetNoCond7DS(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, nf=64, up_mode='upconv', use_dropout=False, return_lowres=False, return_2branches=False):
        super(UnetNoCond7DS, self).__init__()
        assert up_mode in ('upconv', 'upsample')

        self.return_lowres = return_lowres
        self.return_2branches = return_2branches

        self.conv1 = Conv2DBlock(input_nc, nf, 4, 2, 1, use_bias=False, use_bn=False, use_relu=False)
        self.conv2 = Conv2DBlock(1 * nf, 2 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv3 = Conv2DBlock(2 * nf, 4 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv4 = Conv2DBlock(4 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv5 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv6 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv7 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=False)

        self.upconv1 = UpConv2DBlock(8 * nf, 8 * nf, 4, 2, 1, up_mode=up_mode) #2x2, 512
        self.upconv2 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 4x4, 512
        self.upconv3 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512
        self.upconv4 = UpConv2DBlock(8 * nf * 2, 4 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512

        # Coord regressor
        self.upconvC5 = UpConv2DBlock(4 * nf * 2, 2 * nf, 4, 2, 1, up_mode='upsample') # 16
        self.upconvC6 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode='upsample') # 32
        self.upconvC7 = UpConv2DBlock(1 * nf * 2, output_nc, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upsample') # 64x64, 128

        if return_2branches:
            # Normal regressor
            self.upconvN5 = UpConv2DBlock(4 * nf * 2, 2 * nf, 4, 2, 1, up_mode='upconv') # 32x32, 256
            self.upconvN6 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode='upconv') # 64x64, 128
            self.upconvN7 = UpConv2DBlock(1 * nf * 2, 3, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upconv') # 64x64, 128

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)
        d6 = self.conv6(d5)
        d7 = self.conv7(d6)

        # shared decoder layers
        u1 = self.upconv1(d7, d6)
        u2 = self.upconv2(u1, d5)
        u3 = self.upconv3(u2, d4)
        u4 = self.upconv4(u3, d3)

        # coord regressor
        uc5 = self.upconvC5(u4, d2)
        uc6 = self.upconvC6(uc5, d1)
        uc7 = self.upconvC7(uc6)

        if self.return_2branches:
            # normal regressor
            un5 = self.upconvN5(u4, d2)
            un6 = self.upconvN6(un5, d1)
            un7 = self.upconvN7(un6)

            return uc7, un7

        return uc7


class ShapeDecoder(nn.Module):
    '''
    The "Shape Decoder" in the POP paper Fig. 2. The same as the "shared MLP" in the SCALE paper.
    - with skip connection from the input features to the 4th layer's output features (like DeepSDF)
    - branches out at the second-to-last layer, one branch for position pred, one for normal pred
    '''
    def __init__(self, in_size, hsize = 256, actv_fn='softplus'):
        self.hsize = hsize
        super(ShapeDecoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_size, self.hsize, 1)
        self.conv2 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv3 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv4 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv5 = torch.nn.Conv1d(self.hsize+in_size, self.hsize, 1)
        self.conv6 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8 = torch.nn.Conv1d(self.hsize, 3, 1)

        self.conv6N = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7N = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8N = torch.nn.Conv1d(self.hsize, 3, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.hsize)
        self.bn2 = torch.nn.BatchNorm1d(self.hsize)
        self.bn3 = torch.nn.BatchNorm1d(self.hsize)
        self.bn4 = torch.nn.BatchNorm1d(self.hsize)

        self.bn5 = torch.nn.BatchNorm1d(self.hsize)
        self.bn6 = torch.nn.BatchNorm1d(self.hsize)
        self.bn7 = torch.nn.BatchNorm1d(self.hsize)

        self.bn6N = torch.nn.BatchNorm1d(self.hsize)
        self.bn7N = torch.nn.BatchNorm1d(self.hsize)

        self.actv_fn = nn.ReLU() if actv_fn=='relu' else nn.Softplus()

    def forward(self, x):
        x1 = self.actv_fn(self.bn1(self.conv1(x)))
        x2 = self.actv_fn(self.bn2(self.conv2(x1)))
        x3 = self.actv_fn(self.bn3(self.conv3(x2)))
        x4 = self.actv_fn(self.bn4(self.conv4(x3)))
        x5 = self.actv_fn(self.bn5(self.conv5(torch.cat([x,x4],dim=1))))

        # position pred
        x6 = self.actv_fn(self.bn6(self.conv6(x5)))
        x7 = self.actv_fn(self.bn7(self.conv7(x6)))
        x8 = self.conv8(x7)

        # normals pred
        xN6 = self.actv_fn(self.bn6N(self.conv6N(x5)))
        xN7 = self.actv_fn(self.bn7N(self.conv7N(xN6)))
        xN8 = self.conv8N(xN7)

        return x8, xN8

