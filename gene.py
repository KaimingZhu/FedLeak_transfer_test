import functools
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import layer_temp


def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[512] = {'in_channels': [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
                 'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1, 1]],
                 'upsample': [True] * 7,
                 'resolution': [8, 16, 32, 64, 128, 256, 512],
                 'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                               for i in range(3, 10)}}
    arch[256] = {'in_channels': [ch * item for item in [16, 16, 8, 8, 4, 2]],
                 'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1]],
                 'upsample': [True] * 6,
                 'resolution': [8, 16, 32, 64, 128, 256],
                 'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                               for i in range(3, 9)}}
    arch[128] = {'in_channels': [ch * item for item in [16, 16, 8, 4, 2]],
                 'out_channels': [ch * item for item in [16, 8, 4, 2, 1]],
                 'upsample': [True] * 5,
                 'resolution': [8, 16, 32, 64, 128],
                 'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                               for i in range(3, 8)}}
    arch[64] = {'in_channels': [ch * item for item in [16, 16, 8, 4]],
                'out_channels': [ch * item for item in [16, 8, 4, 2]],
                'upsample': [True] * 4,
                'resolution': [8, 16, 32, 64],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                              for i in range(3, 7)}}
    arch[32] = {'in_channels': [ch * item for item in [4, 4, 4]],
                'out_channels': [ch * item for item in [4, 4, 4]],
                'upsample': [True] * 3,
                'resolution': [8, 16, 32],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                              for i in range(3, 6)}}

    return arch


class Generator(nn.Module):
    def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128,
                 G_kernel_size=3, G_attn='64', n_classes=1000,
                 num_G_SVs=1, num_G_SV_itrs=1,
                 G_shared=True, shared_dim=0, hier=False,
                 cross_replica=False, mybn=False,
                 G_activation=nn.ReLU(inplace=False),
                 G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
                 BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
                 G_init='ortho', skip_init=False, no_optim=False,
                 G_param='SN', norm_style='bn',
                 **kwargs):
        super(Generator, self).__init__()
        self.ch = G_ch
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.resolution = resolution
        self.kernel_size = G_kernel_size
        self.attention = G_attn
        self.n_classes = n_classes
        self.G_shared = G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else dim_z
        self.hier = hier
        self.cross_replica = cross_replica
        self.mybn = mybn
        self.activation = G_activation
        self.init = G_init
        self.G_param = G_param
        self.norm_style = norm_style
        self.BN_eps = BN_eps
        self.SN_eps = SN_eps
        self.fp16 = G_fp16
        self.arch = G_arch(self.ch, self.attention)[resolution]

        if self.hier:
            self.num_slots = len(self.arch['in_channels']) + 1
            self.z_chunk_size = (self.dim_z // self.num_slots)
            self.dim_z = self.z_chunk_size * self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0

        if self.G_param == 'SN':
            self.which_conv = functools.partial(layer_temp.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layer_temp.SNLinear,
                                                  num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                  eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear


        self.which_embedding = nn.Embedding
        bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                     else self.which_embedding)
        self.which_bn = functools.partial(layer_temp.ccbn,
                                          which_linear=bn_linear,
                                          cross_replica=self.cross_replica,
                                          mybn=self.mybn,
                                          input_size=(self.shared_dim + self.z_chunk_size if self.G_shared
                                                      else self.n_classes),
                                          norm_style=self.norm_style,
                                          eps=self.BN_eps)


        self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared
                       else layer_temp.identity())
        self.linear = self.which_linear(self.dim_z // self.num_slots,
                                        self.arch['in_channels'][0] * (self.bottom_width ** 2))

        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layer_temp.GBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv=self.which_conv,
                                           which_bn=self.which_bn,
                                           activation=self.activation,
                                           upsample=(functools.partial(F.interpolate, scale_factor=2)
                                                     if self.arch['upsample'][index] else None))]]

            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layer_temp.Attention(self.arch['out_channels'][index], self.which_conv)]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.output_layer = nn.Sequential(layer_temp.bn(self.arch['out_channels'][-1],
                                                    cross_replica=self.cross_replica,
                                                    mybn=self.mybn),
                                          self.activation,
                                          self.which_conv(self.arch['out_channels'][-1], 3))

        if not skip_init:
            self.init_weights()

        if no_optim:
            return
        self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
        if G_mixed_precision:
            import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                      betas=(self.B1, self.B2), weight_decay=0,
                                      eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0,
                                    eps=self.adam_eps)


    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    pass
                self.param_count += sum([p.data.nelement() for p in module.parameters()])


    def forward(self, z, y):
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)

        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h, ys[index])

        return torch.tanh(self.output_layer(h))

