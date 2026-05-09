import os
import torch
import torch.nn as nn
# import torch.nn.functional as F  # Removed since F.interpolate is replaced with ConvTranspose2d
from torchvision.models.utils import load_state_dict_from_url
import matplotlib.pyplot as plt

__all__ = ['hrnet18_expanded', 'hrnet32_expanded', 'hrnet48_expanded', 'hrnet64_expanded']


model_urls = {
    'hrnet18_imagenet': 'https://opr0mq.dm.files.1drv.com/y4mIoWpP2n-LUohHHANpC0jrOixm1FZgO2OsUtP2DwIozH5RsoYVyv_De5wDgR6XuQmirMV3C0AljLeB-zQXevfLlnQpcNeJlT9Q8LwNYDwh3TsECkMTWXCUn3vDGJWpCxQcQWKONr5VQWO1hLEKPeJbbSZ6tgbWwJHgHF7592HY7ilmGe39o5BhHz7P9QqMYLBts6V7QGoaKrr0PL3wvvR4w',
    'hrnet32_imagenet': 'https://opr74a.dm.files.1drv.com/y4mKOuRSNGQQlp6wm_a9bF-UEQwp6a10xFCLhm4bqjDu6aSNW9yhDRM7qyx0vK0WTh42gEaniUVm3h7pg0H-W0yJff5qQtoAX7Zze4vOsqjoIthp-FW3nlfMD0-gcJi8IiVrMWqVOw2N3MbCud6uQQrTaEAvAdNjtjMpym1JghN-F060rSQKmgtq5R-wJe185IyW4-_c5_ItbhYpCyLxdqdEQ',
    'hrnet48_imagenet': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ',
    'hrnet48_cityscapes': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ',
}


class BasicBlockExpanded(nn.Module):
    """
    Expanded BasicBlock with explicitly named layers.
    Each conv, bn, and relu is individually defined.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockExpanded, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # First convolution block
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolution block
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes)

        # Downsample and final relu
        self.downsample = downsample
        self.stride = stride
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # First block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # Second block
        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class BottleneckExpanded(nn.Module):
    """
    Expanded Bottleneck with explicitly named layers.
    Each conv, bn, and relu is individually defined.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BottleneckExpanded, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # First convolution block
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolution block
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.relu2 = nn.ReLU(inplace=True)

        # Third convolution block
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        # Downsample and final relu
        self.downsample = downsample
        self.stride = stride
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # First block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # Second block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # Third block
        out = self.conv3(out)
        out = self.bn3(out)

        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


class HighResolutionModuleExpanded(nn.Module):
    """
    Expanded HighResolutionModule with all branches and fuse layers explicitly defined.
    For HRNet18: stage2 has 2 branches, stage3 has 3 branches, stage4 has 4 branches.
    """

    def __init__(self, num_branches, block_type, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, norm_layer=None):
        super(HighResolutionModuleExpanded, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.num_inchannels = list(num_inchannels)
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        # Define branches explicitly based on num_branches
        if block_type == 'BASIC':
            block = BasicBlockExpanded
        else:
            block = BottleneckExpanded

        # Create branches - each branch is a sequence of blocks
        self.branches = nn.ModuleList()
        temp_inchannels = list(num_inchannels)

        for branch_idx in range(num_branches):
            branch_layers = []
            current_inchannels = temp_inchannels[branch_idx]
            current_channels = num_channels[branch_idx]

            for block_idx in range(num_blocks[branch_idx]):
                if block_idx == 0:
                    # First block may need downsampling
                    downsample = None
                    if current_inchannels != current_channels * block.expansion:
                        downsample = nn.Sequential(
                            nn.Conv2d(current_inchannels, current_channels * block.expansion,
                                      kernel_size=1, stride=1, bias=False),
                            norm_layer(current_channels * block.expansion),
                        )
                    branch_layers.append(block(current_inchannels, current_channels,
                                               stride=1, downsample=downsample, norm_layer=norm_layer))
                    current_inchannels = current_channels * block.expansion
                else:
                    branch_layers.append(block(current_inchannels, current_channels,
                                               stride=1, downsample=None, norm_layer=norm_layer))

            self.branches.append(nn.Sequential(*branch_layers))
            temp_inchannels[branch_idx] = current_inchannels

        # Update num_inchannels after branches
        self.num_inchannels = temp_inchannels

        # Create fuse layers explicitly
        if num_branches == 1:
            self.fuse_layers = None
        else:
            self.fuse_layers = nn.ModuleList()
            for i in range(num_branches if multi_scale_output else 1):
                fuse_layer_i = nn.ModuleList()
                for j in range(num_branches):
                    if j > i:
                        # Upsample: transposed conv to upsample + 1x1 conv + bn
                        # Calculate scale factor based on channel differences
                        scale_factor = 2 ** (j - i)
                        # Use transposed convolution for upsampling
                        if scale_factor == 1:
                            fuse_layer_i.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[i],
                                          kernel_size=1, stride=1, padding=0, bias=False),
                                norm_layer(num_inchannels[i])
                            ))
                        else:
                            # Transposed conv for upsampling
                            fuse_layer_i.append(nn.Sequential(
                                nn.ConvTranspose2d(num_inchannels[j], num_inchannels[i],
                                                   kernel_size=2 * scale_factor, stride=scale_factor, 
                                                   padding=scale_factor // 2, bias=False),
                                norm_layer(num_inchannels[i])
                            ))
                    elif j == i:
                        fuse_layer_i.append(None)
                    else:
                        # Downsample: series of 3x3 convs with stride 2
                        downsample_layers = []
                        for k in range(i - j):
                            if k == i - j - 1:
                                # Last downsample layer - no relu
                                downsample_layers.append(nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_inchannels[i],
                                              kernel_size=3, stride=2, padding=1, bias=False),
                                    norm_layer(num_inchannels[i])
                                ))
                            else:
                                # Intermediate downsample layers - with relu
                                downsample_layers.append(nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_inchannels[j],
                                              kernel_size=3, stride=2, padding=1, bias=False),
                                    norm_layer(num_inchannels[j]),
                                    nn.ReLU(inplace=True)
                                ))
                        fuse_layer_i.append(nn.Sequential(*downsample_layers))
                self.fuse_layers.append(fuse_layer_i)

        self.relu = nn.ReLU(inplace=True)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        # Apply branches
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        # Fuse branches
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    # Use transposed convolution for upsampling instead of F.interpolate
                    y = y + self.fuse_layers[i][j](x[j])
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HighResolutionNetExpanded(nn.Module):
    """
    Fully expanded HRNet with all modules explicitly defined.
    No nested function definitions - everything is in __init__.
    Forward explicitly calls each operator.
    """

    def __init__(self, cfg, norm_layer=None):
        super(HighResolutionNetExpanded, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        # ========== STEM NETWORK ==========
        self.stem_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn1 = norm_layer(64)
        self.stem_relu1 = nn.ReLU(inplace=True)
        self.stem_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn2 = norm_layer(64)
        self.stem_relu2 = nn.ReLU(inplace=True)

        # ========== STAGE 1 ==========
        # Stage 1 uses Bottleneck blocks, single branch
        stage1_cfg = cfg['STAGE1']
        stage1_num_channels = stage1_cfg['NUM_CHANNELS'][0]
        stage1_num_blocks = stage1_cfg['NUM_BLOCKS'][0]

        # Create stage1 blocks explicitly
        stage1_layers = []
        inplanes_stage1 = 64
        planes_stage1 = stage1_num_channels
        block_expansion = 4  # Bottleneck expansion

        for i in range(stage1_num_blocks):
            if i == 0:
                # First block needs downsampling
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes_stage1, planes_stage1 * block_expansion,
                              kernel_size=1, stride=1, bias=False),
                    norm_layer(planes_stage1 * block_expansion),
                )
                stage1_layers.append(BottleneckExpanded(inplanes_stage1, planes_stage1,
                                                        stride=1, downsample=downsample, norm_layer=norm_layer))
            else:
                stage1_layers.append(BottleneckExpanded(planes_stage1 * block_expansion, planes_stage1,
                                                        stride=1, downsample=None, norm_layer=norm_layer))

        self.stage1_blocks = nn.Sequential(*stage1_layers)
        stage1_out_channel = block_expansion * stage1_num_channels

        # ========== TRANSITION 1 (Stage1 -> Stage2) ==========
        stage2_cfg = cfg['STAGE2']
        stage2_num_channels_raw = stage2_cfg['NUM_CHANNELS']
        stage2_block_type = stage2_cfg['BLOCK']
        stage2_block_exp = 1 if stage2_block_type == 'BASIC' else 4
        stage2_num_channels = [c * stage2_block_exp for c in stage2_num_channels_raw]
        num_branches_pre = 1
        num_branches_cur = len(stage2_num_channels)

        self.transition1 = nn.ModuleList()
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if stage2_num_channels[i] != stage1_out_channel:
                    self.transition1.append(nn.Sequential(
                        nn.Conv2d(stage1_out_channel, stage2_num_channels[i],
                                  kernel_size=3, stride=1, padding=1, bias=False),
                        norm_layer(stage2_num_channels[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    self.transition1.append(None)
            else:
                # Downsample from previous branch
                transition_layers = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = stage1_out_channel if j == 0 else stage2_num_channels[i - 1]
                    outchannels = stage2_num_channels[i] if j == i - num_branches_pre else inchannels
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False),
                        norm_layer(outchannels),
                        nn.ReLU(inplace=True)
                    ))
                self.transition1.append(nn.Sequential(*transition_layers))

        # ========== STAGE 2 ==========
        stage2_num_modules = stage2_cfg['NUM_MODULES']
        stage2_num_branches = stage2_cfg['NUM_BRANCHES']
        stage2_num_blocks = stage2_cfg['NUM_BLOCKS']
        stage2_num_channels_final = stage2_num_channels

        self.stage2_modules = nn.ModuleList()
        num_inchannels_stage2 = stage2_num_channels_final[:]

        for m in range(stage2_num_modules):
            multi_scale_output = True  # All modules output multi-scale except possibly last
            self.stage2_modules.append(
                HighResolutionModuleExpanded(
                    num_branches=stage2_num_branches,
                    block_type=stage2_block_type,
                    num_blocks=stage2_num_blocks,
                    num_inchannels=num_inchannels_stage2,
                    num_channels=stage2_num_channels_raw,
                    fuse_method=stage2_cfg['FUSE_METHOD'],
                    multi_scale_output=multi_scale_output,
                    norm_layer=norm_layer
                )
            )
            num_inchannels_stage2 = self.stage2_modules[-1].get_num_inchannels()

        pre_stage_channels = num_inchannels_stage2

        # ========== TRANSITION 2 (Stage2 -> Stage3) ==========
        stage3_cfg = cfg['STAGE3']
        stage3_num_channels_raw = stage3_cfg['NUM_CHANNELS']
        stage3_block_type = stage3_cfg['BLOCK']
        stage3_block_exp = 1 if stage3_block_type == 'BASIC' else 4
        stage3_num_channels = [c * stage3_block_exp for c in stage3_num_channels_raw]
        num_branches_pre = len(pre_stage_channels)
        num_branches_cur = len(stage3_num_channels)

        self.transition2 = nn.ModuleList()
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if stage3_num_channels[i] != pre_stage_channels[i]:
                    self.transition2.append(nn.Sequential(
                        nn.Conv2d(pre_stage_channels[i], stage3_num_channels[i],
                                  kernel_size=3, stride=1, padding=1, bias=False),
                        norm_layer(stage3_num_channels[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    self.transition2.append(None)
            else:
                transition_layers = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = pre_stage_channels[-1]
                    outchannels = stage3_num_channels[i] if j == i - num_branches_pre else inchannels
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False),
                        norm_layer(outchannels),
                        nn.ReLU(inplace=True)
                    ))
                self.transition2.append(nn.Sequential(*transition_layers))

        # ========== STAGE 3 ==========
        stage3_num_modules = stage3_cfg['NUM_MODULES']
        stage3_num_branches = stage3_cfg['NUM_BRANCHES']
        stage3_num_blocks = stage3_cfg['NUM_BLOCKS']

        self.stage3_modules = nn.ModuleList()
        num_inchannels_stage3 = stage3_num_channels[:]

        for m in range(stage3_num_modules):
            multi_scale_output = True
            self.stage3_modules.append(
                HighResolutionModuleExpanded(
                    num_branches=stage3_num_branches,
                    block_type=stage3_block_type,
                    num_blocks=stage3_num_blocks,
                    num_inchannels=num_inchannels_stage3,
                    num_channels=stage3_num_channels_raw,
                    fuse_method=stage3_cfg['FUSE_METHOD'],
                    multi_scale_output=multi_scale_output,
                    norm_layer=norm_layer
                )
            )
            num_inchannels_stage3 = self.stage3_modules[-1].get_num_inchannels()

        pre_stage_channels = num_inchannels_stage3

        # ========== TRANSITION 3 (Stage3 -> Stage4) ==========
        stage4_cfg = cfg['STAGE4']
        stage4_num_channels_raw = stage4_cfg['NUM_CHANNELS']
        stage4_block_type = stage4_cfg['BLOCK']
        stage4_block_exp = 1 if stage4_block_type == 'BASIC' else 4
        stage4_num_channels = [c * stage4_block_exp for c in stage4_num_channels_raw]
        num_branches_pre = len(pre_stage_channels)
        num_branches_cur = len(stage4_num_channels)

        self.transition3 = nn.ModuleList()
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if stage4_num_channels[i] != pre_stage_channels[i]:
                    self.transition3.append(nn.Sequential(
                        nn.Conv2d(pre_stage_channels[i], stage4_num_channels[i],
                                  kernel_size=3, stride=1, padding=1, bias=False),
                        norm_layer(stage4_num_channels[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    self.transition3.append(None)
            else:
                transition_layers = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = pre_stage_channels[-1]
                    outchannels = stage4_num_channels[i] if j == i - num_branches_pre else inchannels
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False),
                        norm_layer(outchannels),
                        nn.ReLU(inplace=True)
                    ))
                self.transition3.append(nn.Sequential(*transition_layers))

        # ========== STAGE 4 ==========
        stage4_num_modules = stage4_cfg['NUM_MODULES']
        stage4_num_branches = stage4_cfg['NUM_BRANCHES']
        stage4_num_blocks = stage4_cfg['NUM_BLOCKS']

        self.stage4_modules = nn.ModuleList()
        num_inchannels_stage4 = stage4_num_channels[:]

        for m in range(stage4_num_modules):
            # Last module of last stage uses multi_scale_output=True
            multi_scale_output = True
            self.stage4_modules.append(
                HighResolutionModuleExpanded(
                    num_branches=stage4_num_branches,
                    block_type=stage4_block_type,
                    num_blocks=stage4_num_blocks,
                    num_inchannels=num_inchannels_stage4,
                    num_channels=stage4_num_channels_raw,
                    fuse_method=stage4_cfg['FUSE_METHOD'],
                    multi_scale_output=multi_scale_output,
                    norm_layer=norm_layer
                )
            )
            num_inchannels_stage4 = self.stage4_modules[-1].get_num_inchannels()

        # Store config for forward
        self.stage2_cfg = stage2_cfg
        self.stage3_cfg = stage3_cfg
        self.stage4_cfg = stage4_cfg

    def forward(self, x):
        outputs = []

        # ========== STEM ==========
        x = self.stem_conv1(x)
        x = self.stem_bn1(x)
        x = self.stem_relu1(x)
        outputs.append(x)

        x = self.stem_conv2(x)
        x = self.stem_bn2(x)
        x = self.stem_relu2(x)

        # ========== STAGE 1 ==========
        x = self.stage1_blocks(x)

        # ========== TRANSITION 1 ==========
        x_list = []
        num_branches_stage2 = self.stage2_cfg['NUM_BRANCHES']
        for i in range(num_branches_stage2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        # ========== STAGE 2 ==========
        for module in self.stage2_modules:
            x_list = module(x_list)

        # ========== TRANSITION 2 ==========
        x_list_next = []
        num_branches_stage3 = self.stage3_cfg['NUM_BRANCHES']
        num_branches_stage2 = self.stage2_cfg['NUM_BRANCHES']
        for i in range(num_branches_stage3):
            if self.transition2[i] is not None:
                if i < num_branches_stage2:
                    x_list_next.append(self.transition2[i](x_list[i]))
                else:
                    x_list_next.append(self.transition2[i](x_list[-1]))
            else:
                x_list_next.append(x_list[i])
        x_list = x_list_next

        # ========== STAGE 3 ==========
        for module in self.stage3_modules:
            x_list = module(x_list)

        # ========== TRANSITION 3 ==========
        x_list_next = []
        num_branches_stage4 = self.stage4_cfg['NUM_BRANCHES']
        num_branches_stage3 = self.stage3_cfg['NUM_BRANCHES']
        for i in range(num_branches_stage4):
            if self.transition3[i] is not None:
                if i < num_branches_stage3:
                    x_list_next.append(self.transition3[i](x_list[i]))
                else:
                    x_list_next.append(self.transition3[i](x_list[-1]))
            else:
                x_list_next.append(x_list[i])
        x_list = x_list_next

        # ========== STAGE 4 ==========
        for module in self.stage4_modules:
            x_list = module(x_list)

        outputs += x_list
        return outputs

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'alpha' not in name:
                yield param


def _hrnet_expanded(arch, pretrained, progress, **kwargs):
    from .hrnet_config import MODEL_CONFIGS
    model = HighResolutionNetExpanded(MODEL_CONFIGS[arch], **kwargs)
    if pretrained:
        if arch == 'hrnet64':
            arch_key = 'hrnet32_imagenet'
            model_url = model_urls[arch_key]
            loaded_state_dict = load_state_dict_from_url(model_url, progress=progress)
            exp_layers = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var',
                          'conv2.weight', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var']
            lista = ['transition1.0.0.weight', 'transition1.1.0.0.weight',
                     'transition2.2.0.0.weight', 'transition3.3.0.0.weight']
            for k, v in loaded_state_dict.items():
                if k not in exp_layers:
                    if ('layer' not in k) and 'conv' in k or k in lista and len(v.size()) > 1:
                        if k in ['transition1.0.0.weight', 'transition1.1.0.0.weight']:
                            loaded_state_dict[k] = torch.cat([loaded_state_dict[k]] * 2, 0)
                        else:
                            loaded_state_dict[k] = torch.cat([v] * 2, 1) / 2
                            loaded_state_dict[k] = torch.cat([loaded_state_dict[k]] * 2, 0)

                    if 'fuse_layer' in k and 'weight' in k and len(v.size()) > 1:
                        loaded_state_dict[k] = torch.cat([v] * 2, 1) / 2
                        loaded_state_dict[k] = torch.cat([loaded_state_dict[k]] * 2, 0)

                    if 'layer' not in k and len(v.size()) == 1:
                        v = v.unsqueeze(1)
                        v = torch.cat([v] * 2, 0)
                        loaded_state_dict[k] = v.squeeze(1)
                    if 'fuse_layer' in k and len(v.size()) == 1:
                        v = v.unsqueeze(1)
                        v = torch.cat([v] * 2, 0)
                        loaded_state_dict[k] = v.squeeze(1)
                    if len(loaded_state_dict[k].size()) == 2:
                        loaded_state_dict[k] = loaded_state_dict[k].squeeze(1)
        else:
            arch_key = arch + '_imagenet'
            # Note: You may need to update the path for your environment
            pth_path = "/test/monodepth2-master/models/hrnet_imagenet/HRNet_W18_C_cosinelr_cutmix_300epoch.pth.tar"
            try:
                loaded_state_dict = torch.load(pth_path, map_location='cuda:0')
            except FileNotFoundError:
                # If file not found, try loading from URL
                model_url = model_urls.get(arch_key)
                if model_url:
                    loaded_state_dict = load_state_dict_from_url(model_url, progress=progress)
                else:
                    loaded_state_dict = {}

        model.load_state_dict({k: v for k, v in loaded_state_dict.items() if k in model.state_dict()}, strict=False)
    return model


def hrnet18_expanded(pretrained=False, progress=True, **kwargs):
    """HRNet-18 expanded model with all submodules explicitly defined."""
    return _hrnet_expanded('hrnet18', pretrained, progress, **kwargs)


def hrnet32_expanded(pretrained=False, progress=True, **kwargs):
    """HRNet-32 expanded model with all submodules explicitly defined."""
    return _hrnet_expanded('hrnet32', pretrained, progress, **kwargs)


def hrnet48_expanded(pretrained=False, progress=True, **kwargs):
    """HRNet-48 expanded model with all submodules explicitly defined."""
    return _hrnet_expanded('hrnet48', pretrained, progress, **kwargs)


def hrnet64_expanded(pretrained=False, progress=True, **kwargs):
    """HRNet-64 expanded model with all submodules explicitly defined."""
    return _hrnet_expanded('hrnet64', pretrained, progress, **kwargs)
