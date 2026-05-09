"""
HRNet BN 和 Conv 融合工具
用于推理时加速，将 BatchNorm 层融合到 Conv 层中
"""

import torch
import torch.nn as nn
from typing import Union


def fuse_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    将 BatchNorm 层融合到 Conv 层中
    
    数学原理:
    BN: y = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta
    Conv: x = W * input + b
    
    融合后: 
    W_new = gamma / sqrt(sigma^2 + eps) * W
    b_new = gamma / sqrt(sigma^2 + eps) * (b - mu) + beta
    
    Args:
        conv: Conv2d 层
        bn: BatchNorm2d 层
    
    Returns:
        融合后的 Conv2d 层
    """
    # 获取 BN 的参数
    gamma = bn.weight
    beta = bn.bias
    mu = bn.running_mean
    sigma = bn.running_var
    eps = bn.eps
    
    # 计算缩放因子
    scale = gamma / torch.sqrt(sigma + eps)
    
    # 融合权重
    conv_weight = conv.weight.data
    conv_bias = conv.bias.data if conv.bias is not None else torch.zeros_like(mu)
    
    # 新的权重: W_new = scale * W
    new_weight = conv_weight * scale.view(-1, 1, 1, 1)
    
    # 新的偏置: b_new = scale * (b - mu) + beta
    new_bias = scale * (conv_bias - mu) + beta
    
    # 创建新的 Conv2d 层
    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True
    )
    
    # 复制融合后的参数
    fused_conv.weight.data = new_weight
    fused_conv.bias.data = new_bias
    
    return fused_conv


def fuse_model(model: nn.Module) -> nn.Module:
    """
    递归遍历模型，将所有 Conv+BN 结构融合
    
    支持的模式:
    1. Sequential 中的连续 Conv2d + BatchNorm2d
    2. 单独的模块属性命名如 conv1 + bn1
    
    Args:
        model: 要融合的模型
    
    Returns:
        融合后的模型 (原地修改)
    """
    # 首先处理所有子模块
    for name, module in list(model.named_children()):
        # 递归处理子模块
        fuse_model(module)
        
        # 如果是 Sequential，检查是否有连续的 Conv+BN
        if isinstance(module, nn.Sequential):
            # 转换为列表以便修改
            children = list(module.named_children())
            i = 0
            while i < len(children) - 1:
                curr_name, curr_module = children[i]
                next_name, next_module = children[i + 1]
                
                # 检查是否是 Conv2d + BatchNorm2d 的组合
                if isinstance(curr_module, nn.Conv2d) and isinstance(next_module, nn.BatchNorm2d):
                    # 融合
                    fused_conv = fuse_bn_into_conv(curr_module, next_module)
                    
                    # 替换当前层为融合后的 Conv
                    setattr(module, curr_name, fused_conv)
                    
                    # 将下一层替换为 Identity
                    setattr(module, next_name, nn.Identity())
                    
                    # 跳过下一层
                    i += 2
                    
                    # 更新 children 列表
                    children = list(module.named_children())
                else:
                    i += 1
        
        # 处理命名模式: conv* + bn* (如 conv1 + bn1, conv2 + bn2)
        elif hasattr(model, name):
            # 尝试查找对应的 BN 层
            if 'conv' in name.lower() and not 'depthwise' in name.lower():
                # 构造可能的 BN 名称
                bn_name = name.replace('conv', 'bn')
                if hasattr(model, bn_name):
                    conv_module = getattr(model, name)
                    bn_module = getattr(model, bn_name)
                    
                    if isinstance(conv_module, nn.Conv2d) and isinstance(bn_module, nn.BatchNorm2d):
                        # 融合
                        fused_conv = fuse_bn_into_conv(conv_module, bn_module)
                        
                        # 替换 Conv 层
                        setattr(model, name, fused_conv)
                        
                        # 将 BN 层替换为 Identity
                        setattr(model, bn_name, nn.Identity())
    
    return model


def prepare_for_inference(model: nn.Module) -> nn.Module:
    """
    准备模型用于推理：设置为 eval 模式并融合 BN
    
    Args:
        model: 原始模型
    
    Returns:
        优化后的推理模型
    """
    # 切换到评估模式
    model.eval()
    
    # 融合 BN 到 Conv
    fused_model = fuse_model(model)
    
    return fused_model


# 使用示例
if __name__ == "__main__":
    from hrnet_encoder_expanded import hrnet18_expanded
    
    # 创建模型
    model = hrnet18_expanded(pretrained=False)
    model.eval()
    
    print("=== 融合前 ===")
    print(model)
    
    # 测试融合前的输出
    dummy_input = torch.randn(1, 3, 256, 512)
    with torch.no_grad():
        output_before = model(dummy_input)
    
    # 融合 BN
    fused_model = prepare_for_inference(model)
    
    print("\n=== 融合后 ===")
    print(fused_model)
    
    # 测试融合后的输出
    with torch.no_grad():
        output_after = fused_model(dummy_input)
    
    # 验证输出一致性
    diff = torch.abs(output_before[0] - output_after[0]).max()
    print(f"\n最大差异: {diff.item():.6f}")
    
    if diff < 1e-5:
        print("✓ 融合成功！输出一致")
    else:
        print("✗ 融合可能有问题，输出不一致")
    
    # 保存融合后的模型
    torch.save(fused_model.state_dict(), "hrnet18_fused.pth")
    print("\n融合后的模型已保存为: hrnet18_fused.pth")
