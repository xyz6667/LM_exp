import torch
import torch.nn as nn
import copy

def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """
    生成 N 个相同的模块层 (Deep Copy)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size: int) -> torch.Tensor:
    """
    生成一个用于屏蔽未来位置的下三角矩阵掩码。

    参数:
        size (int): 序列的长度 (seq_len)

    返回:
        torch.Tensor: 一个形状为 (1, size, size) 的掩码张量
    """
    # 创建一个形状为(size, size)的矩阵
    attn_shape = (1, size, size)

    # torch.triu(torch.ones(attn_shape), diagonal=1) 会生成一个上三角矩阵, 对角线为0
    # (diagonal=1 表示从对角线往上第一个元素开始为1)
    # 把它转为 byte (或 bool) 类型
    mask = torch.triu(torch.ones(attn_shape, dtype = torch.uint8), diagonal = 1)

    # 想要的掩码是：未来位置为 0，当前和过去位置为1
    return mask == 0