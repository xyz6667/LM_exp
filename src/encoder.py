import torch
import torch.nn as nn
import copy
from src.attention import MultiHeadAttention
from src.layers import PositionWiseFeedForward, SublayerConnection, LayerNorm
from collections.abc import Callable

class EncoderBlock(nn.Module):
    """
    单个 Transformer 编码器块 (EncoderBlock)
    """
    def __init__(self, size: int, self_attn: MultiHeadAttention, feed_forward: PositionWiseFeedForward, dropout: float):
        """
        初始化
        参数:
            size (int): d_model, 模型的维度
            self_attn (MultiHeadAttention): 多头自注意力模块
            feed_forward (PositionWiseFeedForward): 前馈网络模块
            dropout (float): Dropout 概率
        """
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 需要两个 SublayerConnection，一个自注意力，一个前馈网络
        self.sublayer1 = SublayerConnection(size, dropout)
        self.sublayer2 = SublayerConnection(size, dropout)
        self.size = size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            x (torch.Tensor): 输入张量, 形状 (batch_size, seq_len, d_model)
            mask (torch.Tensor): 掩码
        返回:
            torch.Tensor: 编码器块的输出, 形状 (batch_size, seq_len, d_model)
        """
        # 多头注意力 + 残差和层归一化
        # self_attn 接收 q, k, v。在自注意力 (self-attention) 中, q, k, v 都是 x
        x = self.sublayer1(x, lambda x:self.self_attn(x, x, x, mask))

        # 前馈网络 + 残差和层归一化
        x = self.sublayer2(x, self.feed_forward)

        return x
    
def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """
    生成 N 个相同的模块层 (Deep Copy)
    
    参数:
        module (nn.Module): 需要被复制的模块
        N (int): 复制的数量
    返回:
        nn.ModuleList: 包含 N 个 module 副本的模块列表
    """
    # 使用 deepcopy 确保每个模块是独立的副本，参数不共享
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    """
    Transformer 编码器 (Encoder), 由 N 个 EncoderBlock 堆叠而成
    """
    def __init__(self, block: EncoderBlock, N: int):
        """
        初始化
        参数:
            block (EncoderBlock): 一个 EncoderBlock 实例, 将被复制 N 次
            N (int): 堆叠的层数
        """
        super().__init__()
        # 使用 clones 函数赋值 N 个 EncoderBlock
        self.layers = clones(block, N)
        # 在 N 层之后再加一个总的 layerNorm
        self.norm = LayerNorm(block.size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播, 让输入依次通过 N 个编码器块
        参数:
            x (torch.Tensor): 输入, 形状 (batch_size, seq_len, d_model)
            mask (torch.Tensor): 掩码
        返回:
            torch.Tensor: 编码器的输出, 形状 (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)

        # 经过 N 层后再通过最后的 LayerNorm
        return self.norm(x)