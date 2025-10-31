import torch
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.layers import PositionWiseFeedForward, SublayerConnection, LayerNorm
from src.utils import clones

class DecoderBlock(nn.Module):
    """
    单个 Transformer 解码器块 (DecoderBlock)
    """
    def __init__(self, size: int, self_attn: MultiHeadAttention, 
                 src_attn: MultiHeadAttention, feed_forward: PositionWiseFeedForward, 
                 dropout: float):
        """
        初始化
        参数:
            size (int): d_model, 模型的维度
            self_attn (MultiHeadAttention): 解码器自身的 (带掩码的) 多头自注意力
            src_attn (MultiHeadAttention): 交叉注意力 (Cross-Attention) 模块
            feed_forward (PositionWiseFeedForward): 前馈网络模块
            dropout (float): Dropout 概率
        """
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        # 三个 sublayerconnection
        self.sublayer1 = SublayerConnection(size, dropout)
        self.sublayer2 = SublayerConnection(size, dropout)
        self.sublayer3 = SublayerConnection(size, dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, 
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            x (torch.Tensor): 目标序列 (来自解码器输入), 形状 (batch_size, tgt_seq_len, d_model)
            memory (torch.Tensor): 编码器的输出, 形状 (batch_size, src_seq_len, d_model)
            src_mask (torch.Tensor): 源序列的掩码 (用于交叉注意力)
            tgt_mask (torch.Tensor): 目标序列的掩码 (用于自注意力, 即 future mask)
        返回:
            torch.Tensor: 解码器块的输出, 形状 (batch_size, tgt_seq_len, d_model)
        """
        # 带掩码的多头自注意力
        # q k v 都是 x（目标序列），掩码是 tgt_mask
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # 交叉注意力 cross-attention
        # query 来自 x 编码器本身； key 和 value 来自 memory 编码器的输出
        # 掩码是 src_mask 屏蔽掉编码器输入中的 padding
        x = self.sublayer2(x, lambda x: self.src_attn(x, memory, memory, src_mask))

        # 前馈网络
        x = self.sublayer3(x, self.feed_forward)

        return x
    
class Decoder(nn.Module):
    """
    Transformer 解码器 (Decoder), 由 N 个 DecoderBlock 堆叠而成
    """
    def __init__(self, block: DecoderBlock, N: int):
        """
        初始化
        参数:
            block (DecoderBlock): 一个 DecoderBlock 实例, 将被复制 N 次
            N (int): 堆叠的层数
        """
        super().__init__()
        # 用 clones 函数复制 N 个 DecoderBlock
        self.layers = clones(block, N)
        # 在 N 层后再加一个总的 layerNorm
        self.norm = LayerNorm(block.size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, 
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播, 让输入依次通过 N 个解码器块
        参数:
            x (torch.Tensor): 目标序列 (batch_size, tgt_seq_len, d_model)
            memory (torch.Tensor): 编码器输出 (batch_size, src_seq_len, d_model)
            src_mask (torch.Tensor): 源序列掩码
            tgt_mask (torch.Tensor): 目标序列掩码
        返回:
            torch.Tensor: 解码器的输出, 形状 (batch_size, tgt_seq_len, d_model)
        """
        # 通过 N 层
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        # 最后通过 LayerNorm
        return self.norm(x)
    