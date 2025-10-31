import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    位置编码 (Positional Encoding) 模块
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        初始化位置编码模块
        参数:
            d_model (int): 模型的总维度 (embedding dimension)
            dropout (float): Dropout 概率
            max_len (int): 位置编码的最大长度
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够长的位置编码矩阵
        # 形状：（max_len, d_model）
        pe = torch.zeros(max_len, d_model)

        # 创建一个位置索引张量
        # 形状：（max_len, 1）
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)

        # 计算除法项的分母
        # 10000^(2i / d_model)
        # div_term 形状：（d_model / 2）
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 计算偶数维度的 sin
        pe[:, 0::2] = torch.sin(position * div_term)

        # 计算奇数维度的 cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将 pe 形状变为(1, max_len, d_model) 以便进行批处理广播
        pe = pe.unsqueeze(0)

        # 将 pe 注册为 buffer
        # buffer 是模型的一部分（会随模型保存和加载），但它不是模型参数（不会被优化器更新）
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            x (torch.Tensor): 输入的词嵌入, 形状 (batch_size, seq_len, d_model)
        返回:
            torch.Tensor: 添加了位置编码和 dropout 后的张量
        """
        # x.size(1) 是输入的实际序列长度 (seq_len)
        # 从预先计算的 pe 中取出对应长度的部分
        # self.pe[:, :x.size(1), :] 形状: (1, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)