import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    """
    逐位置前馈网络 (Position-Wise Feed-Forward Network)
    """
    def __init__(self, d_model:int, d_ff:int):
        """
        初始化函数
        参数：
            d_model (int): 模型的总维度 (embedding dimension)
            d_ff (int): 前馈网络的隐藏层维度
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # 定义两个线性层
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            x (torch.Tensor): 输入张量, 形状 (batch_size, seq_len, d_model)

        返回:
            torch.Tensor: 输出张量, 形状 (batch_size, seq_len, d_model)
        """
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        x = self.relu(self.linear_1(x))
        # (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.linear_2(x)

        return x

class LayerNorm(nn.Module):
    """
    层归一化 (Layer Normalization) 模块。
    """
    def __init__(self, features: int, eps: float = 1e-6):
        """
        初始化
        参数:
            features (int): 特征维度 (d_model)
            eps (float): 一个很小的数，用于防止分母为零
        """
        super().__init__()
        # gamma（可学习的缩放参数）
        self.gamma = nn.Parameter(torch.ones(features))
        # beta（可学习的偏移参数）
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            x (torch.Tensor): 输入, 形状 (batch_size, seq_len, d_model)
        返回:
            torch.Tensor: 归一化后的输出, 形状 (batch_size, seq_len, d_model)
        """
        # 在最后一个维度（d_model）上计算均值和方差
        # keepdim=True 保持维度为 （batch_size, seq_len, 1)以便广播
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1,keepdim = True, unbiased = False)

        # 归一化
        normalized = (x - mean) / (std + self.eps)

        # 应用可学习的参数
        return self.gamma * normalized + self.beta
    
class SublayerConnection(nn.Module):
    """
    子层连接模块 (实现 LayerNorm(x + Sublayer(x)))
    """
    def __init__(self, size: int, dropout: float):
        """
        初始化
        参数:
            size (int): 模型维度 (d_model)
            dropout (float): Dropout 的概率
        """
        super().__init__()
        self.nrom = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        前向传播
        参数:
            x (torch.Tensor): 输入, 形状 (batch_size, seq_len, d_model)
            sublayer (nn.Module): 要包裹的子模块 (例如 MultiHeadAttention 或 FFN)
        返回:
            torch.Tensor: 输出, 形状 (batch_size, seq_len, d_model)
        """
        # x + Sublayer(x)
        residual = x + self.dropout(sublayer(x))
        # LayerNorm(...)
        return self.norm(residual)