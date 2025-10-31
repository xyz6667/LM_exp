import torch.nn as nn
import math
import torch

def scaled_dot_product_attention(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:torch.Tensor=None) -> tuple[torch.Tensor,torch.Tensor]:
    """
    计算缩放点积注意力。

    参数:
        q (torch.Tensor): 查询 Queries, 形状为 (..., seq_len_q, d_k)
        k (torch.Tensor): 键 Keys, 形状为 (..., seq_len_k, d_k)
        v (torch.Tensor): 值 Values, 形状为 (..., seq_len_v, d_v), 其中 seq_len_k == seq_len_v
        mask (torch.Tensor, optional): 掩码, 形状为 (..., seq_len_q, seq_len_k)。默认为 None。

    返回:
        tuple[torch.Tensor, torch.Tensor]: 输出张量和注意力权重。
                                            输出张量形状为 (..., seq_len_q, d_v)
                                            注意力权重形状为 (..., seq_len_q, seq_len_k)
    """
    # 1. 计算 Q 和 K^T 的点积
    # Q(..., seq_len_q, d_k) @ K^T(..., d_k, seq_len_k) -> (..., seq_len_q, seq_len_k)
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2,-1))

    # 2. 缩放
    scores = scores / math.sqrt(d_k)

    # 3. 应用掩码（如果提供）
    # mask==0 的位置被设置为 -1e9，使得 softmax 后对应位置的权重为 0
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    
    # 4. 计算注意力权重
    # P_ATTN(..., seq_len_q, seq_len_k)
    p_attn = torch.softmax(scores, dim=-1)

    # 5. 将权重与 V 相乘得到最终输出
    # output(..., seq_len_q, d_v)
    output = torch.matmul(p_attn, v)
    
    return output, p_attn

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块
    """
    def __init__(self, d_model: int, h: int):
        """
        初始化函数
        参数：
            d_model (int): 模型的总维度 (embedding dimension)
            h (int): 注意力头的数量
        """
        super().__init__()
        assert d_model % h == 0, "d_model 必须能被 h 整除"

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h # 每个头的维度

        # 定义Q, K, V和输出的线性层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:    
        """
        前向传播
        参数:
            q (torch.Tensor): 查询 Queries, 形状 (batch_size, seq_len_q, d_model)
            k (torch.Tensor): 键 Keys, 形状 (batch_size, seq_len_k, d_model)
            v (torch.Tensor): 值 Values, 形状 (batch_size, seq_len_v, d_model)
            mask (torch.Tensor, optional): 掩码。默认为 None。

        返回:
            torch.Tensor: 多头注意力的输出, 形状 (batch_size, seq_len_q, d_model)
        """
        batch_size = q.size(0)

        # 1. 线性映射
        # q, k, v 形状变为 (batch_size, seq_len, d_model)
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)
    
        # 2. 拆分成多头
        # 原始形状: (batch_size, seq_len, d_model)
        # 变形后: (batch_size, seq_len, h, d_k)
        # 轴交换后: (batch_size, h, seq_len, d_k)
        q = q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    
        # 3. 并行计算缩放点积注意力
        # attn_output 形状: (batch_size, h, seq_len_q, d_k)
        # p_attn 形状: (batch_size, h, seq_len_q, seq_len_k)
        attn_output, p_attn = scaled_dot_product_attention(q, k, v, mask)
    
        # 4. 拼接与再次映射
        # 轴交换回: (batch_size, seq_len_q, h, d_k)
        # 拼接后： (batch_size, seq_len_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    
        # 通过最后一个线性层
        output = self.W_o(attn_output)
    
        return output
    
