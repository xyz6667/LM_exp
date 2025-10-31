import torch
import torch.nn as nn
import copy
import math

from src.attention import MultiHeadAttention
from src.layers import PositionWiseFeedForward, LayerNorm
from src.encoder import Encoder, EncoderBlock
from src.decoder import Decoder, DecoderBlock
from src.embedding import PositionalEncoding

class EncoderDecoderModel(nn.Module):
    """
    一个标准的 Encoder-Decoder Transformer 架构
    """
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int, 
                 N: int, h: int, d_ff: int, dropout: float = 0.1):
        """
        初始化完整的模型
        参数:
            src_vocab (int): 源语言词汇表大小
            tgt_vocab (int): 目标语言词汇表大小
            d_model (int): 模型维度
            N (int): Encoder/DecoderBlock 的层数
            h (int): 多头注意力的头数
            d_ff (int): FFN 的中间维度
            dropout (float): Dropout 概率
        """
        super().__init__()
        self.d_model = d_model
        
        # 实例化核心组件
        c = copy.deepcopy
        attn = MultiHeadAttention(d_model, h)
        ffn = PositionWiseFeedForward(d_model, d_ff)
        pe = PositionalEncoding(d_model, dropout)
        
        # 编码器 Encoder
        self.encoder = Encoder(
            EncoderBlock(d_model, c(attn), c(ffn), dropout), 
            N
        )
        
        # 解码器 Decoder
        # 解码器块需要两个独立的注意力模块 (self_attn, src_attn)
        self.decoder = Decoder(
            DecoderBlock(d_model, c(attn), c(attn), c(ffn), dropout), 
            N
        )
        
        # 嵌入层 (Embeddings)
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        
        # 位置编码 (Positional Encoding)
        self.pos_encoder = pe
        
        # 输出层 (Generator)
        # 将解码器输出映射到目标词汇表
        self.generator = nn.Linear(d_model, tgt_vocab)
        
        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        # 对线性层使用 Xavier/Glorot 初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        完整的 Encoder-Decoder 前向传播
        参数:
            src (torch.Tensor): 源序列, 形状 (batch_size, src_seq_len)
            tgt (torch.Tensor): 目标序列, 形状 (batch_size, tgt_seq_len)
            src_mask (torch.Tensor): 源序列掩码
            tgt_mask (torch.Tensor): 目标序列掩码
        返回:
            torch.Tensor: 模型的 logits 输出, 形状 (batch_size, tgt_seq_len, tgt_vocab)
        """
        # 将源序列编码, 得到 memory
        memory = self.encode(src, src_mask)
        
        # 将目标序列连同 memory 一起解码
        decoder_output = self.decode(memory, src_mask, tgt, tgt_mask)
        
        # 通过最后的全连接层生成 logits
        logits = self.generator(decoder_output)
        
        return logits

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        编码器前向传播
        """
        # (batch_size, src_seq_len) -> (batch_size, src_seq_len, d_model)
        src_embedded = self.src_embed(src) * math.sqrt(self.d_model)
        src_with_pos = self.pos_encoder(src_embedded)
        return self.encoder(src_with_pos, src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, 
               tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        解码器前向传播
        """
        # (batch_size, tgt_seq_len) -> (batch_size, tgt_seq_len, d_model)
        tgt_embedded = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_with_pos = self.pos_encoder(tgt_embedded)
        return self.decoder(tgt_with_pos, memory, src_mask, tgt_mask)