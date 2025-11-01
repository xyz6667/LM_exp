import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchtext
torchtext.disable_torchtext_deprecation_warning()

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

import os
import math
import time
import argparse
import random
import numpy as np

from src.model import EncoderDecoderModel
from src.utils import subsequent_mask

# 全局配置和超参数

D_MODEL = 128
H = 4
D_FF = 512
N_LAYERS = 2
DROPOUT = 0.1

# 训练相关超参数
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 10

# 数据集和语言设置
DATASET_NAME = "iwslt2017"
LANG_PAIR = "iwslt2017-de-en"
SRC_LANG = "de"
TGT_LANG = "en"

# 特殊 Token
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# 数据预处理
def yield_tokens(dataset_iter, tokenizer, lang_key):
    """
    一个辅助函数, 用于遍历 Hugging Face datasets 对象并返回分词后的 token 列表
    lang_key: "de" 或 "en"
    """
    for item in dataset_iter:
        # 数据集的结构是 {'translation': {'de': '...', 'en': '...'}}
        yield tokenizer(item['translation'][lang_key])

def data_collate_fn(batch, tokenizers, vocabs, pad_idx, bos_idx, eos_idx, src_lang, tgt_lang):
    """
    自定义的 collate_fn, 用于 DataLoader
    负责分词、添加特殊 token、数值化和 padding
    """
    src_batch_list, tgt_batch_list = [], []
    
    # 遍历批次中的每个样本
    for sample in batch:
        src_text = sample['translation'][src_lang]
        tgt_text = sample['translation'][tgt_lang]
        
        # 1. 分词
        src_tokens = tokenizers[src_lang](src_text)
        tgt_tokens = tokenizers[tgt_lang](tgt_text)
        
        # 2. 数值化 + 添加 BOS/EOS
        src_ids = [bos_idx] + vocabs[src_lang](src_tokens) + [eos_idx]
        tgt_ids = [bos_idx] + vocabs[tgt_lang](tgt_tokens) + [eos_idx]
        
        # 3. 添加到列表中, 准备 padding
        src_batch_list.append(torch.tensor(src_ids, dtype=torch.int64))
        tgt_batch_list.append(torch.tensor(tgt_ids, dtype=torch.int64))

    # 4. 使用 pad_sequence 进行填充
    # batch_first=True 使得输出形状为 (batch_size, max_seq_len)
    src_padded = nn.utils.rnn.pad_sequence(
        src_batch_list, padding_value=pad_idx, batch_first=True
    )
    tgt_padded = nn.utils.rnn.pad_sequence(
        tgt_batch_list, padding_value=pad_idx, batch_first=True
    )
    
    return src_padded, tgt_padded

# 掩码创建

def create_masks(src_batch: torch.Tensor, tgt_batch: torch.Tensor, pad_idx: int, device: torch.device):
    """
    创建 padding 掩码和 future 掩码
    
    参数:
        src_batch (torch.Tensor): 源序列批次, 形状 (batch_size, src_seq_len)
        tgt_batch (torch.Tensor): 目标序列批次, 形状 (batch_size, tgt_seq_len)
        pad_idx (int): PAD token 的索引
        device (torch.device): 要将掩码创建在哪个设备上 (cpu or cuda)
        
    返回:
        tuple[torch.Tensor, torch.Tensor]: src_mask, tgt_mask
    """
    
    # 1. 源序列掩码 (src_mask)
    # (batch_size, src_seq_len) -> (batch_size, 1, 1, src_seq_len)
    src_mask = (src_batch != pad_idx).unsqueeze(1).unsqueeze(2).to(device)
    
    
    # 2. 目标序列掩码 (tgt_mask)
    # (batch_size, tgt_seq_len) -> (batch_size, 1, 1, tgt_seq_len)
    tgt_pad_mask = (tgt_batch != pad_idx).unsqueeze(1).unsqueeze(2).to(device)
    
    # subsequent_mask 会生成 (1, tgt_seq_len, tgt_seq_len)
    tgt_len = tgt_batch.size(1)
    tgt_sub_mask = subsequent_mask(tgt_len).to(device) # 形状: (1, tgt_seq_len, tgt_seq_len)
    
    # tgt_pad_mask (batch_size, 1, 1, tgt_seq_len)
    # tgt_sub_mask (1, 1, tgt_seq_len, tgt_seq_len)
    tgt_mask = tgt_pad_mask & tgt_sub_mask
    
    # 在我们的 MultiHeadAttention 实现中, 我们期望的掩码是
    # (batch_size, 1, query_len, key_len)
    # src_mask: (batch_size, 1, 1, src_seq_len) - 用于交叉注意力的 K/V (key_len=src_seq_len)
    # tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len) - 用于自注意力的 Q/K/V
    
    # 注意: 我们的 scaled_dot_product_attention 实现是:
    # scores = scores.masked_fill(mask == 0, -1e9)
    # 这意味着 mask 中 False (0) 的位置会被屏蔽, True (1) 的位置会保留
    # 而我们上面生成的 (src_batch != pad_idx) 和 subsequent_mask 
    # 都是 True 代表保留, False 代表屏蔽, 所以是正确的。
    
    return src_mask, tgt_mask

# 训练与评估

def train_epoch(model, train_loader, optimizer, criterion, pad_idx, device):
    """
    单个 epoch 的训练循环
    """
    model.train() # 将模型设置为训练模式
    total_loss = 0
    
    # 使用 tqdm 创建进度条
    pbar = tqdm(train_loader, desc=f"Training Epoch", total=len(train_loader))
    
    for src_batch, tgt_batch in pbar:
        # 1. 将数据移动到设备
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        
        # 2. 准备解码器输入和目标输出
        # Teacher Forcing:
        # tgt_input 是 [BOS, w1, w2, ..., wN]
        # tgt_output 是 [w1, w2, ..., wN, EOS]
        # 我们在 collate_fn 中已经添加了 BOS/EOS, 所以:
        tgt_input = tgt_batch[:, :-1] # (batch_size, seq_len - 1)
        tgt_output = tgt_batch[:, 1:]  # (batch_size, seq_len - 1)
        
        # 3. 创建掩码 (基于 tgt_input)
        src_mask, tgt_mask = create_masks(src_batch, tgt_input, pad_idx, device)
        
        # 4. 前向传播
        logits = model(src_batch, tgt_input, src_mask, tgt_mask)
        # logits 形状: (batch_size, seq_len - 1, tgt_vocab_size)
        
        # 5. 梯度清零
        optimizer.zero_grad()
        
        # 6. 计算损失
        # CrossEntropyLoss 期望 (N, C) 和 (N)
        # N = batch_size * (seq_len - 1), C = tgt_vocab_size
        loss = criterion(
            logits.reshape(-1, logits.size(-1)), 
            tgt_output.reshape(-1)
        )
        
        # 7. 反向传播
        loss.backward()
        
        # 8. 梯度裁剪 (防止梯度爆炸) 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 9. 更新参数
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    
    # 返回平均损失
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, pad_idx, device):
    """
    在验证集上评估模型
    """
    model.eval() # 将模型设置为评估模式
    total_loss = 0
    
    # 使用 torch.no_grad() 禁用梯度计算
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validation", total=len(val_loader))
        for src_batch, tgt_batch in pbar:
            # 1. 将数据移动到设备
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            # 2. 准备输入和输出
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]
            
            # 3. 创建掩码
            src_mask, tgt_mask = create_masks(src_batch, tgt_input, pad_idx, device)
            
            # 4. 前向传播
            logits = model(src_batch, tgt_input, src_mask, tgt_mask)
            
            # 5. 计算损失
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), 
                tgt_output.reshape(-1)
            )
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
    
    # 返回平均损失
    return total_loss / len(val_loader)

def set_seed(seed: int):
    """
    设置随机种子以确保可复现性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保 PyTorch 使用确定性的算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 主函数

def main():
    """
    主执行函数
    """
    # ArgParse
    parser = argparse.ArgumentParser(description="Transformer 训练脚本")
    parser.add_argument('--seed', type=int, default=42, 
                        help='设置随机种子以确保可复现性')
    args = parser.parse_args()
    
    # 设置种子
    set_seed(args.seed)

    print("Starting Transformer training process...")

    # 创建唯一的输出路径
    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    # 定义所有输出文件夹
    output_dir = "training_outputs"
    log_dir = os.path.join(output_dir, "logs")
    model_dir = os.path.join(output_dir, "models")

    # 创建这些文件夹 (如果它们不存在)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 创建本次运行的唯一文件名
    log_filename = os.path.join(log_dir, f"training_log_{run_timestamp}.txt")
    model_save_path = os.path.join(model_dir, f"transformer_model_{run_timestamp}.pt")
    vocab_save_path = os.path.join(model_dir, f"vocabs_{run_timestamp}.pt")
    
    # 检查是否有 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    print(f"Loading {DATASET_NAME} dataset...")
    dataset_train = load_dataset(DATASET_NAME, LANG_PAIR, split='train', trust_remote_code=True)
    dataset_val = load_dataset(DATASET_NAME, LANG_PAIR, split='validation', trust_remote_code=True)

    # 构建分词器
    print("Loading tokenizers...")
    # 'basic_english' 是一个简单的、基于空格和标点符号的分词器
    # 我们暂时对德语和英语都使用它, 以保持简单
    tokenizers = {
        SRC_LANG: get_tokenizer('basic_english'),
        TGT_LANG: get_tokenizer('basic_english')
    }

    # 构建词汇表
    print("Building vocabularies...")
    
    # 定义特殊 token 和它们的索引
    special_tokens = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
    unk_idx = special_tokens.index(UNK_TOKEN)
    pad_idx = special_tokens.index(PAD_TOKEN)
    bos_idx = special_tokens.index(BOS_TOKEN)
    eos_idx = special_tokens.index(EOS_TOKEN)

    # 创建源语言词汇表
    src_vocab = build_vocab_from_iterator(
        yield_tokens(dataset_train, tokenizers[SRC_LANG], SRC_LANG),
        specials=special_tokens,
        special_first=True # 确保特殊 token 在索引 0, 1, 2, 3
    )
    src_vocab.set_default_index(unk_idx) # 设置默认索引, 当遇到未登录词时返回 unk_idx

    # 创建目标语言词汇表
    tgt_vocab = build_vocab_from_iterator(
        yield_tokens(dataset_train, tokenizers[TGT_LANG], TGT_LANG),
        specials=special_tokens,
        special_first=True
    )
    tgt_vocab.set_default_index(unk_idx)
    
    print(f"Vocab sizes: SRC={len(src_vocab)}, TGT={len(tgt_vocab)}")

    # 实例化 DataLoader
    print("Creating DataLoaders...")

    # 将 tokenizers 和 vocabs 组合成字典, 方便传递
    vocabs = {SRC_LANG: src_vocab, TGT_LANG: tgt_vocab}

    collate_fn_wrapper = lambda batch: data_collate_fn(
        batch, 
        tokenizers, 
        vocabs, 
        pad_idx, 
        bos_idx, 
        eos_idx, 
        SRC_LANG, 
        TGT_LANG
    )
    
    train_loader = DataLoader(
        dataset_train, 
        batch_size=BATCH_SIZE, 
        shuffle=True, # 训练集需要打乱
        collate_fn=collate_fn_wrapper
    )
    
    val_loader = DataLoader(
        dataset_val, 
        batch_size=BATCH_SIZE, 
        shuffle=False, # 验证集不需要打乱
        collate_fn=collate_fn_wrapper
    )
    
    # 初始化模型、损失函数、优化器
    print("Initializing model...")
    
    # 获取词汇表大小
    src_vocab_size = len(vocabs[SRC_LANG])
    tgt_vocab_size = len(vocabs[TGT_LANG])
    
    # 实例化模型
    model = EncoderDecoderModel(
        src_vocab=src_vocab_size,
        tgt_vocab=tgt_vocab_size,
        d_model=D_MODEL,
        N=N_LAYERS,
        h=H,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(device)
    
    # 打印模型参数统计 
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # 损失函数 (CrossEntropyLoss)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9
    )

    # --- 新增：写入超参数到日志 ---
    print(f"Writing log to: {log_filename}")
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"--- Training Log: {run_timestamp} ---\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Device: {device}\n\n")
        f.write("--- Hyperparameters ---\n")
        f.write(f"d_model: {D_MODEL}\n")
        f.write(f"Layers (N): {N_LAYERS}\n")
        f.write(f"Heads (h): {H}\n")
        f.write(f"d_ff: {D_FF}\n")
        f.write(f"Dropout: {DROPOUT}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n\n")
        f.write("--- Vocab Sizes ---\n")
        f.write(f"SRC Vocab: {src_vocab_size}\n")
        f.write(f"TGT Vocab: {tgt_vocab_size}\n\n")
        f.write("--- Training Starts ---\n")

    # 运行训练循环
    print("Starting training loop...")
    
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, pad_idx, device)
        
        # 评估
        val_loss = evaluate(model, val_loader, criterion, pad_idx, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        # 打印 epoch 结果
        print(f"Epoch: {epoch:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
        print(f"\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}")

        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(f"\nEpoch: {epoch:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s\n")
            f.write(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}\n")
            f.write(f"\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}\n")
        
    # 保存模型和词汇表 
    print("Saving model and vocabs...")
    torch.save(model.state_dict(), model_save_path)
    torch.save(vocabs, vocab_save_path)
    print(f"Model saved to: {model_save_path}")
    print(f"Vocabs saved to: {vocab_save_path}")

if __name__ == "__main__":
    main()