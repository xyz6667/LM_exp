# translate.py

import torch
import torch.nn as nn
import math
from src.model import EncoderDecoderModel
from src.utils import subsequent_mask
from torchtext.data.utils import get_tokenizer

# --- 1. 定义超参数 (必须和训练时完全一致) ---
D_MODEL = 128
H = 4
D_FF = 512
N_LAYERS = 2
DROPOUT = 0.1

SRC_LANG = "de"
TGT_LANG = "en"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

# --- 2. 推理函数 (Greedy Decode) ---

def greedy_decode(model, src, src_mask, max_len, start_symbol_idx, end_symbol_idx, device):
    """
    实现 auto-regressive 循环预测
    """
    model.eval() # 切换到评估模式
    
    # 1. 编码源序列 (只需要做一次)
    memory = model.encode(src, src_mask)
    
    # 2. 初始化解码器输入 (一开始只有 BOS token)
    # 形状: (1, 1)
    tgt_tokens = torch.tensor([[start_symbol_idx]], dtype=torch.long, device=device)
    
    # 3. 开始循环预测
    for _ in range(max_len - 1):
        # 3a. 创建目标序列的 "future" 掩码
        tgt_len = tgt_tokens.size(1)
        tgt_mask = subsequent_mask(tgt_len).to(device)
        
        # 3b. 解码
        # out 形状: (1, current_len, d_model)
        out = model.decode(memory, src_mask, tgt_tokens, tgt_mask)
        
        # 3c. 获取最后一个词的 logits
        # prob 形状: (1, tgt_vocab_size)
        prob = model.generator(out[:, -1])
        
        # 3d. 选出概率最高的词
        # next_word_idx 形状: (1)
        next_word_idx = torch.argmax(prob, dim=-1)
        
        # 3e. 将预测的词拼接到 tgt_tokens
        tgt_tokens = torch.cat(
            [tgt_tokens, next_word_idx.unsqueeze(0)], 
            dim=1
        )
        
        # 3f. (关键) 检查是否预测了 EOS
        if next_word_idx.item() == end_symbol_idx:
            break
            
    return tgt_tokens

# --- 3. 主执行函数 ---

def translate(model, sentence, vocabs, tokenizers, device):
    """
    完整的翻译流程
    """
    # 1. 获取特殊 token 的索引
    src_vocab = vocabs[SRC_LANG]
    tgt_vocab = vocabs[TGT_LANG]
    pad_idx = src_vocab[PAD_TOKEN]
    bos_idx = src_vocab[BOS_TOKEN]
    eos_idx = src_vocab[EOS_TOKEN]
    
    # 2. (源) 分词和数值化
    src_tokens = [bos_idx] + src_vocab(tokenizers[SRC_LANG](sentence)) + [eos_idx]
    src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # 3. (源) 创建掩码
    src_mask = (src_tensor != pad_idx).unsqueeze(1).unsqueeze(2).to(device)
    
    # 4. (目标) 运行 Greedy Decode
    tgt_token_indices = greedy_decode(
        model, 
        src_tensor, 
        src_mask, 
        max_len=100, 
        start_symbol_idx=bos_idx, 
        end_symbol_idx=eos_idx, 
        device=device
    )
    
    # 5. (目标) 将索引转换回单词
    # 移除 BOS (第0个)
    tgt_tokens = tgt_vocab.lookup_tokens(tgt_token_indices.flatten().tolist())[1:]
    
    # 组合成句子
    return " ".join(tgt_tokens).replace(" <eos>", "")

def main():
    print("Loading saved model and vocabs...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载词汇表
    # (确保 vocabs.pt 和你的脚本在同一个文件夹)
    try:
        vocabs = torch.load('vocabs.pt')
    except FileNotFoundError:
        print("错误: 找不到 'vocabs.pt'。")
        print("请先成功运行 train.py 来生成 vocabs.pt 文件。")
        return
        
    src_vocab_size = len(vocabs[SRC_LANG])
    tgt_vocab_size = len(vocabs[TGT_LANG])
    
    # 2. 重新实例化模型结构
    model = EncoderDecoderModel(
        src_vocab=src_vocab_size,
        tgt_vocab=tgt_vocab_size,
        d_model=D_MODEL,
        N=N_LAYERS,
        h=H,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(device)
    
    # 3. 加载模型权重
    try:
        model.load_state_dict(torch.load('transformer_model.pt'))
    except FileNotFoundError:
        print("错误: 找不到 'transformer_model.pt'。")
        print("请先成功运行 train.py 来生成 transformer_model.pt 文件。")
        return
        
    print("Model loaded successfully.")
    
    # 4. 加载分词器
    tokenizers = {
        SRC_LANG: get_tokenizer('basic_english'),
        TGT_LANG: get_tokenizer('basic_english')
    }

    # 5. 启动交互式翻译
    while True:
        sentence = input("\n请输入一句德语 (或输入 'q' 退出): \n> ")
        if sentence.lower() == 'q':
            break
            
        translation = translate(model, sentence, vocabs, tokenizers, device)
        print(f"\n翻译结果:\n> {translation}")

if __name__ == "__main__":
    main()