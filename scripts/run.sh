# 设置固定的随机种子，确保可复现性 
SEED=42

# 运行训练脚本
# 我们使用 python -u 来禁用输出缓冲, 这样可以立即看到 print() 的内容
echo "Starting training with seed $SEED..."
python -u train.py --seed $SEED