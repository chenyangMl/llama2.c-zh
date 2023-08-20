
# 单卡
python -m train.py --compile=False --eval_iters=10 --batch_size=8
后台运行: nohup python -m train.py --compile=False --eval_iters=10 --batch_size=8 > log.out &

# 多卡
CUDA_VISIBLE_DEVICES=0,1  torchrun --master-port 29402 --nproc_per_node=2 train.py

更多方式参考 train.py 