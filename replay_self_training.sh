# | Dataset    | Value 1 | Value 2 | Value 3 |
# |------------|---------|---------|---------|
# | MNIST      | 0.1     | 0.075   | 0.0075  |
# | Eurosat    | 0.05    | 0.0075  |         |
# | Atari      | 0.1     | 0.05    | 0.0075  |
# | CIFAR 5exp | 0.075   | 0.05    | 0.0075  |
# | CIFAR 10exp| 0.1     | 0.05    | 0.025   |

echo "Launching runs..."
# OK
# nohup python main.py --cuda_id 3 --logdir ./logs/ST_SplitMNIST -tp 50 -ep -p 5 -sn Replay > run.log 2>&1 &
# nohup python main.py --cuda_id 3 --logdir ./logs/ST_SplitMNIST -tp 50 -ep -p 5 -sn Replay -ew 0.1 -stcm > run1.log 2>&1 &
# nohup python main.py --cuda_id 3 --logdir ./logs/ST_SplitMNIST -tp 50 -ep -p 5 -sn Replay -ew 0.075 -stcm > run2.log 2>&1 &
# nohup python main.py --cuda_id 3 --logdir ./logs/ST_SplitMNIST -tp 50 -ep -p 5 -sn Replay -ew 0.0075 -stcm > run3.log 2>&1 &

# OK
# nohup python main.py --cuda_id 3 --logdir ./logs/ST_EuroSAT -dn EuroSAT -ts 256 -es 256 -tp 200 -lr 0.001 -ep -p 10 -sn Replay -ms 5000 > run4.log 2>&1 &
# nohup python main.py --cuda_id 1 --logdir ./logs/ST_EuroSAT -dn EuroSAT -ts 256 -es 256 -tp 200 -lr 0.001 -ep -p 10 -sn Replay -ms 5000 -stcm --ent_weight 0.05 > run5.log 2>&1 &
# nohup python main.py --cuda_id 3 --logdir ./logs/ST_EuroSAT -dn EuroSAT -ts 256 -es 256 -tp 200 -lr 0.001 -ep -p 10 -sn Replay -ms 5000 -stcm --ent_weight 0.0075 > run6.log 2>&1 &

# OK
# nohup python main.py --cuda_id 1 --logdir ./logs/ST_Atari --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name Replay -ms 10000 --dataset_name Atari > run7.log 2>&1 &
# nohup python main.py --cuda_id 1 --logdir ./logs/ST_Atari --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name Replay -ms 10000 --dataset_name Atari -stcm --ent_weight 0.1 > run8.log 2>&1 &
# nohup python main.py --cuda_id 3 --logdir ./logs/ST_Atari --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name Replay -ms 10000 --dataset_name Atari -stcm --ent_weight 0.05 > run9.log 2>&1 &
# nohup python main.py --cuda_id 3 --logdir ./logs/ST_Atari --early_stopping --patience 10 --train_mb_size 256 --eval_mb_size 256 --train_epochs 200 -lr 0.0005 --strategy_name Replay -ms 10000 --dataset_name Atari -stcm --ent_weight 0.0075 > run10.log 2>&1 &

# OK
# nohup python main.py --cuda_id 0 --logdir ./logs/ST_SplitCIFAR100_5 -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Replay -ms 5000 > run11.log 2>&1 &
# nohup python main.py --cuda_id 0 --logdir ./logs/ST_SplitCIFAR100_5 -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Replay -ms 5000 -stcm --ent_weight 0.075 > run12.log 2>&1 &
# nohup python main.py --cuda_id 1 --logdir ./logs/ST_SplitCIFAR100_5 -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Replay -ms 5000 -stcm --ent_weight 0.05 > run13.log 2>&1 &
# nohup python main.py --cuda_id 1 --logdir ./logs/ST_SplitCIFAR100_5 -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Replay -ms 5000 -stcm --ent_weight 0.0075 > run14.log 2>&1 &

# OK
# nohup python main.py --cuda_id 2 --logdir ./logs/ST_SplitCIFAR100_10 -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Replay -ms 5000 > run15.log 2>&1 &
# nohup python main.py --cuda_id 2 --logdir ./logs/ST_SplitCIFAR100_10 -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Replay -ms 5000 -stcm --ent_weight 0.1 > run16.log 2>&1 &
# nohup python main.py --cuda_id 2 --logdir ./logs/ST_SplitCIFAR100_10 -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Replay -ms 5000 -stcm --ent_weight 0.05 > run17.log 2>&1 &
# nohup python main.py --cuda_id 0 --logdir ./logs/ST_SplitCIFAR100_10 -ts 256 -es 256 -tp 200 -lr 0.0001 -ep -p 10 -dn SplitCIFAR100 -sn Replay -ms 5000 -stcm --ent_weight 0.025 > run18.log 2>&1 &